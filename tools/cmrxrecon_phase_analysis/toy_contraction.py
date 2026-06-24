"""Minimal toy that reproduces VGGT-MRI's flat-EF failure and tests the fixes.

Essence of the real task:
  - A "ventricle" = a stack of Z slices. Per patient: a contraction amplitude a~U[.3,.9]
    (= EF target), per-z baseline radius r0(z), and an apex->base contraction gradient g(z).
    radius(z,t) = r0(z) * (1 - a*g(z)*s(t)),  s(t) a fixed waveform, s(0)=0 (ED), peak at ES.
  - Volume(t) = sum_z radius(z,t)^2 ;  EF = (maxV - minV)/maxV  (per patient, ~= a).
  - Observation = a scattered slice at (z_j, t_j): the model sees (z_j, radius_j) but is
    BLIND to the phase t_j (mirrors use_t=false). S scattered obs per sample.
  - Query: reconstruct radius(z, target_t) for ALL z -> volume -> EF.

Conditioning schemes compared (the only thing that changes):
  C0  : query=(z, target_t index); blind input phase.            [= current VGGT recipe]
  Clab: query=(z, target_t index); input phase LABELED.          [control: is blindness it?]
  Ccov: C0 but ONE input slot is guaranteed at target_t (blind). [coverage fix, keep index]
  B   : NO target_t index; one input slot is the REFERENCE at target_t (flagged); query=(z).
        The query phase is DEFINED by the reference slice.        [frame-0 camera-token design]
  A   : fixed target_t = ES only (one-phase model).              [separate-models baseline]

Metric: slope of pred-EF vs true-EF across a held-out test set (1.0 = recovers per-patient
amplitude; 0 = regresses to the cohort mean). Trains in seconds on GPU.
"""
import argparse, numpy as np, torch, torch.nn as nn

Z = 8; T = 12
S = 8  # default; overridden per-run in the coverage sweep


def waveform(device):
    t = torch.arange(T, device=device).float()
    s = torch.sin(np.pi * t / (T - 1)).clamp(min=0)          # 0 at t0, peak mid, fixed timing
    s = s / s.max()
    return s                                                  # (T,)


def gen(n, device, seed):
    g_rng = torch.Generator(device=device).manual_seed(seed)
    a = 0.3 + 0.6 * torch.rand(n, generator=g_rng, device=device)            # amplitude/EF (n,)
    r0 = 0.8 + 0.4 * torch.rand(n, Z, generator=g_rng, device=device)        # baseline radius (n,Z)
    zc = torch.linspace(1.0, 0.4, Z, device=device)                          # base->apex gradient
    s = waveform(device)                                                     # (T,)
    # radius(n,Z,T) = r0*(1 - a*g(z)*s(t))
    rad = r0[:, :, None] * (1 - a[:, None, None] * zc[None, :, None] * s[None, None, :])
    return a, r0, rad, g_rng                                                 # rad: (n,Z,T)


def true_ef(rad):                                            # rad (n,Z,T) -> EF (n,)
    V = (rad ** 2).sum(1)                                    # (n,T)
    return (V.max(1).values - V.min(1).values) / V.max(1).values


class SetNet(nn.Module):
    """DeepSets over input slices + a query (z, phase-cond) -> radius. H=64."""
    def __init__(self, scheme):
        super().__init__()
        self.scheme = scheme
        in_obs = 2 + (1 if scheme in ("Clab",) else 0) + (1 if scheme == "B" else 0)  # z,val[,t][,is_ref]
        self.enc = nn.Sequential(nn.Linear(in_obs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        q_dim = 1 + (T if scheme in ("C0", "Clab", "Ccov", "A") else 0) + (1 if scheme == "B" else 0)
        self.dec = nn.Sequential(nn.Linear(64 + q_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, obs, qz, tt_onehot=None, ref_val=None):
        h = self.enc(obs).mean(1)                            # (B,64) permutation-invariant
        q = [qz]
        if tt_onehot is not None: q.append(tt_onehot)
        if ref_val is not None: q.append(ref_val)
        return self.dec(torch.cat([h] + ([] if len(q) == 1 else q[1:]) + [q[0]], -1)).squeeze(-1)


def make_batch(rad, scheme, target_t, device, rng):
    """rad (n,Z,T). Returns obs (n*Z_query, S, obs_dim), qz, tt_onehot, ref_val, y_true."""
    n = rad.shape[0]
    # scattered observations: S random (z,t)
    zt_z = torch.randint(0, Z, (n, S), generator=rng, device=device)
    zt_t = torch.randint(0, T, (n, S), generator=rng, device=device)
    if scheme == "Ccov" or scheme == "B":
        zt_t[:, 0] = target_t                                # force slot 0 to target phase
    val = rad[torch.arange(n)[:, None], zt_z, zt_t]          # (n,S)
    znorm = zt_z.float() / (Z - 1) * 2 - 1
    feats = [znorm[..., None], val[..., None]]
    if scheme == "Clab":
        feats.append(zt_t.float()[..., None] / (T - 1) * 2 - 1)   # labeled input phase
    if scheme == "B":
        is_ref = torch.zeros(n, S, 1, device=device); is_ref[:, 0] = 1.0
        feats.append(is_ref)
    obs = torch.cat(feats, -1)                               # (n,S,obs_dim)
    # query EVERY z at target_t
    qz = (torch.arange(Z, device=device).float() / (Z - 1) * 2 - 1)
    qz = qz[None, :].expand(n, Z).reshape(n * Z, 1)
    obs_q = obs[:, None].expand(n, Z, S, obs.shape[-1]).reshape(n * Z, S, obs.shape[-1])
    tt_onehot = None
    if scheme in ("C0", "Clab", "Ccov", "A"):
        oh = torch.zeros(n, T, device=device); oh[:, target_t] = 1.0
        tt_onehot = oh[:, None].expand(n, Z, T).reshape(n * Z, T)
    ref_val = None
    if scheme == "B":
        rv = val[:, 0:1]                                     # reference slot's value (at target phase, some z)
        ref_val = rv[:, None].expand(n, Z, 1).reshape(n * Z, 1)
    y = rad[:, :, target_t].reshape(n * Z)                  # true radius at target phase, all z
    return obs_q, qz, tt_onehot, ref_val, y


def train_eval(scheme, device, epochs=400):
    a_tr, _, rad_tr, _ = gen(800, device, seed=0)
    a_te, _, rad_te, _ = gen(200, device, seed=1)
    rng = torch.Generator(device=device).manual_seed(123)
    net = SetNet(scheme).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    phases = [6] if scheme == "A" else list(range(T))
    for ep in range(epochs):
        net.train()
        tt = phases[torch.randint(0, len(phases), (1,), generator=rng, device=device).item()]
        obs, qz, oh, rv, y = make_batch(rad_tr, scheme, tt, device, rng)
        pred = net(obs, qz, oh, rv)
        loss = (pred - y).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # eval: reconstruct full curve per test patient -> EF
    net.eval()
    with torch.no_grad():
        n = rad_te.shape[0]
        predV = torch.zeros(n, T, device=device)
        for tt in range(T):
            obs, qz, oh, rv, y = make_batch(rad_te, scheme, tt, device, rng)
            p = net(obs, qz, oh, rv).reshape(n, Z)
            predV[:, tt] = (p ** 2).sum(1)
        if scheme == "A":      # one-phase model: EF undefined; report ES radius accuracy instead
            ef_pred = torch.full((n,), float('nan'))
        else:
            ef_pred = (predV.max(1).values - predV.min(1).values) / predV.max(1).values
    ef_true = true_ef(rad_te).cpu().numpy()
    ef_pred = ef_pred.cpu().numpy()
    if np.isnan(ef_pred).all():
        return dict(scheme=scheme, slope=float('nan'), r=float('nan'),
                    pred_mean=float('nan'), true_mean=float(ef_true.mean()))
    m = ~np.isnan(ef_pred)
    slope, b = np.polyfit(ef_true[m], ef_pred[m], 1)
    r = np.corrcoef(ef_true[m], ef_pred[m])[0, 1]
    return dict(scheme=scheme, slope=float(slope), r=float(r),
                pred_mean=float(ef_pred[m].mean()), true_mean=float(ef_true.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global S
    descs = {"C0": "target_t index, BLIND input phase [current]",
             "Clab": "target_t index, LABELED input phase",
             "Ccov": "target_t index + 1 slot AT target (blind)",
             "B": "frame-0=target ref slice, NO index",
             "A": "fixed-phase (ES only) model"}

    # ── (1) head-to-head at fixed coverage S=8 ──
    S = 8
    print(f"device={device}  toy Z={Z} T={T} S={S}  (EF slope: 1=recover per-patient, 0=regress-to-mean)\n")
    print(f"{'scheme':6s} {'desc':44s} {'slope':>6s} {'r':>6s} {'predEF':>7s} {'trueEF':>7s}")
    for sch in ["C0", "Clab", "Ccov", "B", "A"]:
        o = train_eval(sch, device, epochs=a.epochs)
        f = lambda v: f"{v:.2f}" if v == v else "n/a"
        print(f"{sch:6s} {descs[sch]:44s} {f(o['slope']):>6s} {f(o['r']):>6s} {f(o['pred_mean']):>7s} {o['true_mean']:>7.2f}")

    # ── (2) coverage sweep: EF-slope vs S for the 3 informative schemes ──
    S_list = [2, 3, 4, 6, 8, 12, 20]
    schemes = ["C0", "Ccov", "B"]
    curves = {s: [] for s in schemes}
    print("\ncoverage sweep (EF-slope vs S):")
    print("S    " + "  ".join(f"{s:>6s}" for s in schemes))
    for sval in S_list:
        S = sval
        row = {}
        for sch in schemes:
            row[sch] = train_eval(sch, device, epochs=a.epochs)["slope"]
            curves[sch].append(row[sch])
        print(f"{sval:<4d} " + "  ".join(f"{row[s]:>6.2f}" for s in schemes))

    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        lbl = {"C0": "C0: target_t index (current)", "Ccov": "Ccov: index + 1 target slot",
               "B": "B: reference-slice (frame-0)"}
        col = {"C0": "#cc3311", "Ccov": "#ee9900", "B": "#003366"}
        for s in schemes:
            ax.plot(S_list, curves[s], "-o", color=col[s], label=lbl[s])
        ax.axhline(1.0, color="green", ls=":", lw=1, label="perfect (slope=1)")
        ax.axhline(0.0, color="gray", ls=":", lw=1, label="regress-to-mean (slope=0)")
        ax.set_xlabel("S = number of scattered input slices (coverage)")
        ax.set_ylabel("pred-EF vs true-EF slope")
        ax.set_title("Toy: contraction-amplitude recovery vs coverage & conditioning")
        ax.legend(fontsize=8); ax.set_ylim(-0.1, 1.1); fig.tight_layout()
        fig.savefig(a.plot, dpi=120); print(f"\nplot -> {a.plot}")


if __name__ == "__main__":
    main()
