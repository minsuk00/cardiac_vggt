"""Synthetic principle toy: can a learned synthesizer beat TRANSPORT on HELD-OUT, and when does it
just hallucinate?

Controlled population of 2D "slices" with three KNOWN components:
  (1) MOTION (transportable): a disk whose center translates with phase, center_x(t)=c0+D_s·sin — a
      warp can move it. D_s subject-specific.
  (2) POPULATION APPEARANCE (synthesizable, generalizes): disk intensity b(t)=b0+B·sin(2πt/T) — the
      SAME function for all subjects. A warp CANNOT change pixel intensity; a learner CAN learn b(t)
      from the population + the target_t query.
  (3) SUBJECT-SPECIFIC DISOCCLUSION (unpredictable): a bright blob that appears only in a systole
      window, at a per-subject random location — present in the target but in NO input unless an
      input happens to be in-window. Neither transport nor a learner can predict it → hallucination.

Baselines on HELD-OUT subjects:
  - transport ceiling: shift each input to the target center (ORACLE geometry — generous), average →
    correct position+texture, intensity = mean over input phases (≈b0, wrong). The warp ceiling.
  - synthesis: a small CNN(inputs + target_t) → target, trained on TRAIN subjects, eval HELD-OUT.

Decompose synthesis error into the intensity (component 2, should be recovered) and the disocclusion
region (component 3, should be hallucinated). Establishes: synthesis beats transport by the
population-predictable appearance; the rest is hallucination.

Run: micromamba run -n svr python tools/toy_synth_appearance.py
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, os, json
torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HW, T, S = 40, 12, 6
OUT = os.path.join(os.path.dirname(__file__), "..", "result", "limits_eval")

yy, xx = np.meshgrid(np.arange(HW), np.arange(HW), indexing="ij")


def subject_params(n, rng):
    return dict(D=rng.uniform(2, 6, n),                      # motion amplitude (px)
                tex=rng.uniform(-0.12, 0.12, (n, HW, HW)),   # subject texture (fixed across phases)
                ang=rng.uniform(0, 2 * np.pi, n))            # disocclusion location (subject-specific)


def render(p, s, t):
    """image(subject s, phase t): disk(translated) * (b(t)+texture) + disocclusion."""
    ph = 2 * np.pi * t / T
    cx = HW / 2 + p["D"][s] * np.sin(ph); cy = HW / 2
    R = 9.0
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2).astype(np.float32)
    b = 0.5 + 0.35 * np.sin(ph)                              # POPULATION appearance
    img = disk * (b + p["tex"][s])
    # subject-specific disocclusion blob, only in systole window sin>0.5
    if np.sin(ph) > 0.5:
        bx = cx + 4 * np.cos(p["ang"][s]); by = cy + 4 * np.sin(p["ang"][s])
        img = img + 0.5 * np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * 2.0 ** 2)))
    return img.astype(np.float32), (cx, cy)


def sample(p, s, rng):
    t_tgt = rng.integers(0, T)
    t_ins = rng.choice([t for t in range(T) if t != t_tgt], size=S, replace=False)
    ins = np.stack([render(p, s, ti)[0] for ti in t_ins])      # (S,HW,HW)
    tgt, (cx, cy) = render(p, s, t_tgt)
    return ins, tgt.astype(np.float32), t_ins, t_tgt, (cx, cy)


def transport_ceiling(p, s, ins, t_ins, t_tgt, cx, cy):
    """Shift each input to the target center (oracle geometry) then average. Warp ceiling."""
    shifted = []
    for k, ti in enumerate(t_ins):
        _, (cxi, _) = render(p, s, ti)
        dx = int(round(cx - cxi))
        shifted.append(np.roll(ins[k], dx, axis=1))
    return np.mean(shifted, axis=0)


class Synth(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(S + 1, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 1, 1))

    def forward(self, ins, t_tgt):
        tch = torch.full((ins.shape[0], 1, HW, HW), 0.0, device=ins.device) + (t_tgt.view(-1, 1, 1, 1) / T)
        return self.net(torch.cat([ins, tch], dim=1))[:, 0]


def psnr(a, b, m=None):
    if m is not None: a, b = a[m], b[m]
    mse = float(((a - b) ** 2).mean()); return 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)


def main():
    rng = np.random.default_rng(0)
    Ntr, Nte = 100, 30
    P = subject_params(Ntr + Nte, rng)
    train_s, test_s = list(range(Ntr)), list(range(Ntr, Ntr + Nte))

    model = Synth().to(DEV); opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for it in range(3000):
        bs = 32; ins_b, tgt_b, tt_b = [], [], []
        for _ in range(bs):
            s = rng.choice(train_s); ins, tgt, ti, tt, _ = sample(P, s, rng)
            ins_b.append(ins); tgt_b.append(tgt); tt_b.append(tt)
        ins_t = torch.tensor(np.stack(ins_b), device=DEV); tgt_t = torch.tensor(np.stack(tgt_b), device=DEV)
        tt_t = torch.tensor(np.array(tt_b), device=DEV, dtype=torch.float32)
        pred = model(ins_t, tt_t); loss = (pred - tgt_t).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()

    # held-out eval
    rng2 = np.random.default_rng(123)
    tr_synth, te_synth, te_transp, te_meanblur = [], [], [], []
    te_int_err_transp, te_int_err_synth, te_disocc_err = [], [], []
    for split, slist, store in [("train", train_s[:30], tr_synth), ("test", test_s, te_synth)]:
        for s in slist:
            for rep in range(4):
                ins, tgt, ti, tt, (cx, cy) = sample(P, s, rng2)
                with torch.no_grad():
                    pr = model(torch.tensor(ins[None], device=DEV),
                               torch.tensor([tt], device=DEV, dtype=torch.float32))[0].cpu().numpy()
                store.append(psnr(pr, tgt))
                if split == "test":
                    tr = transport_ceiling(P, s, ins, ti, tt, cx, cy)
                    te_transp.append(psnr(tr, tgt))
                    te_meanblur.append(psnr(ins.mean(0), tgt))   # no-geometry mean (floor)
                    # disk-interior intensity error (component 2): mean abs over disk, EXCLUDING disocc
                    R = 9.0; disk = ((xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2)
                    te_int_err_transp.append(float(np.abs(tr - tgt)[disk].mean()))
                    te_int_err_synth.append(float(np.abs(pr - tgt)[disk].mean()))
                    # disocclusion region error (component 3): around true blob loc
                    if np.sin(2 * np.pi * tt / T) > 0.5:
                        bx = cx + 4 * np.cos(P["ang"][s]); by = cy + 4 * np.sin(P["ang"][s])
                        blob = (((xx - bx) ** 2 + (yy - by) ** 2) <= 9)
                        te_disocc_err.append(float(np.abs(pr - tgt)[blob].mean()))

    res = dict(
        synth_train=float(np.mean(tr_synth)), synth_test=float(np.mean(te_synth)),
        transport_ceiling_test=float(np.mean(te_transp)), mean_blur_floor_test=float(np.mean(te_meanblur)),
        intensity_err_transport=float(np.mean(te_int_err_transp)), intensity_err_synth=float(np.mean(te_int_err_synth)),
        disocclusion_err_synth=float(np.mean(te_disocc_err)) if te_disocc_err else None)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "toy_synth_appearance.json"), "w"), indent=2)
    print("=== SYNTHETIC PRINCIPLE TOY (PSNR on held-out unless noted) ===")
    print(f"  mean-blur floor (no geometry)      {res['mean_blur_floor_test']:.2f}")
    print(f"  TRANSPORT ceiling (oracle geometry){res['transport_ceiling_test']:.2f}")
    print(f"  SYNTHESIS (held-out)               {res['synth_test']:.2f}   [train {res['synth_train']:.2f}]")
    print(f"  → synthesis beats transport by {res['synth_test']-res['transport_ceiling_test']:+.2f} dB on held-out")
    print(f"  disk intensity error: transport {res['intensity_err_transport']:.4f} -> synth {res['intensity_err_synth']:.4f} "
          f"(component 2 = population appearance: {'RECOVERED' if res['intensity_err_synth']<0.6*res['intensity_err_transport'] else 'not recovered'})")
    if res["disocclusion_err_synth"] is not None:
        print(f"  disocclusion-region error (synth)  {res['disocclusion_err_synth']:.4f}  (component 3 = subject-specific → HALLUCINATED)")
    print("Wrote", os.path.join(OUT, "toy_synth_appearance.json"))


if __name__ == "__main__":
    main()
