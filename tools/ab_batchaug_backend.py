"""A/B the batchaug pytorch vs triton backend on the REAL gpu_aug pipeline.

Context: `training/data/gpu_aug.py` forces `batchaug.set_backend("pytorch")`.
torch 2.13 now ships the matched triton 3.7.1, so the triton backend is usable.
This measures whether switching is actually worth it.

From reading batchaug: `triton/__init__.py` does `from ..pytorch import *` and
overrides ONLY intensity transforms (`triton/geometric/__init__.py` is empty).
Our tiers use RandAffined (identical in both backends) + RandAdjustContrastd +
RandBiasFieldd (triton kernels). So only 2 of 3 active transforms can differ,
and the expensive spatial one is not among them.

Measurement traps this script controls for (ALL of them produced wrong published
numbers before being caught — see docs/49):
  * gpu_aug resets the backend to pytorch at module import -> import it FIRST,
    and assert on each transform's __module__ (resolve_backend() is a tautology).
    An unbound "triton" arm silently runs pytorch and reports a clean null.
  * batchaug caches resolved transform classes -> clear before re-dispatch.
  * Per-process GPU clock drift exceeded the effect (pytorch's own median moved
    7.33 -> 5.93 ms between runs) -> INTERLEAVE the two arms in rounds.
  * The arms must be SEEDED-paired, not merely interleaved: sharing the global
    CUDA RNG gives each arm a different Bernoulli mask, so the backend-identical
    affine's on/off gate dominates the variance. Unseeded sd 1.139 ms, seeded
    0.062 ms. The unseeded design sat on its own detection threshold and
    reported a meaningless "null".

Result (2026-07-24, A40, 200 seeded rounds): triton is NOT faster — full pipeline
-0.048 ms +/- 0.0044 (0.993x, marginally slower); isolated at prob=1.0 -0.013 ms
+/- 0.0104 (0.990x, null). Note intensity-transform COST is not prob-gated: both
backends compute unconditionally and gate only the output via torch.where.

Run:  PYTHONPATH=training:. micromamba run -n svr python tools/ab_batchaug_backend.py
"""

import argparse
import statistics
import sys

import torch

sys.path.insert(0, "training")
sys.path.insert(0, ".")


# Our 3 active transforms: RandAffined (identical both backends) +
# RandAdjustContrastd + RandBiasFieldd (triton kernels). So a correctly-bound
# triton arm must show exactly 2 triton-backed transforms.
EXPECTED_TRITON = 2


class _Cfg:
    def __init__(self, tier):
        self.enable = True
        self.tier = tier


def build(tier, backend):
    """Return a Compose genuinely bound to `backend`, plus what it resolved to."""
    import data.gpu_aug as ga  # import FIRST: it calls set_backend("pytorch")
    import batchaug as _B

    for name in _B._TRANSFORM_NAMES:  # drop cached class resolutions
        _B.__dict__.pop(name, None)
    _B.set_backend(backend)

    ga._B = _B
    compose = ga.build_gpu_transforms(_Cfg(tier))

    # NOTE: `assert resolve_backend() == backend` would be a TAUTOLOGY — nothing
    # between set_backend() and here can change the global. The only real evidence
    # is which module each transform's class actually came from.
    mods = [(type(t).__name__, type(t).__module__) for t in compose.transforms]
    n_triton = sum(1 for _, m in mods if "triton" in m)
    expected = EXPECTED_TRITON if backend == "triton" else 0
    assert n_triton == expected, (
        f"{backend}: expected {expected} triton-backed transforms, got {n_triton}. "
        f"The backend did not bind — a 'triton' arm running pytorch silently "
        f"reports a clean null. Modules: {mods}")
    return compose, n_triton, mods


def make_inputs(B, seed):
    """Mirror gpu_augment_batch: phases (B,T,D,H,W) f32, mask (B,1,D,H,W) f32."""
    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(seed)
    phases = torch.rand((B, 12, 12, 256, 256), generator=g).to(dev)
    mask = (torch.rand((B, 12, 256, 256), generator=g) > 0.5).float().to(dev).unsqueeze(1)
    return phases, mask


def time_once(compose, phases, mask, seed):
    # SEEDED PAIRING is mandatory. Both arms must get IDENTICAL random draws in a
    # given round; otherwise the (backend-identical) affine's Bernoulli gate flips
    # independently between arms and injects ~1.6 ms of noise that swamps the
    # effect. Measured: unseeded sd 1.139 ms vs seeded 0.062 ms (18x tighter).
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    d = {"phases": phases.clone(), "content_mask": mask.clone()}
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    compose(d)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="moderate",
                    choices=["conservative", "moderate", "aggressive"])
    ap.add_argument("--rounds", type=int, default=60, help="interleaved A/B rounds")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1, help="B in (B,T,D,H,W)")
    args = ap.parse_args()

    import triton
    print(f"torch {torch.__version__} | triton {triton.__version__}")
    print(f"GPU {torch.cuda.get_device_name(0)} | tier={args.tier} "
          f"rounds={args.rounds} warmup={args.warmup}")
    print(f"phases ({args.batch}, 12, 12, 256, 256) [B,T,D,H,W]\n")

    composes = {}
    for b in ("pytorch", "triton"):
        c, n_triton, mods = build(args.tier, b)
        composes[b] = c
        print(f"{b:8s}: {len(mods)} transforms, {n_triton} from triton kernels")
        for n, m in mods:
            print(f"           {n:24s} <- {m}")
    print()

    phases, mask = make_inputs(args.batch, args.seed)

    for i in range(args.warmup):  # warm both arms together
        for b in ("pytorch", "triton"):
            time_once(composes[b], phases, mask, 10_000_000 + i)

    # INTERLEAVED: alternate arms each round so any clock drift hits both equally
    times = {"pytorch": [], "triton": []}
    for r in range(args.rounds):
        order = ("pytorch", "triton") if r % 2 == 0 else ("triton", "pytorch")
        for b in order:
            times[b].append(time_once(composes[b], phases, mask, args.seed * 100_000 + r))

    print("--- interleaved timing ---")
    for b in ("pytorch", "triton"):
        t = times[b]
        print(f"  {b:8s} median {statistics.median(t):7.3f} ms | mean {statistics.mean(t):7.3f} "
              f"| stdev {statistics.stdev(t):6.3f} | min {min(t):7.3f}")

    p, t = statistics.median(times["pytorch"]), statistics.median(times["triton"])
    # Genuinely paired now (same seed per round in both arms), so the per-round
    # difference cancels the shared randomness instead of just differencing two
    # independent samples.
    diffs = [a - b for a, b in zip(times["pytorch"], times["triton"])]
    md = statistics.mean(diffs)
    sem = statistics.stdev(diffs) / (len(diffs) ** 0.5)
    print(f"\n  SEEDED-paired mean diff (pytorch - triton): {md:+.3f} ms "
          f"+/- {sem:.4f} SEM  ({md / sem if sem else float('nan'):+.1f} sigma)")
    print(f"  median speedup: {p / t:.3f}x")

    # docs/47 measured S=10 -> 3518 ms. The active config is S=20
    # (mri_finetune.yaml img_nums:[20,20]), so the real step is ~2x longer and
    # the aug fraction below is a conservative OVER-estimate.
    STEP_MS = 3518.0
    print(f"\n  aug cost is {p / STEP_MS * 100:.3f}% of a {STEP_MS:.0f} ms train step (docs/47)")
    print(f"  switching would save {md / STEP_MS * 100:.4f}% of train time")

    print("\n--- numerical equivalence ---")
    outs = {}
    for b in ("pytorch", "triton"):
        c, _, _ = build(args.tier, b)   # build() asserts the binding (see above)
        ph, mk = make_inputs(args.batch, args.seed)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        outs[b] = c({"phases": ph.clone(), "content_mask": mk.clone()})["phases"].detach().float()
    a, b_ = outs["pytorch"], outs["triton"]
    d = (a - b_).abs()
    print(f"  bitwise equal : {torch.equal(a, b_)}")
    print(f"  max |diff|    : {d.max().item():.3e}")
    print(f"  mean |diff|   : {d.mean().item():.3e}")


if __name__ == "__main__":
    main()
