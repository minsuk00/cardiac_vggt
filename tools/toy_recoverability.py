"""How much of the target-phase appearance is RECOVERABLE (bounds the synthesis ceiling), on REAL data.

The decoder's held-out ceiling is bounded by how predictable the target-phase volume is from the
available information. We measure cheap UPPER/LOWER bounds (no training) on held-out (val) subjects,
motion PSNR:

  - identity / transport (~16.8 / ~21)         : known floors (transport = warp ceiling, docs/19)
  - population_template[t] = mean over TRAIN subjects of canonical phases[t]
        = the "average heart at phase t" — the subject-AGNOSTIC appearance a population prior gives.
        A decoder that only learns the population mean cannot beat this (the skeptic's beat-the-mean).
  - subject_mean = mean over all phases of the HELD-OUT subject's own volume
        = the subject's static anatomy, no phase-specific appearance (a conditional-mean baseline).
  - subject_temporal_interp = 0.5*(phases[t-1]+phases[t+1]) of the held-out subject (cyclic)
        = appearance recoverable from the subject's OWN dense adjacent phases. This is an UPPER BOUND
          on what ANY synthesizer could recover, because it uses FULL adjacent-phase volumes — far
          more than the ~10 sparse single slices the real model sees. If even this is low, the sparse
          synthesizer cannot be a breakthrough.
  - oracle = V_gt (35, by construction the ceiling)

Reading: a decoder lands between max(transport, population_template) and subject_temporal_interp.
If subject_temporal_interp ≫ population_template, subject-specific temporal info matters (synthesis
has headroom). If subject_temporal_interp ≈ transport, the appearance is NOT recoverable → no
breakthrough.

Run: micromamba run -n svr python tools/toy_recoverability.py
"""
import os, sys, json
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from eval_variants_matrix import build_dataset, build_batch, GRID_SHAPE, NUM_SLICES, DATA_ROOT, SPLIT_FILE
from data.datasets.mri_dataset import MRIDataset
from loss import compute_motion_mask
from omegaconf import OmegaConf
T = 12; D, H, W = GRID_SHAPE
OUT = os.path.join(REPO, "result", "limits_eval")


def psnr(a, b, m):
    a, b = a[m], b[m]; mse = float(((a - b) ** 2).mean()); return 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)


def make_ds(split):
    conf = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True, "rescale_aug": False,
                             "landscape_check": False, "augs": {"scales": [1.0, 1.0]}})
    return MRIDataset(conf, DATA_ROOT, split=split, split_file=SPLIT_FILE, mode="dynamic",
                      mri_mode="axial", num_slices=NUM_SLICES, target_size=518)


def get_phases(ds, seq):
    """Return (T,D,H,W) canonical phase bundle for subject seq%len, splat-order, on cuda."""
    data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
    ph = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).cuda()  # (T,D,H,W)
    return ph


def main():
    dev = "cuda"
    tr = make_ds("train"); va = make_ds("val")
    n_tr_subj = len(tr.subjects); n_va_subj = len(va.subjects)

    # population template: mean over a subset of TRAIN subjects, per phase (canonical, aligned grid)
    N_TEMPLATE = min(60, n_tr_subj)
    acc = torch.zeros((T, D, H, W), device=dev); cnt = 0
    for s in range(N_TEMPLATE):
        acc += get_phases(tr, s); cnt += 1
    pop_template = acc / cnt                                    # (T,D,H,W)
    print(f"population template built from {cnt} train subjects")

    rows = {"population_template": [], "subject_mean": [], "subject_temporal_interp": [],
            "identity_proxy": []}
    for vs in range(n_va_subj):
        ph = get_phases(va, vs)                                 # (T,D,H,W) held-out subject
        mmask = compute_motion_mask(ph.unsqueeze(0))[0].cpu().numpy()
        if not mmask.any():
            continue
        ph_np = ph.cpu().numpy(); subj_mean = ph_np.mean(0)
        for t in range(T):
            gt = ph_np[t]
            rows["population_template"].append(psnr(pop_template[t].cpu().numpy(), gt, mmask))
            rows["subject_mean"].append(psnr(subj_mean, gt, mmask))
            interp = 0.5 * (ph_np[(t - 1) % T] + ph_np[(t + 1) % T])
            rows["subject_temporal_interp"].append(psnr(interp, gt, mmask))

    summary = {k: float(np.mean(v)) for k, v in rows.items() if v}
    summary["transport_warp_ceiling"] = 21.0     # docs/19
    summary["identity_floor"] = 16.8
    summary["trained_model"] = 20.6
    summary["oracle"] = 35.0
    summary["n_val_subjects"] = n_va_subj
    json.dump(summary, open(os.path.join(OUT, "toy_recoverability.json"), "w"), indent=2)

    print("\n=== APPEARANCE RECOVERABILITY (held-out val, motion PSNR) ===")
    print(f"  identity floor                  {summary['identity_floor']:.1f}")
    print(f"  trained model                   {summary['trained_model']:.1f}")
    print(f"  transport / warp ceiling        {summary['transport_warp_ceiling']:.1f}")
    print(f"  population template (avg heart) {summary['population_template']:.2f}  <- subject-agnostic appearance (beat-the-mean bar)")
    print(f"  subject all-phase mean          {summary['subject_mean']:.2f}  <- subject anatomy, no phase appearance")
    print(f"  subject temporal-interp (UB)    {summary['subject_temporal_interp']:.2f}  <- UPPER BOUND on recoverable appearance (uses dense adjacent phases)")
    print(f"  oracle (V_gt)                   {summary['oracle']:.1f}")
    print("\nReading: a sparse-slice synthesizer is bounded ABOVE by subject_temporal_interp")
    print("(which has far more info than the model). population_template = the beat-the-mean bar.")
    print("Wrote", os.path.join(OUT, "toy_recoverability.json"))


if __name__ == "__main__":
    main()
