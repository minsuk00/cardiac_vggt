"""GPU generation (no rendering) for the 7-row sweep: CMRxRecon val subjects (in-distribution) +
regen of the one corrupt gated npz. Saves the same npz format as gated_model_sweep so the sbatch
CPU renderer (rerender_hires.py) produces GIFs identically. Run on the GPU node.

Adds CMRxRecon val (random_8_1_1 split) via MRIDataset.get_data -> canonical phase bundle, then the
same capture() (1frame, clean-ref, dynamic seed 72, 18 phases) used for the gated datasets.
Run: micromamba run -n svr env PYTHONPATH=training:. python -u tools/miitt_viz/gen_cmrxrecon_npz.py
"""
import os, sys, glob, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "training")
from inference.inference import load_rtfb_model_reference
from inference.run_cmrxrecon import build_mri_dataset
from inference.adapters import MIITTGatedAdapter
from tools.miitt_viz.gated_gather05_7row import capture

DEV = "cuda"; OUT = "result/gated_model_sweep"; SEED = 72; MP = 18
MODELS = [
    ("gather05", glob.glob("scratch/logs/216539845_*ftgather05*1frame*/ckpts/checkpoint_last.pt")[0]),
    ("control0", glob.glob("scratch/logs/216539845_*ftctrl_gather0*1frame*/ckpts/checkpoint_last.pt")[0]),
]
CMRX_SUBJS = [0, 7, 14, 21, 28]   # spread of val seq_indices (30 val subjects total)


def save_npz(base, cap):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    np.savez_compressed(base + ".npz", gt=cap["GT"], recon=cap["RE"], inp=cap["IN"], dvf=cap["DV"],
                        cov=cap["CO"], has_slot=np.array(cap["has_slot"]), ref_zmid=cap["z_mid"],
                        zbr=cap["rd"][:, 0], sop=np.array(cap["sop"]), applied_disp=cap["rd"],
                        per_phase_motion=np.array(cap["metr"]["motion"]),
                        per_phase_full=np.array(cap["metr"]["full"]),
                        per_phase_ssim=np.array(cap["metr"]["ssim"]))


def main():
    mri_ds, rcfg = build_mri_dataset()
    ids = [str(p).split("/")[-2] for p in mri_ds.subjects]   # CMRx subject IDs, indexed by seq
    cmrx = {}
    for s in CMRX_SUBJS:
        data = mri_ds.get_data(seq_index=s, img_per_seq=mri_ds.num_slices)
        cmrx[s] = (np.asarray(data["phases"]).astype(np.float32),
                   np.asarray(data["anatomy_bbox"]).astype(np.int64))
    print(f"cmrx val subjects: {[(s, ids[s]) for s in CMRX_SUBJS]}", flush=True)

    for mname, ckpt in MODELS:
        model = load_rtfb_model_reference(ckpt, refiner=False, device=DEV)
        for s in CMRX_SUBJS:
            pb = torch.from_numpy(cmrx[s][0]).to(DEV); bbox = cmrx[s][1]
            for tag, br in [("clean", False), ("normal", True)]:
                base = os.path.join(OUT, mname, "cmrxrecon", f"subj{s:02d}_{ids[s]}", f"{tag}_7row")
                if os.path.exists(base + ".npz"):
                    print(f"  skip {base}", flush=True); continue
                cap = capture(model, pb, bbox, br, rcfg, "1frame", True, seq_index=SEED, max_phases=MP)
                save_npz(base, cap)
                m = {k: float(np.nanmean(v)) for k, v in cap["metr"].items()}
                print(f"  [{mname} cmrx {ids[s]} {tag}] motion={m['motion']:.2f} full={m['full']:.2f}dB -> {base}.npz", flush=True)
        # regen the one corrupt gated npz (killed mid-write in the first sweep)
        if mname == "control0":
            base = os.path.join(OUT, "control0", "miitt", "Volunteer2", "normal_7row")
            if not os.path.exists(base + ".npz"):
                bnp, bbox = MIITTGatedAdapter("scratch/data/MIITT/nifti/Volunteer2/gated/sax/4d_recon.nii.gz").build_canonical_bundle()
                cap = capture(model, torch.from_numpy(bnp).to(DEV), bbox, True, rcfg, "1frame", True,
                              seq_index=SEED, max_phases=MP)
                save_npz(base, cap); print(f"  regen {base}.npz", flush=True)
        del model; torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
