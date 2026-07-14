"""gather05 (1-frame-trained, gather-aux=0.5) run at 1-frame-per-slice on MIITT + CMRxRecon val,
with the honest input-strip render (EVERY fed slice shown at its true physical z, no 12-col collapse).

- MIITT gated (native pitch != 12mm): regime '1frame_contz' — each native slice at its TRUE
  fractional canonical z, nothing snapped, nothing discarded (~13 slices/subj).
- CMRxRecon val (native 12mm = the canonical pitch): regime '1frame' — native slices already sit on
  the canonical planes, so 1-frame-per-in-bbox-plane IS the no-discard result (continuous-z is a
  no-op at 12mm). Bundle from MRIDataset.get_data (in-distribution path, not an adapter).

clean + normal breathing, clean-reference, dynamic seed 72, 18 cardiac phases, hi-res dpi 130.
Out: result/gated_model_sweep/gather05_contz/{miitt,cmrxrecon}/<subj>/<cond>_7row.{gif,npz}
Run inline: micromamba run -n svr env PYTHONPATH=training:. python -u tools/miitt_viz/gather05_contz_sweep.py
"""
import os, sys, glob, time, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "training")
from inference.adapters import MIITTGatedAdapter
from inference.inference import load_rtfb_model_reference
from inference.run_cmrxrecon import build_mri_dataset
from inference.run_gated_ood import load_rcfg
from tools.miitt_viz.gated_gather05_7row import capture, render_inputstrip

DEV = "cuda"; SEED = 72; MP = 18
MODEL_CKPTS = {
    "gather05": "scratch/logs/216539845_*ftgather05*1frame*/ckpts/checkpoint_last.pt",
    "s20contz": "scratch/logs/216949414_*s20contz*/ckpts/checkpoint_last.pt",
    "s20":      "scratch/logs/216949759_mri_volume_diffusion_s20_*/ckpts/checkpoint_last.pt",  # snapped S=20 sibling
    "control0": "scratch/logs/216539845_*ftctrl_gather0_1frame*/ckpts/checkpoint_last.pt",
}
SELECTED = [m for m in sys.argv[1:] if m in MODEL_CKPTS] or ["gather05"]
MIITT_SUBJS = [1, 2]                 # first 2 MIITT volunteers
CMRX_SUBJS = [0, 7]                  # first 2 CMRx val seq_indices


def save(base, cap):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    np.savez_compressed(base + ".npz", gt=cap["GT"], recon=cap["RE"], inp=cap["IN"], dvf=cap["DV"],
                        cov=cap["CO"], has_slot=np.array(cap["has_slot"]), ref_zmid=cap["z_mid"],
                        zbr=cap["rd"][:, 0], sop=np.array(cap["sop"]), applied_disp=cap["rd"],
                        per_phase_motion=np.array(cap["metr"]["motion"]),
                        per_phase_full=np.array(cap["metr"]["full"]),
                        per_phase_ssim=np.array(cap["metr"]["ssim"]),
                        in_slots=cap["IN_slots"], slot_zf=cap["slot_zf"], dvf_slots=cap["DV_slots"])


def run_one(model, mname, out, pb, bbox, rcfg, positions, regime, subj_tag, ds):
    for tag, br in [("clean", False), ("normal", True)]:
        base = os.path.join(out, ds, subj_tag, f"{tag}_7row")
        os.makedirs(os.path.dirname(base), exist_ok=True)
        if os.path.exists(base + ".npz"):
            print(f"  [{ds}/{subj_tag} {tag}] skip (exists)", flush=True); continue
        t0 = time.time()
        cap = capture(model, pb, bbox, br, rcfg, regime, clean_ref=True,
                      seq_index=SEED, max_phases=MP, positions=positions)
        render_inputstrip(cap, f"{mname} {regime} | {ds}/{subj_tag} | {tag}", base + ".gif", dpi=130)
        save(base, cap)
        m = {k: float(np.nanmean(v)) for k, v in cap["metr"].items()}
        S = cap["rd"].shape[0]
        print(f"  [{ds}/{subj_tag} {tag}] S={S} motion={m['motion']:.2f} full={m['full']:.2f}dB "
              f"ssim={m['ssim']:.3f} ({time.time()-t0:.0f}s) -> {base}.gif", flush=True)


def main():
    rn = load_rcfg()                     # normal breathing config (amp 16) for MIITT
    # ---- pre-fetch MIITT continuous-z bundles (shared across models) ----
    miitt = []
    for i in MIITT_SUBJS:
        p = f"scratch/data/MIITT/nifti/Volunteer{i}/gated/sax/4d_recon.nii.gz"
        if not os.path.exists(p):
            continue
        ad = MIITTGatedAdapter(p)
        bnp, bbox = ad.build_canonical_bundle(continuous_z=True)
        miitt.append((f"Volunteer{i}", bnp, bbox, ad.slice_positions_mm()))
    # ---- pre-fetch CMRxRecon val bundles once (expensive dataset build) ----
    mri_ds, rcfg_cmrx = build_mri_dataset()
    ids = [str(p).split("/")[-2] for p in mri_ds.subjects]
    print(f"cmrx val subjects: {[(s, ids[s]) for s in CMRX_SUBJS]}", flush=True)
    cmrx = []
    for s in CMRX_SUBJS:
        data = mri_ds.get_data(seq_index=s, img_per_seq=mri_ds.num_slices)
        cmrx.append((f"subj{s:02d}_{ids[s]}",
                     np.asarray(data["phases"]).astype(np.float32),
                     np.asarray(data["anatomy_bbox"]).astype(np.int64)))

    for mname in SELECTED:
        ckpt = glob.glob(MODEL_CKPTS[mname])[0]
        out = f"result/gated_model_sweep/{mname}_contz"
        model = load_rtfb_model_reference(ckpt, refiner=False, device=DEV)
        print(f"\n########## {mname}: {ckpt} -> {out} ##########", flush=True)
        # MIITT: true continuous-z (fractional, nothing discarded)
        for subj, bnp, bbox, positions in miitt:
            pb = torch.from_numpy(bnp).to(DEV)
            print(f"=== {mname} MIITT {subj}  T={bnp.shape[0]} nslices={len(positions)} bbox={bbox.tolist()} ===", flush=True)
            run_one(model, mname, out, pb, bbox, rn, positions, "1frame_contz", subj, "miitt")
            del pb; torch.cuda.empty_cache()
        # CMRxRecon val: native 12mm → 1-frame-per-canonical-plane (no discard; contz is a no-op)
        for subj, bnp, bbox in cmrx:
            pb = torch.from_numpy(bnp).to(DEV)
            print(f"=== {mname} CMRx {subj}  T={bnp.shape[0]} bbox={bbox.tolist()} ===", flush=True)
            run_one(model, mname, out, pb, bbox, rcfg_cmrx, None, "1frame", subj, "cmrxrecon")
            del pb; torch.cuda.empty_cache()
        del model; torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
