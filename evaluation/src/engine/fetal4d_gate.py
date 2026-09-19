"""Self-gating front end for the Fetal CMR 4D arm (van Amerom 2019 engine, docs/105).

Input: the bundle's `rolled/` stacks — breath/ with every plane's 12 phases circularly shifted by a
frozen, unknown per-slice roll (pooled.py add_rolled). The arm may use ONLY those images. This
script recovers, per slice, the cardiac phase of every frame from the images and writes what
`mirtk reconstructCardiac` consumes:

    <subject>/fetal_cmr_4d/gate/stack4d.nii.gz   (X,Y,Z,T) the rolled frames as one 4D stack
    <subject>/fetal_cmr_4d/gate/cardphase.txt    one theta per frame, slice-major / frame-minor
    <subject>/fetal_cmr_4d/gate/rrintervals.txt  one R-R per slice (nominal, see below)
    <subject>/fetal_cmr_4d/gate/gate.json        per-slice ED frame, reliability, diagnostics

Method (docs/34/35): the paper's cross-slice synchronisation correlates slices where they overlap
in 3D, which parallel SAX slices never do (measured 106 deg ED scatter, docs/34) — so, as on the
MIITT real-time pilot, each slice is anchored to its own ED found from the LV blood-pool AREA
(Akesson et al. 2025, CPFI, doi 10.1111/cpf.70027: ED = area local maximum), measured with the
public nnU-Net Task114 segmenter — the same net the scorer uses, run on the INPUT frames (never on
GT: the bundle's heart_seg carries GT frame order, which is the label the roll hides). The paper's
per-slice heart-rate step (x-f Fourier peak) is not exercised: with one full cycle of 12 frames the
period is the window length by construction (measured on 380 slices), so R-R is a nominal
constant (1.0 s; CMRxRecon ships no timing) and theta = 2pi * ((f - f_ED) mod T) / T. Slices with
no segmentable LV (base/apex, a physical limit — docs/35 par.6) keep offset 0 and are left to the
engine's robust statistics, exactly as on the MIITT pilot.

Stages (nnU-Net lives in the isolated `nnunet` env, so segmentation is one batched call):
    dump     -> write stack4d + per-frame nnU-Net inputs for every subject into a shared dir
    seg      -> nnUNet_predict over that dir (GPU; seconds per subject)
    assemble -> LV area per (slice, frame) -> ED -> cardphase/rrintervals/gate.json

    PYTHONPATH=training:. python evaluation/src/engine/fetal4d_gate.py dump     --split val [--sources ..] [--subjects ..]
    PYTHONPATH=training:. python evaluation/src/engine/fetal4d_gate.py seg      --split val
    PYTHONPATH=training:. python evaluation/src/engine/fetal4d_gate.py assemble --split val [--sources ..] [--subjects ..]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import paths  # noqa: E402

ARM = "fetal_cmr_4d"
ARM_ORACLE = "fetal_cmr_4d_oracle"   # --oracle: same engine, TRUE per-frame theta (docs/110)
RR_NOMINAL_S = 1.0          # no timing in the source data; sets the engine's time axis only
# run_fetal4d.sh passes `-cardphase CPCOUNT <values>`, and the engine reads the COUNT token itself
# as frame 0 / slice 0's phase (docs/105 par.5c). 710 = 113*2pi, i.e. 6e-05 rad -- VERIFIED to be
# what the engine sees: cardPhase is stored raw (reconstructCardiac.cc:400) and its only consumer,
# CalculateAngularDifference, computes (cp-cp0) - 2pi*floor((cp-cp0)/2pi), a true modulo at any
# magnitude, so angdiff(710) is bit-identical to angdiff(710 mod 2pi).
CARDPHASE_COUNT_TOKEN = 710.0
ENV_SH = os.path.join(ROOT, "tools/nnunet_mnms_eval/env.sh")


def work_root(args):
    """nnU-Net in/out dir — on GPFS (paths.VOLUMES -> scratch/eval), not the repo.

    Keyed on --work-tag, defaulting to the split. The default dir is SHARED across every source of
    a split and holds the live campaign's gating data (test/: 2160 inputs + 2163 segs). `dump` has
    no skip guard and overwrites `in/<ds>__<subj>__f*`, and `seg` scans the whole dir, so any new
    experiment must be given its own tag rather than piling into the campaign's."""
    return paths.VOLUMES / "_fetal4d_gate" / (args.work_tag or args.split)


def gate_dir(ds, subj, arm=None):
    return paths.subject_dir(ds, subj) / (arm or ARM) / "gate"


def subjects_of(args):
    out = []
    for ds in args.sources:
        keep, _ = paths.filter_by_split(ds, paths.subjects(ds), args.split)
        for s in keep:
            if args.subjects and s not in args.subjects:
                continue
            out.append((ds, s))
    return out


def load_rolled(ds, subj):
    man = json.load(open(paths.manifest(ds, subj)))
    if "rolled" not in man:
        raise KeyError(f"{ds}/{subj}: bundle has no rolled/ stack — run pooled.py --add-rolled")
    T = int(man["T"])
    ims = [nib.load(str(paths.bundle_stack(ds, subj, "rolled", k))) for k in range(T)]
    arr = np.stack([np.asarray(im.dataobj, dtype=np.float32) for im in ims], -1)   # (X,Y,Z,T)
    return arr, ims[0].affine, man


def dump(args):
    inp = work_root(args) / "in"
    inp.mkdir(parents=True, exist_ok=True)
    n = 0
    for ds, subj in subjects_of(args):
        arr, aff, man = load_rolled(ds, subj)
        gd = gate_dir(ds, subj); gd.mkdir(parents=True, exist_ok=True)
        # The 4D header's time pixdim IS the frame duration reconstructCardiac uses for its
        # temporal PSF (ReconstructionCardiac4D.cc: _slice_dt = attr._dt; dtrad = 2*pi*dt/rr sets the
        # sinc width). nibabel's default of 1.0 s = one whole R-R would make every output phase a
        # blend of the entire cycle (found on the first pilot, docs/105 par.5a). One frame = RR/T.
        # The time-unit code must stay UNSET: MIRTK multiplies a 'sec'-tagged pixdim by 1000 on
        # read while the R-R stays in seconds, so dtrad = 2*pi*83.3/1.0 -> a flat window over the
        # whole cycle (measured on the P012 debug run, docs/105 par.5c). The authors' own writer
        # (ktrecon mrecon_writenifti.m) stores seconds with xyzt_units = 0 for the same reason.
        # Slice 0's frames are re-rolled in `assemble` (ED first) — see there.
        img = nib.Nifti1Image(arr, aff)
        img.header.set_zooms((*img.header.get_zooms()[:3], RR_NOMINAL_S / arr.shape[-1]))
        img.header.set_xyzt_units("mm", None)
        nib.save(img, str(gd / "stack4d.nii.gz"))
        for k in range(arr.shape[-1]):        # full-FOV input frames, one 3D file per frame
            nib.save(nib.Nifti1Image(arr[..., k], aff), str(inp / f"{ds}__{subj}__f{k:02d}_0000.nii.gz"))
        n += 1
    print(f"dump: {n} subjects -> {inp}")


def seg(args):
    inp, out = work_root(args) / "in", work_root(args) / "seg"
    out.mkdir(parents=True, exist_ok=True)
    if not list(inp.glob("*_0000.nii.gz")):
        sys.exit(f"seg: nothing in {inp} — run dump first")
    cmd = (f"source '{ENV_SH}' && nnUNet_predict -i '{inp}' -o '{out}' -t 114 -m 2d "
           f"-tr nnUNetTrainerV2_MMS")
    print(f"[seg] Task114 2d nnUNetTrainerV2_MMS  in={inp}  out={out}", flush=True)
    subprocess.run(["micromamba", "run", "-n", "nnunet", "bash", "-c", cmd], check=True)


def detect_ed(area):
    """Per-slice ED frame = argmax of the LV-area curve over the one circular cycle (the Akesson
    local-maximum rule reduces to this on a single-beat series). No smoothing: the diastolic
    plateau is asymmetric (slow late filling, fast ejection), so a 3-tap circular smooth biased the
    argmax one frame early on 7/9 slices of the first pilot subject. Returns None when the slice
    has no LV at all (base/apex)."""
    if area.max() <= 0:
        return None
    return int(np.argmax(area))


def assemble(args):
    segdir = work_root(args) / "seg"
    summary = []
    for ds, subj in subjects_of(args):
        man = json.load(open(paths.manifest(ds, subj)))
        T, D = int(man["T"]), int(man["D"])
        theta = np.zeros((D, T)); per = []
        roll = man.get("rolled", {}).get("roll_per_plane")          # diagnostic only
        # The oracle arm takes theta from the truth manifest and never reads `area`, so it must not
        # require a GPU nnU-Net pass whose every output it discards.
        if not args.oracle:
            segs = []
            for k in range(T):
                f = segdir / f"{ds}__{subj}__f{k:02d}.nii.gz"
                if not f.is_file():
                    sys.exit(f"{ds}/{subj}: missing seg {f} — run seg first")
                segs.append(np.asarray(nib.load(str(f)).dataobj))
            seg4 = np.stack(segs, -1)                               # (X,Y,Z,T) labels 1=LV
            area = (seg4 == 1).sum((0, 1)).astype(float)            # (Z,T)
            for z in range(D):
                ed = detect_ed(area[z])
                off = 0 if ed is None else ed
                theta[z] = 2 * np.pi * ((np.arange(T) - off) % T) / T
                rec = {"z": z, "ed_frame": ed, "anchored": ed is not None,
                       "lv_coverage": float((area[z] > 0).mean()), "lv_area_max": float(area[z].max())}
                if roll is not None and ed is not None:
                    rec["ed_err_frames_diag"] = int(((ed - roll[z] + T // 2) % T) - T // 2)
                per.append(rec)
        # ORACLE (docs/110): the same engine handed the TRUE per-(slice, frame) cardiac phase from
        # the rhythm truth manifest, instead of its own self-gating estimate. Everything else --
        # the stack, the parameters, the readout rule -- is identical, so the arm isolates GATING
        # error from reconstruction error. Unlike the self-gated theta (which assumes uniformly
        # spaced frames, the very assumption under test) these thetas are fractional.
        if args.oracle:
            r = man.get("rhythm")
            if not r:
                sys.exit(f"{ds}/{subj}: --oracle needs a bundle with a 'rhythm' block "
                         f"(tools/build_af_bundle.py); this is a plain bundle")
            pos = np.asarray(r["pos_per_plane"], dtype=np.float64)   # (D, T), GT-frame units
            if pos.shape != (D, T):
                sys.exit(f"{ds}/{subj}: rhythm.pos_per_plane is {pos.shape}, expected {(D, T)}")
            theta = 2 * np.pi * ((pos % T) / T)
            # Shift every theta by the constant that puts theta[0,0] at the count token's residue.
            # The engine reads that token as frame 0 / slice 0's phase, and for the self-gated arm
            # theta[0,0] is exactly 0 so the literal 710 works. Oracle thetas are fractional, and
            # NO integer token can encode an arbitrary phase: 710 is a convergent of 2pi, so the
            # reachable residues have ~9e-3 rad gaps and a search over 20k candidates misses 1e-3
            # for ~58% of subjects (measured). A constant offset instead only relabels which output
            # bin is "phase 0", and the ref_phase readout samples these SAME shifted thetas, so it
            # cancels exactly in the score while keeping the argv at 710 tokens.
            theta = (theta - theta[0, 0] + CARDPHASE_COUNT_TOKEN % (2 * np.pi)) % (2 * np.pi)
        gd = gate_dir(ds, subj, ARM_ORACLE if args.oracle else ARM)
        gd.mkdir(parents=True, exist_ok=True)
        # The container's reconstructCardiac reads `-cardphase N v...` as N+1 entries with the COUNT
        # token as frame 0's phase (measured, docs/105 par.5c; the authors' scripts hit the same
        # thing). run_fetal4d.sh passes a count of 710 = 113*2pi (0.0004 rad), so frame 0 must be
        # an ED-phase frame: re-roll slice 0 in the stack so its detected ED (offset 0 if none)
        # comes first, and shift its thetas to match. Frame order elsewhere is unchanged.
        arr, aff, _ = load_rolled(ds, subj)
        if args.oracle:
            # No re-roll: the oracle thetas are fractional, so no frame order can put an exact
            # theta=0 at frame 0. The constant offset applied above makes theta[0,0] match the
            # count token instead, so the acquisition frame order is preserved here.
            ed0 = 0
        else:
            ed0 = per[0]["ed_frame"] or 0
            arr[:, :, 0, :] = np.roll(arr[:, :, 0, :], -ed0, axis=-1)
            theta[0] = np.roll(theta[0], -ed0)
            assert abs(theta[0, 0]) < 1e-9, theta[0, 0]
        img = nib.Nifti1Image(arr, aff)
        img.header.set_zooms((*img.header.get_zooms()[:3], RR_NOMINAL_S / T))
        img.header.set_xyzt_units("mm", None)                      # see dump(): must stay unset
        nib.save(img, str(gd / "stack4d.nii.gz"))
        with open(gd / "cardphase.txt", "w") as fh:               # slice-major / frame-minor
            fh.write(" ".join(f"{v:.6f}" for v in theta.reshape(-1)) + "\n")
        with open(gd / "rrintervals.txt", "w") as fh:
            fh.write(" ".join(f"{RR_NOMINAL_S:.6f}" for _ in range(D)) + "\n")
        errs = [p["ed_err_frames_diag"] for p in per if "ed_err_frames_diag" in p]
        diag = None
        if len(errs) > 1:
            ang = np.array(errs) * 2 * np.pi / T
            R = abs(np.mean(np.exp(1j * ang)))
            diag = {"n": len(errs), "within_1_frame": float(np.mean(np.abs(errs) <= 1)),
                    "circ_std_deg": float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9)))))}
        meta = {"source": ds, "subject": subj, "T": T, "D": D, "rr_nominal_s": RR_NOMINAL_S,
                "n_anchored": int(sum(p["anchored"] for p in per)),
                "gater": "oracle" if args.oracle else "self",
                "arm": ARM_ORACLE if args.oracle else ARM,
                "segmenter": "nnU-Net Task114 2d nnUNetTrainerV2_MMS on rolled/ input frames (full FOV)",
                "ed_rule": "LV-area (label 1) argmax over the single circular cycle, no smoothing; no LV -> offset 0",
                "theta": ("2*pi*(rhythm.pos_per_plane mod T)/T -- TRUE per-frame phase from the "
                          "simulation truth manifest (fractional)" if args.oracle
                          else "2*pi*((f - f_ED) mod T)/T, wrapped [0, 2pi)"),
                "slice0_frame_roll": -ed0,      # stack4d[:, :, 0] = rolled[:, :, 0] rolled by this (ED first)
                "stack4d_time_pixdim_s": RR_NOMINAL_S / T, "stack4d_time_units": "unset (MIRTK: no x1000)",
                "diag_vs_manifest_roll": diag, "per_slice": per}
        json.dump(meta, open(gd / "gate.json", "w"), indent=1)
        summary.append((ds, subj, meta["n_anchored"], D, diag))
    for ds, subj, na, D, diag in summary:
        d = f"within1={diag['within_1_frame']:.2f} scatter={diag['circ_std_deg']:.1f}deg" if diag else "no diag"
        print(f"  {ds:9} {subj:45} anchored {na:2d}/{D:2d}  {d}")
    print(f"assemble: {len(summary)} subjects")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dump", "seg", "assemble"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--sources", nargs="+", default=list(paths.DATASETS))
    ap.add_argument("--work-tag", default=None,
                    help="name of the shared nnU-Net work dir under _fetal4d_gate/ (default: the "
                         "split). Give a NEW experiment its own tag: the default dir holds the live "
                         "campaign's gating data, `dump` overwrites inputs without a guard, and "
                         "`seg` scans the whole directory.")
    ap.add_argument("--oracle", action="store_true",
                    help="assemble only: take theta from the bundle's rhythm truth manifest "
                         "(TRUE per-frame cardiac phase) instead of the nnU-Net LV-area anchor, and "
                         f"write to <subject>/{ARM_ORACLE}/gate/. The ceiling arm that separates "
                         "gating error from reconstruction error (docs/110).")
    ap.add_argument("--subjects", nargs="+", default=None)
    args = ap.parse_args()
    {"dump": dump, "seg": seg, "assemble": assemble}[args.stage](args)


if __name__ == "__main__":
    main()
