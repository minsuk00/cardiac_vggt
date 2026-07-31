"""Emit the whole-heart-seg work-list: one line per (dataset, regime, recon-path) unit.

Consumed by the sbatch array job `sbatch/whs_segment.sh` (line N -> array task N). A "unit" is one
subject-regime = one nnUNet_predict call over all its phases/frames. 421 units total:
CMRx 301 (all subjects, split file) + MIITT 13 gated + 13 rt + OCMR 8 gated + 17 rtfb + Goettingen 69.

Line format:  "<dataset> <regime> <path>"
  cmrx:       path = the subject's `sax/` DIR (12 native `3d_recon/sax_frame_{tt}.nii.gz` inside)
  others:     path = the 4D recon NIfTI file (X,Y,Z,T)
"""
import os, glob

ROOT = "/home/minsukc/vggt"
DATA = os.path.join(ROOT, "scratch/data")
OUT = os.path.join(DATA, "whs/worklist.txt")


def main():
    lines = []

    # --- The pooled training cohort: CMRx 2023/24/25 + converted ACDC + converted M&Ms-1 ---
    # Enumerated by GLOBBING the directories, NOT from a split file. Two reasons: the seg should
    # cover everything on disk regardless of how train/val/test is later partitioned, and the old
    # code read `training/splits/random_8_1_1.txt`, which is deprecated, is CMRxRecon2024-only, and
    # lists PRE-RENAME names (`Train_P140` vs the on-disk `CMRx24_Train_P140`) — the latter is what
    # made cardiac_phase.csv un-joinable and `ef_val_sweep` raise KeyError.
    #
    # All five sources share the CMRx layout `<ID>/sax/3d_recon/sax_frame_{tt}.nii.gz` (ACDC and
    # M&Ms via tools/convert_to_sax_layout.py, docs/58), so all five use the same path convention:
    # path = the subject's `sax/` DIR. `acdc_sax`/`mnms_sax` are separate dataset tokens purely so
    # the manifest keeps a real source label for per-source metric bucketing; prep_one.py and
    # assemble_whs.py treat them exactly like `cmrx`.
    POOL = [
        ("cmrx",     os.path.join(DATA, "CMRxRecon2023/Cine_combined/*/sax")),
        ("cmrx",     os.path.join(DATA, "CMRxRecon2024/Cine_combined/*/sax")),
        ("cmrx",     os.path.join(DATA, "CMRxRecon2025/Cine_combined/*/sax")),
        ("acdc_sax", os.path.join(DATA, "ACDC_sax/*/sax")),
        ("mnms_sax", os.path.join(DATA, "MNMs_sax/*/sax")),
    ]
    for token, pattern in POOL:
        found = sorted(glob.glob(pattern))
        if not found:
            print(f"WARN no subjects matched {pattern}")
        for saxdir in found:
            # Require all 12 frames — a partially-written subject must not enter the worklist.
            n = len(glob.glob(os.path.join(saxdir, "3d_recon", "sax_frame_*.nii.gz")))
            if n == 12:
                lines.append(f"{token} gated {saxdir}")   # every pooled source is ECG-gated cine
            else:
                print(f"WARN skipping {saxdir}: {n} frames, expected 12")

    # --- MIITT: 13 subjects x {gated, rt} ---
    mroot = os.path.join(DATA, "MIITT/nifti")
    for s in sorted(os.listdir(mroot)):
        for regime, sub in [("gated", "gated"), ("rt", "realtime")]:
            r = os.path.join(mroot, s, sub, "sax", "4d_recon.nii.gz")
            if os.path.exists(r):
                lines.append(f"miitt {regime} {r}")

    # --- OCMR: gated (fs) + rtfb (us) SAX cines ---
    for regime, dom in [("gated", "gated"), ("rtfb", "rtfb")]:
        for r in sorted(glob.glob(os.path.join(DATA, "ocmr/recon", dom, "exam_*", "sax__*", "sax_cine.nii.gz"))):
            lines.append(f"ocmr {regime} {r}")

    # --- Goettingen: RT SAX stacks (one 4D vol per dir) ---
    for r in sorted(glob.glob(os.path.join(DATA, "goettingen/recon", "vol*", "vol*.nii.gz"))):
        lines.append(f"goettingen rt {r}")

    # --- ACDC, RAW download (native full T, 4D) ---
    # NOT part of the training pool — that is `acdc_sax` above, on the 12-frame converted stacks.
    # This segments the untouched download at its native T (13-35) and writes siblings inside the
    # read-only ACDC/ tree. Kept because it is a genuinely different product: the native-T LV-volume
    # curve is what made the "cost of 12-frame sampling" measurement possible (median EF error
    # 0.24 pts; see scratch/data/ACDC/README.md), and it is the independent QC reference for the
    # converted stacks. All 150 already exist, so these units SKIP on re-run.
    for split in ("training", "testing"):
        for p in sorted(glob.glob(os.path.join(DATA, "ACDC", split, "patient*"))):
            r = os.path.join(p, os.path.basename(p) + "_4d.nii.gz")
            if os.path.exists(r):
                lines.append(f"acdc gated {r}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    # per-dataset tally
    from collections import Counter
    c = Counter(l.split()[0] + "/" + l.split()[1] for l in lines)
    print(f"{len(lines)} units -> {OUT}")
    for k, v in sorted(c.items()):
        print(f"  {k:18} {v}")


if __name__ == "__main__":
    main()
