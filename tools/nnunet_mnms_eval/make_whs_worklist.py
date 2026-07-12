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

    # --- CMRx: all 301 subjects from the split file ---
    split = os.path.join(ROOT, "training/splits/random_8_1_1.txt")
    subs = []
    with open(split) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("[") or s.startswith("#"):
                continue
            subs.append(s)
    for s in subs:
        saxdir = os.path.join(DATA, "CMRxRecon2024/Cine_combined", s, "sax")
        if os.path.isdir(os.path.join(saxdir, "3d_recon")):
            lines.append(f"cmrx gated {saxdir}")   # CMRx cine is ECG-gated (regime label unified to 'gated')
        else:
            print("WARN missing CMRx sax:", saxdir)

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

    # --- ACDC: ECG-gated cine (SVR recon target); one 4D cine per patient, train+test ---
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
