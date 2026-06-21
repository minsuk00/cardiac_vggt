"""Stage ACDC ED/ES frames as nnU-Net v1 inputs + collect matching GT.

ACDC ships proper-geometry SAX NIfTIs already, so we just copy the ED/ES image
frame to <case>_0000.nii.gz and the GT seg to a parallel folder as <case>.nii.gz.
ED/ES frame numbers come from each patient's Info.cfg.
"""
import argparse, os, glob, shutil


def frame_no(cfg, key):
    for line in open(cfg):
        if line.startswith(key + ":"):
            return int(line.split(":")[1].strip())
    raise KeyError(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc_split", required=True, help="path to ACDC training/ or testing/")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all patients")
    args = ap.parse_args()
    os.makedirs(args.img_dir, exist_ok=True)
    os.makedirs(args.gt_dir, exist_ok=True)

    pats = sorted(glob.glob(os.path.join(args.acdc_split, "patient*")))
    if args.limit:
        pats = pats[: args.limit]
    n = 0
    for p in pats:
        pid = os.path.basename(p)
        cfg = os.path.join(p, "Info.cfg")
        for phase in ("ED", "ES"):
            fr = frame_no(cfg, phase)
            img = os.path.join(p, f"{pid}_frame{fr:02d}.nii.gz")
            gt = os.path.join(p, f"{pid}_frame{fr:02d}_gt.nii.gz")
            if not (os.path.exists(img) and os.path.exists(gt)):
                print("MISSING", img); continue
            case = f"{pid}_{phase}"
            shutil.copy(img, os.path.join(args.img_dir, f"{case}_0000.nii.gz"))
            shutil.copy(gt, os.path.join(args.gt_dir, f"{case}.nii.gz"))
            n += 1
    print(f"staged {n} cases ({len(pats)} patients) -> {args.img_dir}")


if __name__ == "__main__":
    main()
