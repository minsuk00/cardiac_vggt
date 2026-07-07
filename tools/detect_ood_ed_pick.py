"""Step 3 of OOD ED detection: read nnU-Net Task114 segs of the mid-slice frames, take the
frame with the largest LV blood-pool (label 1) as ED, write ed_frames.json
({"OCMR": {subj: ed_frame}, "Goett": {...}, "MIITT": {...}}).

Run: micromamba run -n svr python tools/detect_ood_ed_pick.py
"""
import os, sys, json, glob, re
import numpy as np
import nibabel as nib

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEG = os.path.join(_ROOT, "result", "reference_models_io", "ed_detect_segs")
OUT = os.path.join(_ROOT, "result", "reference_models_io", "ed_frames.json")
LV = 1


def main():
    segs = glob.glob(os.path.join(SEG, "*.nii.gz"))
    areas = {}  # (ds,subj) -> {frame: lv_area}
    for s in segs:
        m = re.match(r"(\w+)__(.+)__f(\d+)$", os.path.basename(s)[:-7])
        if not m:
            continue
        ds, subj, f = m.group(1), m.group(2), int(m.group(3))
        lv = int((np.asarray(nib.load(s).dataobj) == LV).sum())
        areas.setdefault((ds, subj), {})[f] = lv
    out = {}
    for (ds, subj), fa in sorted(areas.items()):
        ed = max(fa, key=fa.get)
        out.setdefault(ds, {})[subj] = ed
        print(f"  {ds}/{subj}: ED=f{ed} (LV={fa[ed]} vox; {len(fa)} frames, "
              f"range {min(fa.values())}-{max(fa.values())})")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
