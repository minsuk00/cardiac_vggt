"""Self-gate the af12 cohorts for Fetal CMR 4D WITHOUT re-running nnU-Net.

The af24 gate's per-frame LV segmentations (rolled/ frames, nnU-Net Task114 2d) are cached under
scratch/eval/_fetal4d_gate/af24hold_s*/seg/<src>_af24__<subj>__f{k}.nii.gz. af12's rolled/ frames
0..11 are the same files (tools/build_af12_from_af24.py), so their segs are too. This links frames
0..11 under the af12 names into a fresh work tag and then runs the STANDING
evaluation/src/engine/fetal4d_gate.py assemble (unchanged code) on it: ED = LV-area argmax over
the 12 frames, theta = ramp mod n_cardphase=12, exactly the 12-frame gate.

    PYTHONPATH=training:. python tools/af12_fetal_gate.py            # link + assemble, all 5 cohorts
    PYTHONPATH=training:. python tools/af12_fetal_gate.py --rhythm hrv   # same for hrv12, from hrv24's segs

hrv24's segs live in hrv24_g3_s* (the ones the shipped hrv24 gates were assembled from: 14/14
subjects reproduce gate.json's per-slice ED + LV area). hrv24_local_s0 is a partial pilot whose segs
differ (only 11/14 reproduce) and is NOT used.
"""
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
import paths  # noqa: E402

SOURCES = ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms")
#          rhythm: (work tag written here, glob of the 24-frame gate's seg tags)
GATES = {"af": ("af12_from_af24hold", "af24hold_s*"),
         "hrv": ("hrv12_from_hrv24g3", "hrv24_g3_s*")}
NF = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhythm", choices=list(GATES), default="af")
    rhythm = ap.parse_args().rhythm
    TAG, src_tags = GATES[rhythm]
    seg = paths.VOLUMES / "_fetal4d_gate" / TAG / "seg"
    seg.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in SOURCES:
        ds12, ds24 = f"{src}_{rhythm}12", f"{src}_{rhythm}24"
        keep, _ = paths.filter_by_split(ds12, paths.subjects(ds12), "test")
        for s in keep:
            gate = paths.subject_dir(ds12, s) / "fetal_cmr_4d" / "gate"
            if (gate / "cardphase.txt").is_file():
                continue
            for k in range(NF):
                hits = sorted(glob.glob(str(paths.VOLUMES / "_fetal4d_gate" / src_tags / "seg" / f"{ds24}__{s}__f{k:02d}.nii.gz")))
                assert len(hits) == 1, f"expected 1 cached {ds24} gate seg for {s} frame {k}, got {hits}"
                dst = seg / f"{ds12}__{s}__f{k:02d}.nii.gz"
                if not dst.exists():
                    os.symlink(hits[0], dst)
            n += 1
    print(f"linked segs for {n} subjects -> {seg}")
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'training'}:{ROOT}", "SPLIT": "test"}
    r = subprocess.run([sys.executable, str(ROOT / "evaluation/src/engine/fetal4d_gate.py"), "assemble",
                        "--split", "test", "--sources", *[f"{s}_{rhythm}12" for s in SOURCES], "--work-tag", TAG], env=env)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
