"""Build the `<src>_af12` cohorts = the first 12 frames per slice of the `<src>_af24` bundles.

Same simulator, same subjects, same rhythm draw: build_af_bundle.simulate's per-plane RNG streams
do not depend on n_frames, so af24's frames 0..11 ARE the 12-frame AF record on the current
(hold-rule, docs/114) position model. Nothing is re-simulated; every image/GT/mask file is a
relative SYMLINK into the af24 bundle (zero bytes, byte-identical by construction), and only the
manifest is rewritten -- T=12, scatter draw folded to frames 0..11, rhythm block truncated.

Also links the af24 GT seg cache (seg_gt_3d_fullres, frames 0..11; gt_t00 and the ROI are the same
files so the content keys match) with a T=12 src.json, so ef_dice reuses it read-only.
cine_gt.nii.gz is deliberately NOT linked: image_metrics writes a fresh 12-frame one.

NEVER OVERWRITES: an existing <src>_af12/out/<subject> dir is refused (no --overwrite here).

    PYTHONPATH=training:. python tools/build_af12_from_af24.py [--sources ...] [--check]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
import paths  # noqa: E402

SOURCES = ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms")
NF = 12
SIBLINGS = ("mask.nii.gz", "mask_fov.nii.gz", "mask_heart.nii.gz", "mask_heart_pad10.nii.gz")
SEG_CACHE = "seg_gt_3d_fullres"


def rel_link(dst: Path, target: Path):
    dst.symlink_to(os.path.relpath(target, dst.parent))


def truncate_manifest(m):
    T = int(m["n_cardphase"])
    assert m["T"] == 24 and T == NF, (m["T"], T)
    out = json.loads(json.dumps(m))
    out["T"] = NF
    out["cohort"] = f"{m['source']}_af12"
    sc = out["scatter"]
    # The builder's own rule for nf == T: frame j = (phase + roll) mod T, beat 0 for every plane.
    sc["phase_per_plane"] = [int(p % T) for p in m["scatter"]["phase_per_plane"]]
    sc["beat_per_plane_scatter"] = [0] * len(sc["phase_per_plane"])
    sc["note"] = ("FRAME index per plane (not cardiac phase): stack_t{j}[z] is that slice's j-th "
                  "real-time frame. af24's draw folded onto its first beat: j = (phase + roll_z) mod T.")
    r = out["rhythm"]
    pos = np.asarray(m["rhythm"]["ref_pos"][:NF], float)
    r["arm"] = "af12"
    r["ref_pos"] = pos.tolist()
    r["gt_map"] = m["rhythm"]["gt_map"][:NF]
    r["duplicate_targets"] = [[int(i) for i in np.where(np.abs((pos % T) - p) < 1e-9)[0]]
                              for p in sorted({round(float(x) % T, 9) for x in pos})
                              if int((np.abs((pos % T) - p) < 1e-9).sum()) > 1]
    r["pos_per_plane"] = [row[:NF] for row in m["rhythm"]["pos_per_plane"]]
    r["beat_per_plane"] = [row[:NF] for row in m["rhythm"]["beat_per_plane"]]
    r["params"] = {**m["rhythm"]["params"], "n_frames": NF}
    r["builder"] = "build_af12_from_af24.py (prefix of build_af_bundle.py's af24 record)"
    r["truncated_from"] = f"{m['source']}_af24"
    r["physio_note"] = "physio post-conditions were measured on the 24-frame record and are kept as-is"
    return out


def build(src, s):
    ds24, ds12 = f"{src}_af24", f"{src}_af12"
    a = paths.subject_dir(ds24, s)
    d = paths.subject_dir(ds12, s)
    if d.exists():
        return "skipped"
    m = json.load(open(a / "manifest.json"))
    man = truncate_manifest(m)
    d.mkdir(parents=True)
    (d / "breath").mkdir(); (d / "gt").mkdir()
    for f in range(NF):
        rel_link(d / "breath" / f"stack_t{f:02d}.nii.gz", a / "breath" / f"stack_t{f:02d}.nii.gz")
        rel_link(d / "gt" / f"gt_t{f:02d}.nii.gz", a / "gt" / f"gt_t{f:02d}.nii.gz")
    os.symlink("breath", d / "rolled")
    for name in SIBLINGS:
        if (a / name).exists():
            rel_link(d / name, a / name)
    cache = a / SEG_CACHE
    if (cache / "src.json").is_file():
        src_json = json.load(open(cache / "src.json"))
        if src_json.get("T") == 24 and all((cache / f"seg_t{f:02d}.nii.gz").is_file() for f in range(NF)):
            (d / SEG_CACHE).mkdir()
            for f in range(NF):
                rel_link(d / SEG_CACHE / f"seg_t{f:02d}.nii.gz", cache / f"seg_t{f:02d}.nii.gz")
            json.dump({**src_json, "T": NF, "note": f"frames 0..{NF - 1} of {ds24}'s cache (same gt_t00 + ROI)"},
                      open(d / SEG_CACHE / "src.json", "w"))
    json.dump(man, open(d / "manifest.json", "w"), indent=1)   # LAST: paths.subjects() keys on it
    return "built"


def check(src, s):
    ds24, ds12 = f"{src}_af24", f"{src}_af12"
    a, d = paths.subject_dir(ds24, s), paths.subject_dir(ds12, s)
    m24, m12 = json.load(open(a / "manifest.json")), json.load(open(d / "manifest.json"))
    assert m12["T"] == NF and m12["n_cardphase"] == NF and m12["rhythm"]["params"]["n_frames"] == NF
    assert max(m12["scatter"]["phase_per_plane"]) < NF
    assert m12["scatter"]["ref_plane"] == m24["scatter"]["ref_plane"]
    assert np.allclose(m12["rhythm"]["ref_pos"], m24["rhythm"]["ref_pos"][:NF])
    for f in range(NF):
        for sub, pre in (("breath", "stack"), ("gt", "gt")):
            x = np.asarray(nib.load(d / sub / f"{pre}_t{f:02d}.nii.gz").dataobj)
            y = np.asarray(nib.load(a / sub / f"{pre}_t{f:02d}.nii.gz").dataobj)
            assert x.shape == y.shape and np.array_equal(x, y), (s, sub, f)
    assert paths.gt_sha256(ds12, s) == paths.gt_sha256(ds24, s)
    assert not (d / "cine_gt.nii.gz").exists()
    c = d / SEG_CACHE / "src.json"
    assert c.is_file() and json.load(open(c))["T"] == NF, s
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--split", default="test")
    ap.add_argument("--check", action="store_true", help="verify built cohorts, write nothing")
    a = ap.parse_args()
    n = {}
    for src in a.sources:
        ds24 = f"{src}_af24"
        keep, _ = paths.filter_by_split(ds24, paths.subjects(ds24), a.split)
        for s in keep:
            st = "ok" if a.check and check(src, s) else (None if a.check else build(src, s))
            n[st] = n.get(st, 0) + 1
        print(f"  {src}: {len(keep)} subjects -> {n}", flush=True)
    print(n)


if __name__ == "__main__":
    main()
