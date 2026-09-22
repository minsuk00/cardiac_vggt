"""Frames-per-slice probe for Fetal CMR 4D (self-gated) on af24: fit on the first N in {12, 24}
frames of each slice's rolled/ record, read out and score at the SAME 24 GT frames. Sibling of
tools/cinevol_nframes_probe.py (same 10 subjects, same temp/ discipline).

Gate: the live af24 gate's nnU-Net LV segmentations are cached under scratch/eval/_fetal4d_gate/
af24hold_s*/seg (24 frames per subject), so the N-frame gate is rebuilt from the LV-area curve of
frames 0..N-1 exactly as fetal4d_gate.assemble does (ED = argmax area, theta = uniform ramp mod 12,
slice 0 re-rolled ED-first). `setup` asserts the N=24 rebuild reproduces the live gate.json ED
frames. Readout thetas extend the same ramp to all 24 GT frames (for N=12 frames f and f+12 share a
theta, i.e. the periodic cine is read out twice -- exactly what a 12-frame Fetal fit predicts).

Everything is written under temp/fetal4d_nframes_probe/; scratch/eval is reached read-only through
symlinks, evaluation/ is untouched (the recon shell is a tools/ copy, see fetal4d_probe_recon.sh).

    PYTHONPATH=training:. python tools/fetal4d_nframes_probe.py setup
    PYTHONPATH=training:. python tools/fetal4d_nframes_probe.py run --index 3 --n 12   # one (subject, N), CPU
    PYTHONPATH=training:. python tools/fetal4d_nframes_probe.py report
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("tools", "evaluation", "evaluation/src/score", "evaluation/src/engine", "training"):
    sys.path.insert(0, str(ROOT / p))
import paths                                  # noqa: E402
from run_baselines import thickness_mm        # noqa: E402
from fetal4d_gate import detect_ed, RR_NOMINAL_S   # noqa: E402
from cinevol_nframes_probe import SUBJECTS, LINKED, _link   # noqa: E402  (same 10 subjects)

PROBE = ROOT / "temp" / "fetal4d_nframes_probe"
REAL = paths.VOLUMES.resolve()
ARM = "fetal_cmr_4d"
NS = (12, 24)
NCP = 12
N_OUT = 24
SEG_TAGS = "af24hold_s*"     # the live (post-hold-rule, docs/114) af24 gate's seg cache
SHELL = ROOT / "tools" / "fetal4d_probe_recon.sh"


def n_root(n):
    return PROBE / f"n{n}"


def use_root(root):
    paths.VOLUMES = Path(root)


def lv_area(ds, s, T):
    """(D, T) LV-label voxel count per (slice, frame) from the cached gate segs (rolled/ frames)."""
    segs = []
    for k in range(T):
        hits = glob.glob(str(REAL / "_fetal4d_gate" / SEG_TAGS / "seg" / f"{ds}__{s}__f{k:02d}.nii.gz"))
        assert len(hits) >= 1, f"no cached gate seg for {ds}/{s} frame {k}"
        segs.append(np.asarray(nib.load(hits[0]).dataobj))
    return (np.stack(segs, -1) == 1).sum((0, 1)).astype(float)


def write_gate(ds, s, n, gd, area, man):
    """fetal4d_gate.assemble, restricted to frames 0..n-1, plus the 24-frame readout ramp."""
    D, ref = int(man["D"]), int(man["scatter"]["ref_plane"])
    assert ref != 0, "readout row for slice 0 would be the re-rolled one"
    ims = [nib.load(str(REAL / ds / "out" / s / "rolled" / f"stack_t{k:02d}.nii.gz")) for k in range(n)]
    arr = np.stack([np.asarray(im.dataobj, dtype=np.float32) for im in ims], -1)
    ed = [detect_ed(area[z, :n]) for z in range(D)]
    off = np.array([0 if e is None else e for e in ed])
    theta = 2 * np.pi * ((np.arange(n)[None] - off[:, None]) % NCP) / NCP          # (D, n) fit
    theta_out = 2 * np.pi * ((np.arange(N_OUT)[None] - off[:, None]) % NCP) / NCP  # (D, 24) readout
    ed0 = int(off[0])
    arr[:, :, 0, :] = np.roll(arr[:, :, 0, :], -ed0, axis=-1)
    theta[0] = np.roll(theta[0], -ed0)
    assert abs(theta[0, 0]) < 1e-9
    gd.mkdir(parents=True)
    img = nib.Nifti1Image(arr, ims[0].affine)
    img.header.set_zooms((*img.header.get_zooms()[:3], RR_NOMINAL_S / NCP))
    img.header.set_xyzt_units("mm", None)
    nib.save(img, str(gd / "stack4d.nii.gz"))
    (gd / "cardphase.txt").write_text(" ".join(f"{v:.6f}" for v in theta.reshape(-1)) + "\n")
    (gd / "cardphase_readout.txt").write_text(" ".join(f"{v:.6f}" for v in theta_out.reshape(-1)) + "\n")
    (gd / "rrintervals.txt").write_text(" ".join(f"{RR_NOMINAL_S:.6f}" for _ in range(D)) + "\n")
    json.dump({"probe": "nframes", "n_frames_fit": n, "n_out": N_OUT, "n_cardphase": NCP, "D": D,
               "ed_frame": [None if e is None else int(e) for e in ed], "slice0_frame_roll": -ed0,
               "gater": "self", "seg_cache": SEG_TAGS}, open(gd / "gate.json", "w"), indent=1)
    return ed


def setup():
    for src, s in SUBJECTS:
        ds = f"{src}_af24"
        real = REAL / ds / "out" / s
        man = json.load(open(real / "manifest.json"))
        assert man["T"] == 24 and man.get("n_cardphase", 24) == NCP
        live = json.load(open(real / ARM / "gate" / "gate.json"))
        area = lv_area(ds, s, 24)
        for n in NS:
            d = n_root(n) / ds / "out" / s
            d.mkdir(parents=True, exist_ok=True)
            for name in LINKED + ("rolled",):
                if (real / name).exists():
                    _link(d / name, real / name)
            gd = d / ARM / "gate"
            if gd.is_dir():
                continue
            ed = write_gate(ds, s, n, gd, area, man)
            if n == 24:   # must reproduce the live gate from the same segs
                live_ed = [p["ed_frame"] for p in live["per_slice"]]
                assert ed == live_ed, (s, ed, live_ed)
                a = open(gd / "cardphase.txt").read().split()
                b = open(real / ARM / "gate" / "cardphase.txt").read().split()
                assert a == b, f"{s}: 24-frame cardphase differs from the live gate"
        print(f"  {s}: gates n12/n24 written; n24 == live gate (ED frames + thetas)")
    print("setup done")


def run_one(src, s, n):
    import image_metrics
    use_root(n_root(n))
    ds = f"{src}_af24"
    sd = paths.subject_dir(ds, s)
    if paths.recon_stamp(ds, s, ARM, "breath").is_file() and paths.metrics(ds, s, ARM).is_file():
        print(f"[{s} n={n}] already done, skipping")
        return
    if paths.recon_dir(ds, s, ARM, "breath").is_dir():
        raise FileExistsError(f"{paths.recon_dir(ds, s, ARM, 'breath')} exists unstamped -- refusing")
    man = json.load(open(sd / "manifest.json"))
    thick = thickness_mm(man["source"], man["rel_path"], float(man["dz_mm"]))
    env = {**os.environ, "PROBE_SD": str(sd), "EVAL_DATASET": ds, "T": str(man["T"]), "THICK": f"{thick:g}",
           "MASK_FILE": paths.HEART_MASK_PAD, "METHOD": ARM, "INPUT": "breath",
           "OMP": os.environ.get("OMP", os.environ.get("SLURM_CPUS_PER_TASK", "16"))}
    env.pop("T")   # run_fetal4d.sh reads NF/NCP from the manifest only (docs/114)
    r = subprocess.run(["bash", str(SHELL), s, "breath"], env=env)
    if r.returncode != 0 or not paths.recon_stamp(ds, s, ARM, "breath").is_file():
        raise RuntimeError(f"[{s} n={n}] recon failed rc={r.returncode}")
    image_metrics.score_subject(ds, s, ARM)


def report():
    from scipy.stats import wilcoxon
    rows = {n: {} for n in NS}
    for n in NS:
        use_root(n_root(n))
        for src, s in SUBJECTS:
            p = paths.metrics(f"{src}_af24", s, ARM)
            if p.is_file():
                rows[n][s] = json.load(open(p))
    use_root(REAL)
    ref = {}
    for src, s in SUBJECTS:
        p = paths.metrics(f"{src}_af24", s, ARM)
        if p.is_file():
            ref[s] = json.load(open(p))
    common = [s for _, s in SUBJECTS if all(s in rows[n] for n in NS)]
    print(f"subjects with all of N={NS}: {len(common)}/{len(SUBJECTS)}")
    keys = ("breath_psnr_mean", "breath_ssim_mean", "breath_ncc_mean", "breath_psnr_unit_peak_mean")
    print(f"{'N':>4} " + " ".join(f"{k.replace('breath_', '').replace('_mean', ''):>14}" for k in keys))
    for n in NS:
        print(f"{n:>4} " + " ".join(f"{np.mean([rows[n][s][k] for s in common]):>14.4f}" for k in keys))
    for k in keys:
        d = np.array([rows[12][s][k] - rows[24][s][k] for s in common])
        p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d) else np.nan
        print(f"  N=12 - N=24  {k:32} mean {d.mean():+.4f}  min {d.min():+.4f} max {d.max():+.4f}  p={p:.3f}")
    anchored = [s for s in common if s in ref]
    for k in keys:
        d = np.array([rows[24][s][k] - ref[s][k] for s in anchored])
        print(f"  anchor N=24 - live fetal_cmr_4d  {k:32} mean {d.mean():+.4f} maxabs {np.abs(d).max():.4f}  (n={len(anchored)})")
    per = {s: {str(n): {k: rows[n][s][k] for k in keys} for n in NS} for s in common}
    json.dump({"subjects": per, "anchor_live": {s: {k: ref[s][k] for k in keys} for s in anchored}},
              open(PROBE / "report.json", "w"), indent=1)
    print(f"-> {PROBE / 'report.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["setup", "run", "report"])
    ap.add_argument("--index", type=int)
    ap.add_argument("--n", type=int, choices=NS)
    a = ap.parse_args()
    if a.cmd == "setup":
        setup()
    elif a.cmd == "run":
        src, s = SUBJECTS[a.index]
        run_one(src, s, a.n)
    else:
        report()


if __name__ == "__main__":
    main()
