"""Frames-per-slice probe for CiNeVol on af24: fit on the first N in {12, 24, 48} frames of ONE
48-frame AF timeline, score every fit against the SAME 24 GT volumes. Only N varies.

Why this is a clean probe: build_af_bundle.simulate draws each plane's beats from a private RNG
stream that does not depend on n_frames, so an `af48` build's frames 0..23 are bit-identical to
the real af24 bundle (asserted in `setup`). The 12/24/48 inputs are nested prefixes of one record.

Everything lives under temp/cinevol_nframes_probe/ (gitignored). The harness's paths.VOLUMES is
redirected there at runtime; scratch/eval and evaluation/ are read-only for this script. Inputs
are symlinked, so nothing is copied. Each N gets its own cohort root (n12/ n24/ n48/) so the arm
can keep the name `cinevol` -- pose_psf.base_method() only recognises that literal, and a renamed
arm would silently lose the PSF operator every other CiNeVol number was scored with.

    source baselines/cinevol/env.sh
    $CINEVOL_PY tools/cinevol_nframes_probe.py setup                 # symlinks + af48 build + checks
    $CINEVOL_PY tools/cinevol_nframes_probe.py run --index 3 [--ns 12 24 48]   # one subject, GPU
    $CINEVOL_PY tools/cinevol_nframes_probe.py report                # paired table
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("tools", "evaluation", "evaluation/src/score", "evaluation/src/engine", "baselines/cinevol", "training"):
    sys.path.insert(0, str(ROOT / p))
import paths                      # noqa: E402
import cinevol_prepare as cp      # noqa: E402

PROBE = ROOT / "temp" / "cinevol_nframes_probe"
BUILD = PROBE / "build"           # builder eval-root: <src>/out/<s> -> real bundle, <src>_af48/out/<s> built here
REAL = paths.VOLUMES.resolve()    # scratch/eval, read-only here
ARM = "cinevol"
NS = (12, 24, 48)
N_QUERY = 24                      # every fit is queried at the af24 GT frames
# 2 test subjects per source (first two with a scored cinevol_masked arm at the time of writing).
SUBJECTS = [("cmrx2023", "CMRx23_Test_P005"), ("cmrx2023", "CMRx23_Test_P015"),
            ("cmrx2024", "CMRx24_Test_P001"), ("cmrx2024", "CMRx24_Test_P004"),
            ("cmrx2025", "CMRx25_R1test_Center002_Siemens_30T_CIMAX_P017"),
            ("cmrx2025", "CMRx25_R1test_Center003_UIH_30T_umr880_P012"),
            ("acdc", "ACDC_patient102"), ("acdc", "ACDC_patient105"),
            ("mnms", "MNMs_A1K2P5"), ("mnms", "MNMs_A4K8R4")]
LINKED = ("manifest.json", "mask.nii.gz", "mask_fov.nii.gz", "mask_heart.nii.gz", "mask_heart_pad10.nii.gz",
          "gt", "breath", "cine_gt.nii.gz", "cine_gt.src.json")


def n_root(n):
    return PROBE / f"n{n}"


def use_root(root):
    """Point the harness at a probe root. dataset_root() reads paths.VOLUMES at call time."""
    paths.VOLUMES = Path(root)


def _link(dst, src):
    if not dst.exists() and not dst.is_symlink():
        os.symlink(src, dst)


def setup():
    for src, s in SUBJECTS:
        ds = f"{src}_af24"
        keep, _ = paths.filter_by_split(ds, [s], "test")
        assert keep == [s], f"{ds}/{s} is not a test subject"
        real = REAL / ds / "out" / s
        for n in NS:                                   # scoring roots: real inputs, probe arms
            d = n_root(n) / ds / "out" / s
            d.mkdir(parents=True, exist_ok=True)
            for name in LINKED:
                if (real / name).exists():
                    _link(d / name, real / name)
        b = BUILD / src / "out"                        # builder source = the base (gated) bundle
        b.mkdir(parents=True, exist_ok=True)
        _link(b / s, REAL / src / "out" / s)
    by_src = {}
    for src, s in SUBJECTS:
        by_src.setdefault(src, []).append(s)
    for src, subs in by_src.items():
        subprocess.check_call([sys.executable, str(ROOT / "tools/build_af_bundle.py"), "--source", src,
                               "--subjects", ",".join(subs), "--arms", "af48", "--eval-root", str(BUILD)])
    # The whole probe rests on this: af48 frames/GT 0..23 == the real af24 bundle.
    for src, s in SUBJECTS:
        real, b48 = REAL / f"{src}_af24" / "out" / s, BUILD / f"{src}_af48" / "out" / s
        m24, m48 = json.load(open(real / "manifest.json")), json.load(open(b48 / "manifest.json"))
        assert m48["rhythm"]["params"]["n_frames"] == 48 and m24["T"] == 24
        for k in ("rr_per_plane", "beat_per_plane", "pos_per_plane", "ref_plane"):
            a, c = m24["rhythm"][k], m48["rhythm"][k]
            if isinstance(a, list) and isinstance(a[0], list):
                assert all(np.allclose(x[:24], y[:24]) for x, y in zip(a, c)), (s, k)
            elif isinstance(a, list):
                assert np.allclose(a, c), (s, k)
            else:
                assert a == c, (s, k)
        for f in range(24):
            for sub, pre in (("breath", "stack"), ("gt", "gt")):
                x = np.asarray(nib.load(real / sub / f"{pre}_t{f:02d}.nii.gz").dataobj)
                y = np.asarray(nib.load(b48 / sub / f"{pre}_t{f:02d}.nii.gz").dataobj)
                assert x.shape == y.shape and np.array_equal(x, y), (s, sub, f)
        print(f"  {s}: af48 frames+GT 0..23 == af24  OK")
    print("setup done")


def prepare_n(src, s, n, out):
    """cinevol_prepare.prepare, but observations = the first n frames of the af48 record; labels
    from the af48 manifest (same timeline); queries = the reference plane's phase at the 24
    af24 GT frames. Mirrors cp.prepare line for line otherwise."""
    b48 = BUILD / f"{src}_af48" / "out" / s
    man = json.load(open(b48 / "manifest.json"))
    D, ref = man["D"], man["rhythm"]["ref_plane"]
    thick = cp.thickness_mm(man["source"], man["rel_path"], float(man["dz_mm"]))
    imgs = [nib.load(b48 / "breath" / f"stack_t{f:02d}.nii.gz") for f in range(n)]
    affine = imgs[0].affine
    mask_img = nib.load(b48 / paths.HEART_MASK_PAD)
    assert np.allclose(mask_img.affine, affine, atol=1e-4) and mask_img.shape == imgs[0].shape
    mask = np.asarray(mask_img.dataobj) > 0
    lo, hi, shape, grid = cp.crop_and_grid(mask, affine)
    sl = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
    cine = np.stack([np.asarray(im.dataobj, dtype=np.float32)[sl] for im in imgs], -1)
    cine[~mask[sl]] = np.nan
    keep = list(range(lo[2], hi[2] + 1))
    assert lo[2] <= ref <= hi[2]
    crop_aff = affine.copy()
    crop_aff[:3, 3] = nib.affines.apply_affine(affine, lo)
    phi, psi = cp.cardiac_labels(man), cp.resp_labels(man)     # (D, 48)
    assert phi.shape[1] == 48
    os.makedirs(out)
    nib.save(nib.Nifti1Image(cine, crop_aff), os.path.join(out, "sax_observed.nii.gz"))
    np.save(os.path.join(out, "cardiac_states.npy"), phi[keep, :n].astype(np.float32))
    np.save(os.path.join(out, "respiratory_states.npy"), psi[keep, :n].astype(np.float32))
    queries = {"cardiac": [float(x) for x in phi[ref, :N_QUERY]], "respiratory": 0.0, "ref_plane": int(ref),
               "rule": "reference plane's own Feinstein phase at af24 frame f; psi=0"}
    json.dump(queries, open(os.path.join(out, "query_states.json"), "w"), indent=1)
    sim = {"name": "vggt_rhythm_bundle_nframes_probe", "cohort": f"{src}_af48", "arm": "af48",
           "seed": man["seed"], "n_frames": n, "n_frames_record": 48, "slice_thickness_mm": thick,
           "dz_mm": float(man["dz_mm"]),
           "cardiac_state": "Feinstein bilinear phase (paper Eq.16) from simulated R-peak times",
           "respiratory_state": "true breath level sin^{2n}(pi r) in [0,1]"}
    spec = {"subject": s, "protocol": "acdc_single_sax_simulated", "simulation": sim,
            "state_provenance": sim["cardiac_state"] + " | " + sim["respiratory_state"],
            "reference_cardiac_state": queries["cardiac"][0], "end_expiration_state": 0.0,
            "output_grid": {"shape": shape, "affine": grid.tolist()},
            "stacks": [{"stack_id": "SAX", "image": "sax_observed.nii.gz", "slice_thickness_mm": thick,
                        "cardiac_states": "cardiac_states.npy", "respiratory_states": "respiratory_states.npy"}]}
    json.dump(spec, open(os.path.join(out, "acquisitions.json"), "w"), indent=1)
    from cinevol.prepare import prepare as cinevol_prepare
    return cinevol_prepare(os.path.join(out, "acquisitions.json"), os.path.join(out, "prepared"))


def zero_outside(src_dir, dst_dir, mask_path):
    """tools/cinevol_maskzero_score.make_zeroed, same nearest-slab mask resample (docs/116 s4c)."""
    from scipy.ndimage import map_coordinates
    mask_img = nib.load(str(mask_path))
    m = None
    for f in range(N_QUERY):
        v = nib.load(os.path.join(src_dir, f"vol_t{f:02d}.nii.gz"))
        if m is None:
            ijk = np.stack(np.meshgrid(*[np.arange(k) for k in v.shape], indexing="ij"), -1).reshape(-1, 3)
            sidx = nib.affines.apply_affine(np.linalg.inv(mask_img.affine) @ v.affine, ijk)
            m = map_coordinates(np.asarray(mask_img.dataobj, np.float32), sidx.T, order=0,
                                mode="nearest").reshape(v.shape) > 0.5
        nib.save(nib.Nifti1Image(np.asarray(v.dataobj, np.float32) * m, v.affine),
                 os.path.join(dst_dir, f"vol_t{f:02d}.nii.gz"))


def run_one(src, s, n, microbatch, chunk, tmp_root):
    import torch
    from cinevol.fit import fit, model_from_checkpoint
    from cinevol.reconstruct import query
    import image_metrics
    use_root(n_root(n))
    ds = f"{src}_af24"
    final = paths.recon_dir(ds, s, ARM, "breath")
    if paths.recon_stamp(ds, s, ARM, "breath").is_file() and paths.metrics(ds, s, ARM).is_file():
        print(f"[{s} n={n}] already done, skipping")
        return
    if final.exists():
        raise FileExistsError(f"{final} exists unstamped -- refusing; move it aside")
    work = Path(tmp_root) / f"{ds}__{s}__n{n}__{os.getpid()}"
    out = Path(f"{final}.partial{os.getpid()}_{int(time.time())}")
    out.mkdir(parents=True)
    work.mkdir(parents=True)
    raw = out / "export_raw"                              # unmasked export, kept for inspection
    raw.mkdir()
    t0 = time.perf_counter()
    manifest = prepare_n(src, s, n, str(work / "prep"))
    ckpt = fit(str(manifest), str(work / "run"), profile="invivo", backend="grid4d",
               device="cuda", microbatch=microbatch, seed=7)
    torch.cuda.synchronize()
    t_fit = time.perf_counter() - t0
    model, state = model_from_checkpoint(ckpt, "cuda")
    meta, cfg = state["subject_metadata"], state["config"]
    q = json.load(open(work / "prep" / "query_states.json"))
    shape, affine = tuple(meta["output_grid"]["shape"]), np.asarray(meta["output_grid"]["affine"])
    for f, phi in enumerate(q["cardiac"]):
        vol = query(model, shape, affine, phi, q["respiratory"], cfg["psf_samples"]["inference"], chunk) \
            * meta["intensity_scale"]
        img = nib.Nifti1Image(vol.astype(np.float32), affine)
        img.header.set_xyzt_units("mm")
        nib.save(img, str(raw / f"vol_t{f:02d}.nii.gz"))
    zero_outside(raw, out, paths.heart_mask_pad(ds, s))
    shutil.copy(work / "run" / "losses.jsonl", out / "losses.jsonl")
    last = json.loads(open(work / "run" / "losses.jsonl").read().splitlines()[-1])
    total = time.perf_counter() - t0
    stamp = {"engine": "cinevol", "probe": "nframes", "n_frames": n, "n_query": N_QUERY,
             "thickness_mm": meta["simulation"]["slice_thickness_mm"], "resolution_mm": cp.RES_MM,
             "cardiac_state": "feinstein_rpeak", "respiratory_state": "true_breath_level",
             "zeroed_outside": paths.HEART_MASK_PAD, "steps": cfg["optimization"]["steps_per_subject"],
             "batch_pixels": cfg["optimization"]["batch_observed_pixels"], "profile": cfg["profile"],
             "backend": cfg["backend"], "microbatch": microbatch, "seed": 7, "torch": torch.__version__,
             "gpu": torch.cuda.get_device_name(), "final_mae": last["mae"], "wall_fit_s": t_fit, "wall_s": total}
    json.dump(stamp, open(out / "stamp.json", "w"), indent=1)
    open(out / "total_wall.sec", "w").write(f"{total:.0f}\n")
    os.rename(out, final)
    shutil.rmtree(work, ignore_errors=True)
    print(f"[{s} n={n}] fit {t_fit:.0f}s total {total:.0f}s mae {last['mae']:.4f}", flush=True)
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
    for src, s in SUBJECTS:                              # anchor: the real cinevol_masked arm (24 frames)
        p = paths.metrics(f"{src}_af24", s, "cinevol_masked")
        if p.is_file():
            ref[s] = json.load(open(p))
    common = [s for _, s in SUBJECTS if all(s in rows[n] for n in NS)]
    print(f"subjects with all of N={NS}: {len(common)}/{len(SUBJECTS)}")
    keys = ("breath_psnr_mean", "breath_ssim_mean", "breath_ncc_mean", "breath_psnr_unit_peak_mean")
    print(f"{'N':>4} " + " ".join(f"{k.replace('breath_', '').replace('_mean', ''):>14}" for k in keys) + "   fit_s")
    for n in NS:
        v = [np.mean([rows[n][s][k] for s in common]) for k in keys]
        fs = np.median([rows[n][s].get("wall_fit_s", np.nan) for s in common]) if common else np.nan
        print(f"{n:>4} " + " ".join(f"{x:>14.4f}" for x in v) + f"   -")
    for n in (12, 48):
        for k in keys:
            d = np.array([rows[n][s][k] - rows[24][s][k] for s in common])
            p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d) else np.nan
            print(f"  N={n} - N=24  {k:32} mean {d.mean():+.4f}  min {d.min():+.4f} max {d.max():+.4f}  p={p:.3f}")
    anchored = [s for s in common if s in ref]
    if anchored:
        for k in keys:
            d = np.array([rows[24][s][k] - ref[s][k] for s in anchored])
            print(f"  anchor N=24 - real cinevol_masked  {k:32} mean {d.mean():+.4f} maxabs {np.abs(d).max():.4f}  (n={len(anchored)})")
    per = {s: {str(n): {k: rows[n][s][k] for k in keys} for n in NS} for s in common}
    json.dump({"subjects": per, "anchor_cinevol_masked": {s: {k: ref[s][k] for k in keys} for s in anchored}},
              open(PROBE / "report.json", "w"), indent=1)
    print(f"-> {PROBE / 'report.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["setup", "run", "report"])
    ap.add_argument("--index", type=int, help="run: SUBJECTS index")
    ap.add_argument("--ns", nargs="+", type=int, default=list(NS))
    ap.add_argument("--microbatch", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--tmp", default=f"/tmp/cinevol_probe_{os.environ.get('USER', 'user')}")
    a = ap.parse_args()
    if a.cmd == "setup":
        setup()
    elif a.cmd == "run":
        src, s = SUBJECTS[a.index]
        for n in a.ns:
            run_one(src, s, n, a.microbatch, a.chunk, a.tmp)
    else:
        report()


if __name__ == "__main__":
    main()
