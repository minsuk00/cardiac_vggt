"""CiNeVol baseline (Vogt et al. 2026) over a rhythm cohort: prepare -> fit -> 24-volume export.

Per subject, from scratch (CiNeVol has no training set -- the fit IS its inference):
  1. tools/cinevol_prepare.py   bundle -> observations + R-peak/breath states   (node-local /tmp)
  2. cinevol.fit                published settings: 500 steps, 32768 px, 16 PSF samples, in-vivo
                                loss profile, Grid4D CUDA encoder. `--microbatch` only chunks the
                                SAME batch (exact gradient accumulation), it is not a hyperparameter.
  3. export                     vol_t{00..NF-1}: the fitted model queried at the reference plane's
                                own cardiac label for frame f, breathing 0, 1.4 mm isotropic.
Writes the harness's arm contract (evaluation/paths.py): <subject>/<arm>/recon_breath/
{vol_t*.nii.gz, provenance.txt, total_wall.sec, losses.jsonl, stamp.json}. stamp.json is written
LAST and a stamped subject is skipped, exactly like run_baselines.py. The checkpoint is NOT kept
unless --keep-checkpoint is passed (then also last.pt -- for evaluation/src/analysis/motion_epe/cinevol.py, which
queries the fitted model's own respiratory offset field; see that file for the analysis).

    source baselines/cinevol/env.sh
    $CINEVOL_PY tools/run_cinevol.py --build-only                     # compile the encoder ONCE
    $CINEVOL_PY tools/run_cinevol.py --arm af24 --shard 0 8 [--dry-run]
    $CINEVOL_PY tools/run_cinevol.py --arm af24 --sources cmrx2023 --subjects CMRx23_Test_P005
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import cinevol_prepare as cp  # noqa: E402  (also puts evaluation/, baselines/cinevol, training on sys.path)
import paths  # noqa: E402

SOURCES = ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms")
VARIANT = "breath"


def git_head(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def export_frames(ckpt, devices, shape, affine, cardiac, respiratory, chunk):
    """Query every frame's volume; with several GPUs, contiguous frame blocks run in parallel,
    one model copy per GPU. query() reseeds its own CPU RNG per frame, so the split is exact."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from cinevol.fit import model_from_checkpoint
    from cinevol.reconstruct import query
    blocks = [b.tolist() for b in np.array_split(np.arange(len(cardiac)), len(devices))]

    def run(r):
        torch.cuda.set_device(devices[r])      # Grid4D launches on the current device
        model, state = model_from_checkpoint(ckpt, devices[r])
        return state, [query(model, shape, affine, cardiac[f], respiratory,
                             state["config"]["psf_samples"]["inference"], chunk) for f in blocks[r]]

    with ThreadPoolExecutor(len(devices)) as ex:
        parts = list(ex.map(run, range(len(devices))))
    return parts[0][0], [v for _, vols in parts for v in vols]


def run_subject(ds, subj, arm_name, microbatch, chunk, tmp_root, keep_checkpoint=False, devices=("cuda",)):
    import torch
    from cinevol.fit import fit
    # NEVER OVERWRITE: everything is written to a unique sibling dir and published by ONE rename,
    # which is refused if the final dir already exists (even unstamped -- a human decides then).
    final = paths.recon_dir(ds, subj, arm_name, VARIANT)
    if os.path.exists(final):
        raise FileExistsError(f"{final} exists (unstamped?) -- refusing to write into it; inspect and move it aside")
    out = f"{final}.partial{os.getpid()}_{int(time.time())}"
    work = os.path.join(tmp_root, f"{ds}__{subj}__{os.getpid()}")   # node-local, unique, this tool's own
    os.makedirs(out)                                               # no exist_ok: must be new
    os.makedirs(work)
    sync = lambda: [torch.cuda.synchronize(d) for d in devices]
    sync()
    t0 = time.perf_counter()
    manifest = cp.prepare(ds, subj, os.path.join(work, "prep"))
    t_prep = time.perf_counter() - t0
    ckpt = fit(str(manifest), os.path.join(work, "run"), profile="invivo", backend="grid4d",
               device=devices[0], microbatch=microbatch, seed=7, devices=list(devices))
    sync()
    t_fit = time.perf_counter() - t0 - t_prep
    q = json.load(open(os.path.join(work, "prep", "query_states.json")))
    meta = json.load(open(manifest))
    shape, affine = tuple(meta["output_grid"]["shape"]), np.asarray(meta["output_grid"]["affine"])
    state, vols = export_frames(ckpt, list(devices), shape, affine, q["cardiac"], q["respiratory"], chunk)
    meta, cfg = state["subject_metadata"], state["config"]
    for f, vol in enumerate(vols):
        img = nib.Nifti1Image((vol * meta["intensity_scale"]).astype(np.float32), affine)
        img.header.set_xyzt_units("mm")
        nib.save(img, os.path.join(out, f"vol_t{f:02d}.nii.gz"))
    sync()
    total = time.perf_counter() - t0
    if keep_checkpoint:
        shutil.copy(ckpt, os.path.join(out, "last.pt"))   # for evaluation/src/analysis/motion_epe/cinevol.py (model.offsets)
    shutil.copy(os.path.join(work, "run", "losses.jsonl"), os.path.join(out, "losses.jsonl"))
    last = json.loads(open(os.path.join(work, "run", "losses.jsonl")).read().splitlines()[-1])
    sim = meta["simulation"]
    with open(os.path.join(out, "provenance.txt"), "w") as fh:
        fh.write(f"engine          : CiNeVol reimplementation, baselines/cinevol (upstream e30aecc), repo {git_head(ROOT)}\n"
                 f"encoder         : Grid4D a8992a9 hashencoder, patched (baselines/cinevol/DEVIATIONS.md)\n"
                 f"torch / gpu     : {torch.__version__} / {torch.cuda.get_device_name(devices[0])}"
                 f"{f' x{len(devices)} {list(devices)}' if len(devices) > 1 else ''}\n"
                 f"cohort/subject  : {ds} / {subj}\n"
                 f"input           : breath/ stack, {sim['n_frames']} frames/slice, pixels inside {paths.HEART_MASK_PAD}\n"
                 f"cardiac state   : {sim['cardiac_state']}\n"
                 f"resp state      : {sim['respiratory_state']}\n"
                 f"query           : {q['rule']}\n"
                 f"fit             : {cfg['optimization']} profile={cfg['profile']} microbatch={microbatch}\n"
                 f"final losses    : {last}\n"
                 f"observations    : {state['subject_metadata'].get('protocol')} thickness {sim['slice_thickness_mm']} mm\n"
                 f"wall seconds    : prepare {t_prep:.1f} | fit {t_fit:.1f} | export {total - t_prep - t_fit:.1f} | total {total:.1f}\n"
                 f"peak gpu GB     : {max(torch.cuda.max_memory_allocated(d) for d in devices) / 2**30:.2f}\n")
    open(os.path.join(out, "total_wall.sec"), "w").write(f"{total:.0f}\n")
    stamp = {"engine": "cinevol", "input_stack": "breath", "n_frames": sim["n_frames"],
             "thickness_mm": sim["slice_thickness_mm"], "resolution_mm": cp.RES_MM,
             "cardiac_state": "feinstein_rpeak", "respiratory_state": "true_breath_level",
             "query": "ref_plane_label_psi0", "ref_plane": q["ref_plane"], "train_mask": paths.HEART_MASK_PAD,
             "steps": cfg["optimization"]["steps_per_subject"], "batch_pixels": cfg["optimization"]["batch_observed_pixels"],
             "psf_samples": cfg["psf_samples"]["fitting"], "profile": cfg["profile"], "backend": cfg["backend"],
             "microbatch": microbatch, "gpus": len(devices), "seed": 7, "torch": torch.__version__, "final_mae": last["mae"]}
    json.dump(stamp, open(os.path.join(out, "stamp.json"), "w"))
    if os.path.exists(final):                                # another process published meanwhile
        raise FileExistsError(f"{final} appeared during the run -- keeping {out}, publishing nothing")
    os.rename(out, final)                                    # atomic publish: complete + stamped, or absent
    assert paths.recon_stamp(ds, subj, arm_name, VARIANT).is_file()
    shutil.rmtree(work, ignore_errors=True)                  # this run's own /tmp checkpoint + observations
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="af24", help="rhythm arm: cohorts <source>_<arm>")
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--arm-name", default="cinevol", help="output arm dir under each subject")
    ap.add_argument("--microbatch", type=int, default=None, help="pixels per forward on one GPU "
                    "(memory only; default 8192 on 1 GPU, 32768/N = one chunk per GPU on N GPUs)")
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids (relative to CUDA_VISIBLE_DEVICES, "
                    "at most 4): each fit step's batch and the frame export are split across them")
    ap.add_argument("--chunk", type=int, default=65536, help="export voxels per forward (memory only)")
    ap.add_argument("--tmp", default=f"/tmp/cinevol_{os.environ.get('USER', 'user')}")
    ap.add_argument("--keep-checkpoint", action="store_true", help="also copy last.pt into the "
                    "published recon dir (~80 MB/subject) -- for evaluation/src/analysis/motion_epe/cinevol.py")
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--build-only", action="store_true", help="compile/load the Grid4D encoder and exit")
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(",")]
    if len(set(gpus)) != len(gpus) or not 1 <= len(gpus) <= 4:
        ap.error("--gpus needs 1-4 distinct ids")
    devices = [f"cuda:{g}" for g in gpus]
    if a.microbatch is None:
        a.microbatch = 8192 if len(gpus) == 1 else -(-32768 // len(gpus))

    if not a.dry_run:
        import torch
        assert torch.cuda.is_available(), "CiNeVol's Grid4D encoder needs a CUDA GPU"
        torch.cuda.set_device(devices[0])                        # Grid4D launches on the current device
        from cinevol.encodings import Grid4DHashGrid
        Grid4DHashGrid(2, 2, [4] * 3, [8] * 3, 8).cuda()       # compile (first time) / load, OUTSIDE the timing
        print(f"Grid4D encoder ready: {os.environ.get('GRID4D_BUILD_DIR', './tmp_build/')}", flush=True)
        if a.build_only:
            return

    work = []
    for src in a.sources:
        ds = f"{src}_{a.arm}"
        keep, _ = paths.filter_by_split(ds, paths.subjects(ds), a.split)
        work += [(ds, s) for s in keep if not a.subjects or s in a.subjects]
    if a.shard:
        work = [w for k, w in enumerate(work) if k % a.shard[1] == a.shard[0]]
    todo = [w for w in work if not paths.recon_stamp(w[0], w[1], a.arm_name, VARIANT).is_file()]
    print(f"[{a.arm_name}/{a.arm}] {len(work)} subjects in shard: {len(work) - len(todo)} already stamped, "
          f"{len(todo)} to run", flush=True)
    if a.dry_run:
        for ds, s in todo:
            print(f"  {ds:18s} {s}")
        return
    failed = []
    for k, (ds, s) in enumerate(todo):
        print(f"--- [{k + 1}/{len(todo)}] {ds}/{s} ---", flush=True)
        try:
            sec = run_subject(ds, s, a.arm_name, a.microbatch, a.chunk, a.tmp, a.keep_checkpoint, devices)
            print(f"    done in {sec:.0f} s", flush=True)
        except Exception as e:                                # one bad subject must not kill the shard
            import traceback
            traceback.print_exc()
            failed.append(f"{ds}/{s}: {type(e).__name__}: {e}")
    print(f"DONE [{a.arm_name}/{a.arm}]: {len(todo) - len(failed)} ok, {len(failed)} failed")
    for f in failed:
        print(f"  FAILED: {f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
