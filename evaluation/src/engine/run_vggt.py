#!/usr/bin/env python
"""Score a VGGT-MRI checkpoint on the frozen breathing bundles — the GPU head-to-head vs SVR.

The GPU analog of `engine/run_svrtk3d.sh` / `run_nesvor.sh`: load a checkpoint ONCE, loop subjects,
write per-subject recons into `<source>/out/<subject>/<arm>/` so the SAME `score/image_metrics.py`
scorer and `aggregate.py` roll-up consume it identically to the classical baselines.

## The one idea in this file

**The harness does not build batches. It asks the trainer for one.**

    batch = ds.get_data(seq_index=...)          # the run's own sampler + geometry
    batch["phases"] = frozen_bundle             # swap clean pixels -> frozen breathed pixels
    batch.pop("images")                         # force re-extraction from the swapped phases
    batch = gpu_augment_batch(batch, None, device, respiratory_cfg=None, train=False)
    preds = model(batch["images"], batch=batch)
    V, cov = _splat_preds_native(preds, batch, grid_shape, z_scale)

That is `trainer.val_epoch` with breathing supplied by frozen pixels instead of re-sampled ones.
Everything geometric — `scanner_coords`, `z_indices`, `dz_mm`, `z_scale`, the one-frame-per-slice
slot draw, the reference slot at z_mid — comes from `MRIDataset.get_data`, so there is exactly ONE
implementation of the native-z contract (docs/58) and it is the one training uses. The previous
version of this file hand-wrote its own copy, which is how it silently kept the retired fixed-12
plane convention `z/(D-1)*2-1` long after training moved to physical `(z-(D-1)/2)*dz/90`.

Three consequences worth stating, because they used to be CLI flags:

* **No `--regime`.** Whether this is a one-frame-per-slice model is a property of the run
  (`one_frame_per_slice`), read from its `run_meta.jsonl`. Feeding a 1-frame model a multiframe
  reference burst piles constant companions on the reference plane, which the splat's coverage
  mean averages away — the "frozen recon" artifact. The eval regime now cannot disagree with
  training.
* **No `--continuous-z`.** Same: `continuous_z` is the run's knob. There is no 12 mm snap to
  opt out of any more — z is native per subject.
* **No `--refiner`.** It was a silent no-op: `VGGT.__init__` absorbs `enable_refiner` via
  `**kwargs` and builds nothing.

## Why frozen breathing rather than re-simulated

Re-applying breathing per run (what `inference/run_cmrxrecon.py` did) draws a DIFFERENT realization
than the one the classical baselines were given, so the comparison is not same-input. Here the
breathed pixels are read off disk byte-identically to what SVRTK saw, and GT stays the unshifted
`gt/` bundle. Model and baselines provably share one corruption, one target, one ROI, one scorer.

## Reference-phase sweep

Slot 0 is the target-phase reference (docs/25). To query phase `t` we set slot 0's `timesteps`
entry to `t` and re-extract — the companions are untouched, so it is the same inputs with a
varying query, mirroring the trainer's cardiac-cycle filmstrip.

Run:
  micromamba run -n svr env PYTHONPATH=training:. python evaluation/src/engine/run_vggt.py \
      --dataset cmrx2024 --ckpt scratch/logs/<run>/ckpts/checkpoint_last.pt \
      --model-name augaggr224hw2_ep300
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import nibabel as nib
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import paths                                                                   # noqa: E402
from data.datasets.mri_dataset import MRIDataset                               # noqa: E402
from data.gpu_aug import gpu_augment_batch                                     # noqa: E402
from data.preprocess import Z_HALF_MM                                          # noqa: E402
from loss import _splat_preds_native                                           # noqa: E402
from inference.load_run import load_model_from_run, mri_dataset_kwargs         # noqa: E402
from omegaconf import OmegaConf                                                # noqa: E402

ED_PHASE = 0
INPLANE_MM = 1.4
# normalized [-1,1] -> mm. In-plane: (256-1)/2 * 1.4. Through-plane: Z_HALF_MM, which is a
# CONSTANT for every subject by construction under physical z (docs/58) — using a per-subject
# (D-1)/2*dz here would systematically understate Δz, exactly as loss.py warns.
MM_PER_NORM = (0.5 * (256 - 1) * INPLANE_MM, 0.5 * (256 - 1) * INPLANE_MM, Z_HALF_MM)


def name_seed(source, name):
    """Same stable hash the bundle builder uses — see `build_inputs/pooled.py`."""
    return int(hashlib.sha256(f"{source}/{name}".encode()).hexdigest(), 16) % (2 ** 31)


def _load_xyz_to_dhw(path):
    """(X,Y,Z) NIfTI -> (D,H,W)=(Z,Y,X) splat order (inverse of the builder's save_xyz)."""
    return np.transpose(np.asarray(nib.load(path).dataobj, dtype=np.float32), (2, 1, 0))


def load_bundle(subj_dir, T, kind):
    """The frozen (T,D,H,W) stack. kind in {'gt','clean','breath'}."""
    pre = "gt" if kind == "gt" else "stack"
    return np.stack([_load_xyz_to_dhw(os.path.join(subj_dir, kind, f"{pre}_t{t:02d}.nii.gz"))
                     for t in range(T)])


def make_dataset(cfg, subject_rel, split, tmpdir):
    """A ONE-SUBJECT MRIDataset, built with the run's own knobs.

    One subject per dataset is what decouples the slot draw from cohort composition.
    `get_data` uses `seq_index` for BOTH the subject index (`seq_index % len(subjects)`) and the
    val RNG seed (`random.Random(seq_index)`). With a single subject the index term is always 0,
    so `seq_index` is free to be the subject's NAME hash — the draw then depends only on the
    subject, never on where it sits in the split file or on how many subjects are in the cohort.

    `ef_val_sweep` is dropped: this runner sweeps the reference phase over all T itself, and the
    sweep's ED/ES pairing would otherwise double the dataset and re-couple seq_index to position.
    """
    kw = dict(mri_dataset_kwargs(cfg, "val"))
    kw.pop("ef_val_sweep", None)
    kw.pop("split", None)
    kw.pop("split_file", None)
    data_root = kw.pop("data_root", os.path.join(ROOT, "scratch/data"))
    kw.pop("_target_", None)
    sf = os.path.join(tmpdir, "one_subject.txt")
    with open(sf, "w") as f:
        f.write(f"[{split}]\n{subject_rel}\n")
    common = OmegaConf.create({"img_size": int(cfg.get("img_size", 518)), "patch_size": 14,
                               "rescale": True, "rescale_aug": False, "landscape_check": False,
                               "augs": {"scales": [1.0, 1.0]}})
    # t_target_fixed=0 pins slot 0 to ED for the base draw; the sweep overwrites slot-0's t per
    # queried phase, so the COMPANION slots stay fixed across the sweep (same inputs, varying query).
    kw["t_target_fixed"] = 0
    kw["t_target_phases"] = None
    return MRIDataset(common, data_root, split=split, split_file=sf, **kw)


def build_batch(ds, seq_index, phases_bundle, device, dz_bundle=None):
    """`get_data` -> swap in the frozen pixels -> let the trainer's own code finish the batch."""
    b = ds.get_data(seq_index=seq_index, img_per_seq=ds.num_slices)
    D_ds = int(np.asarray(b["phases"]).shape[1])
    if phases_bundle.shape[1] != D_ds:
        raise ValueError(f"bundle D={phases_bundle.shape[1]} != dataset D={D_ds}")
    if phases_bundle.shape[0] != np.asarray(b["phases"]).shape[0]:
        raise ValueError(f"bundle T={phases_bundle.shape[0]} != dataset T={np.asarray(b['phases']).shape[0]}")
    # dz guard: a pitch relabel of the source NIfTIs (has happened twice — docs/27, docs/56)
    # usually keeps D, so D/T alone can't catch a bundle breathed at one dz being splatted/scored
    # at another. The manifest froze dz at build time; the dataset carries today's.
    if dz_bundle is not None:
        dz_ds = float(np.asarray(b["dz_mm"]).reshape(-1)[0])
        if abs(dz_ds - float(dz_bundle)) > 1e-3:
            raise ValueError(f"bundle dz_mm={dz_bundle} != dataset dz_mm={dz_ds} — "
                             f"rebuild the bundle (pitch changed since it was frozen)")

    # Stand in for the trainer's collate (batch size 1). `get_data` returns a mix of ndarrays and
    # python lists — `timesteps`/`slice_indices` are list[int], the per-slot fields are
    # list[ndarray] — and gpu_aug indexes them as (B,S) tensors. Dtypes match the contract in
    # `gpu_augment_batch`'s docstring: timesteps int64, slice_indices float32 (it may be
    # continuous under continuous_z).
    _DTYPE = {"timesteps": torch.int64, "slice_indices": torch.float32}
    out = {}
    for k, v in b.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v)[None].to(device)
        elif torch.is_tensor(v):
            out[k] = v[None].to(device)
        elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
            out[k] = torch.from_numpy(np.stack(v))[None].to(device)
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            out[k] = torch.tensor(v, dtype=_DTYPE.get(k, torch.float32))[None].to(device)
        else:
            out[k] = v                      # str / scalar metadata (seq_name); passed through
    # THE swap. `images` must go too: gpu_augment_batch only rebuilds it when absent, so leaving
    # the dataset's clean `images` in place would feed the model CLEAN slices while the splat
    # rendered BREATHED content — a silent, invisible inconsistency.
    out["phases"] = torch.from_numpy(phases_bundle)[None].to(device)
    out.pop("images", None)
    out.pop("images_splat", None)
    return out


def _extract(batch, device):
    """Re-run the trainer's finisher with BOTH augmentations off: rebuilds `images_splat` (native
    resolution, what the loss/render splats) and `images` (the model input) from `phases` at the
    batch's own (timesteps, slice_indices). This is the identity path through gpu_augment_batch."""
    batch.pop("images", None)
    return gpu_augment_batch(batch, None, device, respiratory_cfg=None, train=False)


@torch.no_grad()
def reconstruct(model, ds, seq_index, phases_bundle, device, disp_applied, dz_bundle=None):
    """Sweep the reference phase over T -> (pred_vols (T,D,H,W), per_phase_ms, ed_pack)."""
    batch = build_batch(ds, seq_index, phases_bundle, device, dz_bundle=dz_bundle)
    T = phases_bundle.shape[0]
    D = phases_bundle.shape[1]
    grid_shape = (D, 256, 256)
    z_scale = float(batch["z_scale"].reshape(-1)[0])

    pred_vols, per_phase_ms, ed_pack = [], [], None
    for t in range(T):
        batch["timesteps"][0, 0] = int(t)            # slot 0 = the reference at the queried phase
        _extract(batch, device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
        wp = preds["world_points"].float()
        V, _cov = _splat_preds_native({"world_points": wp}, batch, grid_shape, z_scale)
        torch.cuda.synchronize(); per_phase_ms.append((time.perf_counter() - t0) * 1e3)
        pred_vols.append(V[0].float().cpu().numpy())
        if t == ED_PHASE:
            delta = (wp[0] - batch["scanner_coords"][0].float()).cpu().numpy()   # (S,R,R,3) normalized
            z_slots = batch["slice_indices"][0].cpu().numpy()
            ed_pack = dict(delta=delta,
                           # (S,3,R,R) -> (S,R,R): the FOV gate is on slice content, not colour.
                           images=batch["images"][0].float().mean(1).cpu().numpy(),
                           slot_z=z_slots,
                           slot_t=batch["timesteps"][0].cpu().numpy(),
                           applied=np.stack([disp_applied[int(round(float(z)))] for z in z_slots])
                           if disp_applied is not None else np.zeros((len(z_slots), 3)))
    return np.stack(pred_vols), per_phase_ms, ed_pack


def resp_diag(ed_pack, breathing):
    """Predicted through-plane Δz (mm) vs the APPLIED sim shift per slot at ED.

    Deliberately mirrors `loss.py`'s `metric_resp_slope_dz`: all slots including slot 0, the same
    0.05 FOV intensity gate, and `Z_HALF_MM` (not a per-subject half-span) as the normalized->mm
    factor. For the clean arm `applied` is 0, making it a negative control: predicted Δz on
    un-breathed input should sit near zero.
    """
    if ed_pack is None:
        return {}
    delta, imgs = ed_pack["delta"], ed_pack["images"]
    pred_dz, appl_dz = [], []
    for s in range(delta.shape[0]):
        m = imgs[s] > 0.05
        if not m.any():
            continue
        pred_dz.append(float(delta[s, ..., 2][m].mean() * MM_PER_NORM[2]))
        appl_dz.append(float(ed_pack["applied"][s, 0]) if breathing else 0.0)
    pred_dz, appl_dz = np.asarray(pred_dz), np.asarray(appl_dz)
    out = {"breathing": bool(breathing), "n_slots": int(pred_dz.size),
           "pred_dz_mm": pred_dz.tolist(), "applied_dz_mm": appl_dz.tolist(),
           "epe_dz_mm": float(np.mean(np.abs(pred_dz - appl_dz))) if pred_dz.size else None}
    if pred_dz.size >= 2 and appl_dz.std() > 1e-6:
        out["slope"] = float(np.polyfit(appl_dz, pred_dz, 1)[0])
        out["corr"] = float(np.corrcoef(appl_dz, pred_dz)[0, 1])
    return out


# ── provenance / guards ──────────────────────────────────────────────────────────────────────
def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _wandb_id(ckpt):
    runs = glob.glob(os.path.join(os.path.dirname(ckpt), "..", "wandb", "wandb", "run-*"))
    return runs[0].split("-")[-1] if runs else "unknown"


def _ckpt_fingerprint(ckpt):
    try:
        st = os.stat(ckpt)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return None


def _same_ckpt(prev, ident):
    pf, cf = prev.get("ckpt_fingerprint"), ident.get("ckpt_fingerprint")
    if pf and cf:
        return pf == cf
    return os.path.realpath(prev.get("ckpt") or "") == os.path.realpath(ident.get("ckpt") or "")


def check_overwrite(ds_name, subjects, method, ident, overwrite):
    """Refuse to write into an arm whose existing recons came from a DIFFERENT checkpoint, unless
    --overwrite. Fails fast BEFORE the ~3 min model load."""
    conflicts = []
    for subject in subjects:
        mpath = str(paths.metadata(ds_name, subject, method))
        if not os.path.exists(mpath):
            continue
        try:
            with open(mpath) as fh:
                prev = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[run_vggt] WARNING: unreadable {mpath} ({e}); treating as fresh", flush=True)
            continue
        if not isinstance(prev, dict):
            print(f"[run_vggt] WARNING: {mpath} is not a JSON object; treating as fresh", flush=True)
            continue
        if not _same_ckpt(prev, ident):
            conflicts.append((subject, prev.get("ckpt")))
    if conflicts and not overwrite:
        detail = "\n".join(f"    {s}: {p}" for s, p in conflicts[:5])
        more = "" if len(conflicts) <= 5 else f"\n    ... +{len(conflicts) - 5} more"
        sys.exit(f"REFUSING to overwrite arm '{method}': {len(conflicts)} subject(s) came from a "
                 f"DIFFERENT checkpoint.\n  this run: {ident['ckpt']}\n  on disk:\n{detail}{more}\n"
                 f"  -> pass --overwrite, or use a different --model-name.")
    if conflicts:
        print(f"[run_vggt] --overwrite: replacing {len(conflicts)} subject(s) of arm '{method}'",
              flush=True)


def check_bundle_split(ds_name, subjects, split):
    """Drop bundles built for a different split (rule + rationale: paths.filter_by_split)."""
    keep, dropped = paths.filter_by_split(ds_name, subjects, split)
    for s, why in dropped:
        print(f"[run_vggt] SKIP {s}: {why}", flush=True)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(paths.DATASETS))
    ap.add_argument("--ckpt", required=True,
                    help="path INSIDE the run's log_dir; the protocol is read from its run_meta.jsonl")
    ap.add_argument("--model-name", required=True,
                    help="arm slug; output dir is vggt_<model-name>. Required so a stray run cannot "
                         "silently overwrite a named arm.")
    ap.add_argument("--date", default=None, help="legacy arm-name form vggt_<date>_<model>")
    ap.add_argument("--split", default="val", help="which split the bundles were built for")
    ap.add_argument("--arms", nargs="+", default=["breath"], choices=["clean", "breath"],
                    help="input condition(s) to reconstruct. DEFAULT `breath` — that is the "
                         "deliverable: the model targets free-breathing acquisition, so the "
                         "breathing-corrupted input is what it is actually evaluated on. Add "
                         "`clean` (`--arms clean breath`) only when you want the no-breathing "
                         "PSNR ceiling; it roughly doubles runtime and buys nothing else. Its "
                         "negative-control value is redundant, measured: predicted |dz| is 0.47mm "
                         "on the clean arm AND 0.47mm on the breath arm's own near-zero-"
                         "displacement slots, and slope is regressed against VARYING applied "
                         "displacement inside the breath arm, so a constant-dz model already "
                         "scores slope~0 there without any clean run.")
    ap.add_argument("--subjects", nargs="*", default=None, help="default: all built subjects")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--note", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    method = paths.canonical_arm(args.model_name, date=args.date)
    ds_name = args.dataset
    root = paths.dataset_root(ds_name)
    subjects = args.subjects or paths.subjects(ds_name)
    if not subjects:
        sys.exit(f"no built subjects under {root}/*/manifest.json — run build_inputs/pooled.py first")
    subjects = check_bundle_split(ds_name, subjects, args.split)
    if args.limit:
        subjects = subjects[: args.limit]
    if not subjects:
        sys.exit(f"no subjects left for split '{args.split}'")

    ident = {"ckpt": args.ckpt, "ckpt_fingerprint": _ckpt_fingerprint(args.ckpt)}
    check_overwrite(ds_name, subjects, method, ident, args.overwrite)

    device = torch.device("cuda")
    t0 = time.perf_counter()
    model, cfg = load_model_from_run(args.ckpt, device=device)
    model_load_s = time.perf_counter() - t0
    metadata = {
        "method": method, "model_name": args.model_name, "date": args.date,
        "ckpt": args.ckpt, "ckpt_fingerprint": _ckpt_fingerprint(args.ckpt),
        "exp_name": cfg.get("exp_name"), "wandb_id": _wandb_id(args.ckpt),
        "img_size": cfg.get("img_size"), "backbone": cfg.get("backbone") or "dinov2_vitl14_reg",
        "one_frame_per_slice": cfg.get("one_frame_per_slice"),
        "continuous_z": cfg.get("continuous_z"), "reference_slot": cfg.get("reference_slot"),
        "aug_tier": ((cfg.get("data") or {}).get("augmentation") or {}).get("tier"),
        "protocol_source": "run_meta.jsonl (NOT the live default.yaml)",
        "breathing_source": "frozen (eval bundle breath/ pixels + manifest disp; NOT re-sampled)",
        "geometry": "native-z (docs/58): per-subject D and dz, z_scale=Z_HALF_MM/dz, no 12mm snap",
        "git_commit": _git_commit(), "note": args.note,
    }
    print(f"[run_vggt] {ds_name}: arm={method} subjects={len(subjects)} "
          f"(model load {model_load_s:.0f}s)", flush=True)

    for subject in subjects:
        subj_dir = str(paths.subject_dir(ds_name, subject))
        man = json.load(open(paths.manifest(ds_name, subject)))
        T = man["T"]
        disp = np.asarray(man["breath"]["disp_dhw_mm"], dtype=np.float64)   # (D,3) per z-plane
        md = str(paths.arm_dir(ds_name, subject, method))
        os.makedirs(md, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            dset = make_dataset(cfg, man["rel_path"], args.split, tmpdir)
            seq = name_seed(ds_name, subject)          # name-keyed: cohort-composition independent
            # metadata_draw is filled by the breath arm only (it owns ed_dvf.npz), but metadata.json
            # is written for EVERY arm — initialise it or `--arms clean` raises UnboundLocalError
            # AFTER the recons are on disk, leaving an arm with no metadata for check_overwrite.
            timing, rdiag, metadata_draw = {"model_load_sec": model_load_s}, {}, {}
            for breathing, var in [(b, v) for b, v in ((False, "clean"), (True, "breath"))
                                   if v in args.arms]:
                rv = str(paths.recon_dir(ds_name, subject, method, var))
                os.makedirs(rv, exist_ok=True)
                bundle = load_bundle(subj_dir, T, "breath" if breathing else "clean")
                ts = time.perf_counter()
                pred_vols, per_phase_ms, ed_pack = reconstruct(
                    model, dset, seq, bundle, device, disp if breathing else None,
                    dz_bundle=man["dz_mm"])
                wall = time.perf_counter() - ts
                for t in range(T):
                    p = str(paths.recon(ds_name, subject, method, var, t))
                    nib.save(nib.Nifti1Image(
                        np.ascontiguousarray(pred_vols[t].transpose(2, 1, 0).astype(np.float32)),
                        np.diag([INPLANE_MM, INPLANE_MM, man["dz_mm"], 1.0])), p)
                # Per-variant identity, written AFTER the phases so a crashed variant has no stamp
                # (unstamped reads as "cannot verify", never as "verified"). See paths.recon_stamp.
                json.dump({"ckpt": metadata["ckpt"],
                           "ckpt_fingerprint": metadata["ckpt_fingerprint"],
                           "git_commit": metadata["git_commit"]},
                          open(str(paths.recon_stamp(ds_name, subject, method, var)), "w"), indent=2)
                timing[var] = {"per_phase_ms": per_phase_ms, "total_sec": wall}
                rdiag[var] = resp_diag(ed_pack, breathing)
                if breathing:
                    np.savez_compressed(os.path.join(md, "ed_dvf.npz"),
                                        delta=ed_pack["delta"].astype(np.float16),
                                        slot_z=ed_pack["slot_z"], slot_t=ed_pack["slot_t"],
                                        applied_disp_mm=ed_pack["applied"])
                    # The realized draw makes the recon replayable even if the sampler changes.
                    metadata_draw = {"seq_index_used": int(seq),
                                     "seq_index_basis": f"sha256('{ds_name}/{subject}')",
                                     "slot_z": ed_pack["slot_z"].tolist(),
                                     "slot_t": ed_pack["slot_t"].tolist()}
                print(f"  [{subject} {var}] T={T} D={bundle.shape[1]} wall={wall:.1f}s "
                      f"{np.mean(per_phase_ms):.0f}ms/phase", flush=True)

        json.dump(timing, open(os.path.join(md, "timing.json"), "w"), indent=2)
        json.dump(rdiag, open(os.path.join(md, "resp_diag.json"), "w"), indent=2)
        json.dump({**metadata, "draw": metadata_draw},
                  open(os.path.join(md, "metadata.json"), "w"), indent=2)
        with open(os.path.join(md, "provenance.txt"), "w") as f:
            f.write(f"method: {method}\ncommand: {' '.join(sys.argv)}\nckpt: {args.ckpt}\n")
            f.write(f"exp_name: {cfg.get('exp_name')}\nwandb_id: {_wandb_id(args.ckpt)}\n")
            f.write(f"git_commit: {_git_commit()}\ngpu: {torch.cuda.get_device_name(0)}\n")
            f.write(f"torch: {torch.__version__}  cuda: {torch.version.cuda}\n")
            f.write(f"img_size: {cfg.get('img_size')}  one_frame_per_slice: "
                    f"{cfg.get('one_frame_per_slice')}  continuous_z: {cfg.get('continuous_z')}\n")
            f.write("breathing_source: FROZEN (bundle breath/ pixels; NOT re-sampled)\n")
            f.write("geometry: native-z (per-subject D/dz, no 12mm snap)\n")
            f.write(f"model_load_sec: {model_load_s:.1f}\n")

    print(f"DONE -> {root}/*/{method}/", flush=True)


if __name__ == "__main__":
    main()
