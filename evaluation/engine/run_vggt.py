#!/usr/bin/env python
"""VGGT model eval on the FROZEN breathing bundles — the GPU head-to-head vs SVRTK / NeSVoR.

The GPU analog of engine/run_svrtk3d.sh / run_nesvor.sh: loads a VGGT-MRI checkpoint ONCE and
loops subjects, writing per-subject recons into `<dataset>/out/<subject>/<method>/` so the SAME
engine/assemble_and_gif.py scorer + aggregate.py roll-up consume it identically to the classical
baselines (model recon is canonical [0,1] -> `prep_recon` scores it AS-IS, like SVRTK).

WHY a new runner (not inference/run_cmrxrecon.py): those re-APPLY breathing via
gpu_augment_batch(rcfg, seq_index), which re-samples the trainer's POSITIONAL seq_index — a
DIFFERENT realization than the eval harness's name-hash breath. That is unfair vs the frozen
baselines. Here we consume the FROZEN breathing directly (the on-disk `breath/stack_t*.nii.gz`
pixels — byte-identical to what SVRTK saw), never re-sampling. GT stays the unshifted `gt/`
bundle. So model + baselines provably share ONE corruption + ONE target + ONE ROI + ONE scorer.

Regime (docs, memory `1frame_vs_multiframe_eval_regime`): a 1-frame model (e.g. gather05) must be
fed 1 frame/plane with the reference plane = the swept slot 0 ONLY; piling the multiframe reference
burst on it averages under the splat coverage-mean -> the "frozen" artifact. `--regime multiframe`
is for future s20-style models.

MIITT z placement: `--continuous-z` keeps each 10 mm native slice at its true fractional canonical
z (no 12 mm snap); default snaps (matches gather05's integer-z training). CMRx is genuinely 12 mm
so the flag is a no-op there.

Records everything (README §6b) for offline re-analysis with NO re-run: recon vols (EF/Dice-ready),
resp_diag.json (predicted Δz vs applied disp), timing.json (feed-forward wall vs SVR's minutes),
provenance.txt, metadata.json (model card), ed_dvf.npz (ED Δ field, the VGGT analog of SVR's .dof).

Run:
  EVAL_DATASET=cmrxrecon micromamba run -n svr env PYTHONPATH=training:. python \
      evaluation/engine/run_vggt.py --dataset cmrxrecon --model-name gather05 --regime onef
  EVAL_DATASET=miitt micromamba run -n svr env PYTHONPATH=training:. python \
      evaluation/engine/run_vggt.py --dataset miitt --model-name gather05 --regime onef [--continuous-z]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

VGGT = "/home/minsukc/vggt"
sys.path.insert(0, VGGT)
sys.path.insert(0, os.path.join(VGGT, "training"))
sys.path.insert(0, os.path.join(VGGT, "evaluation"))

import paths                                                                    # noqa: E402
from inference.inference import load_rtfb_model_reference                       # noqa: E402
from inference.adapters.base import (                                           # noqa: E402
    GRID_SHAPE, INPUT_IMG_SIZE, D_CANON, MM_PER_NORM,
    assign_canonical_z, to_canonical_inplane,
)
from inference.adapters.miitt import MIITTGatedAdapter                          # noqa: E402
from inference.adapters.ocmr import OCMRAdapter                                 # noqa: E402
from inference.adapters.acdc import ACDCGatedAdapter                            # noqa: E402
from vggt.utils.splat import splat_predictions                                  # noqa: E402

CANON_SPACING = (1.4, 1.4, 12.0)
ED_PHASE = 0
DEFAULT_CKPT = glob.glob(f"{VGGT}/scratch/logs/216539845_*ftgather05*1frame*/ckpts/checkpoint_last.pt")


def _canon_affine():
    return np.diag([*CANON_SPACING, 1.0])


def _load_xyz_to_dhw(path):
    """(X,Y,Z) NIfTI -> (D,H,W)=(Z,Y,X) splat order (inverse of build_inputs.save_xyz)."""
    a = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    return np.transpose(a, (2, 1, 0))


def _save_dhw_to_xyz(dhw, path):
    """(D,H,W) splat order -> (X,Y,Z) canonical NIfTI (matches build_inputs.save_xyz)."""
    nib.save(nib.Nifti1Image(np.ascontiguousarray(np.asarray(dhw, np.float32).transpose(2, 1, 0)),
                             _canon_affine()), path)


# ── per-subject input prep: one representation for both datasets + both z-modes ──────────────
def prep_cmrxrecon(subj_dir, T, continuous_z):
    """CMRx: gt/ and breath/ are ALREADY canonical -> read direct. z snapped (genuine 12 mm)."""
    gt = np.stack([_load_xyz_to_dhw(os.path.join(subj_dir, "gt", f"gt_t{t:02d}.nii.gz")) for t in range(T)])
    breath = np.stack([_load_xyz_to_dhw(os.path.join(subj_dir, "breath", f"stack_t{t:02d}.nii.gz")) for t in range(T)])
    content = _load_xyz_to_dhw(os.path.join(subj_dir, "mask.nii.gz")) > 0.5   # (D,H,W) native FOV
    planes = [int(z) for z in np.where(content.any(axis=(1, 2)))[0]]           # in-data canonical planes
    entries = [{"z_val": z / max(1, D_CANON - 1) * 2.0 - 1.0, "z_plane": z, "slice_idx": z} for z in planes]

    def fetch(phase, slice_idx, breathing):
        return (breath if breathing else gt)[phase, slice_idx]                 # (256,256) in [0,1]

    disp = np.asarray(json.load(open(os.path.join(subj_dir, "manifest.json")))["breath"]["disp_dhw_mm"],
                      dtype=np.float64)                                         # (D,3) per canonical plane
    return gt, entries, fetch, (lambda slice_idx: disp[slice_idx])


def prep_miitt(subj_dir, subject, T, continuous_z):
    """MIITT: gt/ is canonical; clean/ + breath/ are NATIVE (Z,H,W) -> place to canonical per z-mode."""
    gt = np.stack([_load_xyz_to_dhw(os.path.join(subj_dir, "gt", f"gt_t{t:02d}.nii.gz")) for t in range(T)])
    clean_nat = np.stack([_load_xyz_to_dhw(os.path.join(subj_dir, "clean", f"stack_t{t:02d}.nii.gz")) for t in range(T)])
    breath_nat = np.stack([_load_xyz_to_dhw(os.path.join(subj_dir, "breath", f"stack_t{t:02d}.nii.gz")) for t in range(T)])
    adapter = MIITTGatedAdapter(os.path.join(VGGT, "scratch/data/MIITT/nifti", subject, "gated/sax/4d_recon.nii.gz"))
    inpl = adapter.inplane_mm()
    z_map = assign_canonical_z(adapter.slice_positions_mm(), continuous_z=continuous_z)  # [(z_canon,slice_idx)]
    entries = [{"z_val": float(zc) / max(1, D_CANON - 1) * 2.0 - 1.0,
                "z_plane": min(max(int(round(float(zc))), 0), D_CANON - 1), "slice_idx": si}
               for zc, si in z_map]

    def fetch(phase, slice_idx, breathing):
        nat = (breath_nat if breathing else clean_nat)[phase, slice_idx]       # (H,W) native, already [0,1]
        return to_canonical_inplane(nat, inpl).numpy()                          # (256,256)

    disp = np.asarray(json.load(open(os.path.join(subj_dir, "manifest.json")))["breath"]["disp_dhw_mm"],
                      dtype=np.float64)                                         # (Z_native,3) per native slice
    return gt, entries, fetch, (lambda slice_idx: disp[slice_idx])


def _prep_gated_native(subj_dir, adapter, T, continuous_z):
    """Shared gated-OOD prep (OCMR/ACDC): gt/ canonical; clean/+breath/ NATIVE -> placed to
    canonical per z-mode via the adapter's geometry — identical machinery to prep_miitt, only the
    adapter (hence in-plane spacing + slice positions) differs."""
    gt = np.stack([_load_xyz_to_dhw(os.path.join(subj_dir, "gt", f"gt_t{t:02d}.nii.gz")) for t in range(T)])
    clean_nat = np.stack([_load_xyz_to_dhw(os.path.join(subj_dir, "clean", f"stack_t{t:02d}.nii.gz")) for t in range(T)])
    breath_nat = np.stack([_load_xyz_to_dhw(os.path.join(subj_dir, "breath", f"stack_t{t:02d}.nii.gz")) for t in range(T)])
    inpl = adapter.inplane_mm()
    z_map = assign_canonical_z(adapter.slice_positions_mm(), continuous_z=continuous_z)
    entries = [{"z_val": float(zc) / max(1, D_CANON - 1) * 2.0 - 1.0,
                "z_plane": min(max(int(round(float(zc))), 0), D_CANON - 1), "slice_idx": si}
               for zc, si in z_map]

    def fetch(phase, slice_idx, breathing):
        nat = (breath_nat if breathing else clean_nat)[phase, slice_idx]       # (H,W) native, already [0,1]
        return to_canonical_inplane(nat, inpl).numpy()                          # (256,256)

    disp = np.asarray(json.load(open(os.path.join(subj_dir, "manifest.json")))["breath"]["disp_dhw_mm"],
                      dtype=np.float64)                                         # (Z_native,3) per native slice
    return gt, entries, fetch, (lambda slice_idx: disp[slice_idx])


def prep_ocmr(subj_dir, subject, T, continuous_z):
    """OCMR gated: native source dir stored in manifest.native_source -> OCMRAdapter."""
    src = json.load(open(os.path.join(subj_dir, "manifest.json")))["native_source"]
    return _prep_gated_native(subj_dir, OCMRAdapter(src), T, continuous_z)


def prep_acdc(subj_dir, subject, T, continuous_z):
    """ACDC gated: native 4d nii path stored in manifest.native_source -> ACDCGatedAdapter (LPS)."""
    src = json.load(open(os.path.join(subj_dir, "manifest.json")))["native_source"]
    return _prep_gated_native(subj_dir, ACDCGatedAdapter(src), T, continuous_z)


# ── batch assembly + reference sweep ────────────────────────────────────────────────────────
def build_slots(n_entries, ref_k, T, regime, frames_per_slice, seq_index):
    """Slot list [(entry_k, phase)]. Slot 0 = swept reference (phase overwritten per step).
    onef: 1 frame/plane, no reference companions. multiframe: reference plane contributes all T
    phases as companions + a `frames_per_slice` consecutive burst per other plane (mirrors
    inference/run_cmrxrecon.py:_build_multiframe_batch). Burst START is random (seeded by
    seq_index) so ED isn't trivially observed everywhere."""
    rng = np.random.default_rng(seq_index)
    slots = [(ref_k, 0)]
    if regime == "multiframe":
        slots += [(ref_k, t) for t in range(T)]
    n = 1 if regime == "onef" else min(frames_per_slice, T)
    for k in range(n_entries):
        if k == ref_k:
            continue
        s0 = int(rng.integers(T))
        slots += [(k, (s0 + j) % T) for j in range(n)]
    return slots


def assemble_batch(slots, entries, fetch, breathing, device):
    hw = INPUT_IMG_SIZE
    py, px = np.meshgrid(np.arange(hw), np.arange(hw), indexing="ij")
    x_norm = (px / (hw - 1) * 2.0 - 1.0).astype(np.float32)
    y_norm = (py / (hw - 1) * 2.0 - 1.0).astype(np.float32)
    imgs, coords, zidx = [], [], []
    for k, phase in slots:
        e = entries[k]
        img = fetch(phase, e["slice_idx"], breathing)                          # (256,256) [0,1]
        up = F.interpolate(torch.as_tensor(img)[None, None].float(), size=(hw, hw),
                           mode="bilinear", align_corners=True)[0, 0].numpy()
        imgs.append(np.repeat(up[None], 3, axis=0))
        zv = e["z_val"]
        coords.append(np.stack([x_norm, y_norm, np.full_like(x_norm, zv)], -1))
        zidx.append([zv])
    return {
        "images": torch.from_numpy(np.stack(imgs)).float()[None].to(device),           # (1,S,3,hw,hw)
        "scanner_coords": torch.from_numpy(np.stack(coords)).float()[None].to(device),  # (1,S,hw,hw,3)
        "z_indices": torch.tensor(zidx, dtype=torch.float32)[None].to(device),          # (1,S,1)
    }


@torch.no_grad()
def reconstruct(model, prep, breathing, regime, frames_per_slice, seq_index, device, grid_shape):
    """Sweep the reference slot over T phases -> per-phase V_canon. Companions fixed; only slot 0
    (the reference at the queried phase) changes. -> (pred_vols (T,D,H,W), per_phase_ms, ed_pack)."""
    gt, entries, fetch, applied_disp = prep
    T = gt.shape[0]
    # Reference plane = content-bbox center (matches the trainer + inference/run_cmrxrecon, which
    # anchor slot 0 at z_mid=(z0+z1)//2). NOT argmin|z_val| (canonical-cube center) — that diverges
    # by one 12mm plane for off-center FOVs (22/30 CMRx subjects), feeding an out-of-convention anchor.
    z_planes = [e["z_plane"] for e in entries]
    z_mid = (min(z_planes) + max(z_planes) + 1) // 2
    ref_k = int(np.argmin([abs(e["z_plane"] - z_mid) for e in entries]))
    ref_slice = entries[ref_k]["slice_idx"]
    slots = build_slots(len(entries), ref_k, T, regime, frames_per_slice, seq_index)
    batch = assemble_batch(slots, entries, fetch, breathing, device)
    hw = batch["images"].shape[-1]

    pred_vols, per_phase_ms, ed_pack = [], [], None
    for t in range(T):
        ref = fetch(t, ref_slice, breathing)                                   # reference at queried phase
        up = F.interpolate(torch.as_tensor(ref)[None, None].float(), size=(hw, hw),
                           mode="bilinear", align_corners=True).to(device)
        batch["images"][:, 0] = up.repeat(1, 3, 1, 1)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
        wp = preds["world_points"].float()
        V_ref = preds.get("V_refined")                                         # refiner ckpts emit V_refined/V_canon
        V_canon = preds.get("V_canon")
        if V_canon is None:                                                    # no-refiner model: splat here
            V_canon, _ = splat_predictions({"world_points": wp}, batch, grid_shape)
        V_out = V_ref if V_ref is not None else V_canon                        # prefer refined (mirrors inference.forward)
        torch.cuda.synchronize(); per_phase_ms.append((time.perf_counter() - t0) * 1e3)
        pred_vols.append(V_out[0].float().cpu().numpy())
        if t == ED_PHASE:
            delta = (wp[0] - batch["scanner_coords"][0].float()).cpu().numpy()  # (S,hw,hw,3) normalized
            ed_pack = dict(delta=delta, slots=slots, ref_k=ref_k, entries=entries,
                           images=batch["images"][0].mean(1).cpu().numpy(),
                           # clean run = negative control: nothing was applied, so applied disp is 0
                           applied=(np.stack([applied_disp(entries[k]["slice_idx"]) for k, _ in slots])
                                    if breathing else np.zeros((len(slots), 3), np.float64)))
    return np.stack(pred_vols), per_phase_ms, ed_pack


def resp_diag(ed_pack, breathing):
    """Predicted through-plane Δz (mm) vs applied disp d_D per slot at ED: slope/corr/EPE, a faithful
    analog of the trainer's metric_resp_slope_dz (loss.py) — includes ALL slots (slot 0, the reference
    anchor, INCLUDED, matched to the trainer) and uses the same 0.05 FOV gate. For the breath run
    applied = the manifest disp; for the clean run applied = 0 (a negative control: predicted Δz on
    un-breathed input, should be ~0)."""
    if ed_pack is None:
        return {}
    delta = ed_pack["delta"]; imgs = ed_pack["images"]                          # (S,hw,hw,3),(S,hw,hw)
    pred_dz, appl_dz = [], []
    for s in range(delta.shape[0]):                                             # all slots incl. slot 0 (reference)
        m = imgs[s] > 0.05                                                      # FOV gate (matches trainer)
        if not m.any():
            continue
        pred_dz.append(float(delta[s, ..., 2][m].mean() * MM_PER_NORM[2]))      # channel 2 = z(D); mm
        appl_dz.append(float(ed_pack["applied"][s, 0]))                         # applied d_D (through-plane) mm
    pred_dz, appl_dz = np.asarray(pred_dz), np.asarray(appl_dz)
    out = {"breathing": bool(breathing), "n_slots": int(pred_dz.size),
           "pred_dz_mm": pred_dz.tolist(), "applied_dz_mm": appl_dz.tolist(),
           "epe_dz_mm": float(np.mean(np.abs(pred_dz - appl_dz))) if pred_dz.size else None}
    if pred_dz.size >= 2 and appl_dz.std() > 1e-6:
        out["slope"] = float(np.polyfit(appl_dz, pred_dz, 1)[0])
        out["corr"] = float(np.corrcoef(appl_dz, pred_dz)[0, 1])
    return out


# ── provenance / metadata ────────────────────────────────────────────────────────────────────
def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", VGGT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _wandb_id(ckpt):
    runs = glob.glob(os.path.join(os.path.dirname(ckpt), "..", "wandb", "wandb", "run-*"))
    return runs[0].split("-")[-1] if runs else "unknown"


def stage_ckpt_to_tmp(ckpt):
    """Copy the GPFS ckpt to node-local /tmp + strip to weights-only, for fast repeated loads.
    Direct torch.load from GPFS is ~50x slower (storage-by-storage small reads that GPFS handles
    terribly; see global CLAUDE.md) — ~266s vs ~2-5s from /tmp. The ORIGINAL file is never touched.
    Idempotent: reuses the /tmp weights-only file if present (so multiple runs on a node stage once).
    Returns the /tmp weights-only path to load from."""
    import torch
    tag = hashlib.md5(os.path.abspath(ckpt).encode()).hexdigest()[:8]
    wo = f"/tmp/vggt_ckpt_{tag}_weightsonly.pt"
    if os.path.exists(wo):
        print(f"[stage] reusing {wo} ({os.path.getsize(wo)/1e9:.2f} GB)", flush=True)
        return wo
    tmp_full = f"/tmp/vggt_ckpt_{tag}_full.pt"
    t0 = time.perf_counter()
    shutil.copyfile(ckpt, tmp_full)                      # one GPFS *sequential* read (fast-ish)
    ck = torch.load(tmp_full, map_location="cpu", weights_only=False)  # from /tmp: fast, no small-read penalty
    torch.save({"model": ck["model"]}, wo)               # weights-only (~half the bytes)
    del ck; os.remove(tmp_full)                          # drop the full copy; keep only weights-only
    print(f"[stage] {ckpt}\n        -> {wo} (weights-only, {os.path.getsize(wo)/1e9:.2f} GB) "
          f"in {time.perf_counter()-t0:.0f}s; original untouched", flush=True)
    return wo


def _ckpt_fingerprint(ckpt):
    """Cheap identity for the checkpoint without hashing GBs off GPFS: (size:mtime)."""
    try:
        st = os.stat(ckpt)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return None


def _run_identity(args):
    """The fields that make two runs the SAME model/config. A mismatch under one arm name means a
    different run would clobber the existing recons."""
    return {"ckpt": args.ckpt, "ckpt_fingerprint": _ckpt_fingerprint(args.ckpt),
            "regime": args.regime, "z_mode": "continuous" if args.continuous_z else "snapped"}


def _same_ckpt(prev, ident):
    """Same checkpoint? Prefer the content fingerprint (size:mtime) when BOTH sides have it — that
    catches a same-PATH file retrained in place (fingerprint differs) and ignores abs-vs-rel path
    spelling. Legacy metadata has no fingerprint -> fall back to realpath(path) so a pre-guard resume
    with the real ckpt does NOT false-conflict."""
    pf, cf = prev.get("ckpt_fingerprint"), ident.get("ckpt_fingerprint")
    if pf and cf:
        return pf == cf
    return os.path.realpath(prev.get("ckpt") or "") == os.path.realpath(ident.get("ckpt") or "")


def check_overwrite(ds, subjects, method, ident, overwrite):
    """Refuse to write into an arm whose existing recons came from a DIFFERENT run (ckpt content/
    regime/z_mode), unless --overwrite. Same identity = a legit resume/re-run; arms with no
    metadata.json yet (fresh, or classical baselines) never conflict. A corrupt/unreadable
    metadata.json is treated as no-conflict (warn, don't crash) so a killed prior run can't wedge the
    guard. Fail fast BEFORE the ~1 min model load."""
    conflicts = []
    for subject in subjects:
        mpath = str(paths.metadata(ds, subject, method))
        if not os.path.exists(mpath):
            continue
        try:
            with open(mpath) as fh:
                prev = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[run_vggt] WARNING: unreadable {mpath} ({e}); treating as fresh", flush=True)
            continue
        if not isinstance(prev, dict):   # valid JSON but not an object (null/list/scalar) -> don't .get-crash
            print(f"[run_vggt] WARNING: {mpath} is not a JSON object; treating as fresh", flush=True)
            continue
        if (prev.get("regime") != ident["regime"] or prev.get("z_mode") != ident["z_mode"]
                or not _same_ckpt(prev, ident)):
            conflicts.append((subject, {k: prev.get(k) for k in ("ckpt", "regime", "z_mode")}))
    if conflicts and not overwrite:
        detail = "\n".join(f"    {s}: {p}" for s, p in conflicts[:5])
        more = "" if len(conflicts) <= 5 else f"\n    ... +{len(conflicts) - 5} more"
        sys.exit(
            f"REFUSING to overwrite arm '{method}': {len(conflicts)} existing subject(s) came from a "
            f"DIFFERENT run.\n  this run: {ident}\n  on disk:\n{detail}{more}\n"
            f"  -> pass --overwrite to replace, or use a different --model-name.")
    if conflicts:
        print(f"[run_vggt] --overwrite: replacing {len(conflicts)} subject(s) of arm '{method}' "
              f"from a different prior run", flush=True)


def build_metadata(args, ckpt, method):
    exp = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
    return {
        "method": method, "model_name": args.model_name, "date": args.date,
        "ckpt": ckpt, "ckpt_fingerprint": _ckpt_fingerprint(ckpt), "exp_name": exp, "wandb_id": _wandb_id(ckpt),
        "config": args.config, "regime": args.regime, "frames_per_slice": args.frames_per_slice,
        "z_mode": "continuous" if args.continuous_z else "snapped",
        "breathing_source": "frozen (eval bundle breath/ + manifest disp)",
        "git_commit": _git_commit(), "note": args.note,
    }


def write_provenance(path, args, ckpt, method, model_load_s):
    with open(path, "w") as f:
        f.write(f"method: {method}\ncommand: {' '.join(sys.argv)}\nckpt: {ckpt}\n")
        f.write(f"exp_name: {os.path.basename(os.path.dirname(os.path.dirname(ckpt)))}\n")
        f.write(f"wandb_id: {_wandb_id(ckpt)}\ngit_commit: {_git_commit()}\n")
        f.write(f"gpu: {torch.cuda.get_device_name(0)}\n")
        f.write(f"torch: {torch.__version__}  cuda: {torch.version.cuda}\n")
        f.write(f"regime: {args.regime}  frames_per_slice: {args.frames_per_slice}  "
                f"z_mode: {'continuous' if args.continuous_z else 'snapped'}\n")
        f.write("breathing_source: FROZEN (eval bundle breath/ pixels + manifest disp; NOT re-sampled)\n")
        f.write(f"model_load_sec: {model_load_s:.1f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(paths.DATASETS))
    ap.add_argument("--ckpt", default=(DEFAULT_CKPT[0] if DEFAULT_CKPT else None))
    ap.add_argument("--model-name", required=True,
                    help="arm slug; the output dir is vggt_<model-name>. REQUIRED (no default) so a "
                         "stray run can't silently overwrite a named arm like gather05.")
    ap.add_argument("--date", default=None, help="legacy: include date in the arm name "
                    "(vggt_<date>_<model>); omit for the slug form vggt_<model> (scheme/date -> MODELS.md)")
    ap.add_argument("--regime", choices=["onef", "multiframe"], default="onef")
    ap.add_argument("--frames-per-slice", type=int, default=5, help="multiframe only")
    ap.add_argument("--continuous-z", action="store_true",
                    help="OOD (miitt/ocmr/acdc): fractional physical z, no 12mm snap; no-op on cmrxrecon")
    ap.add_argument("--refiner", action="store_true")
    ap.add_argument("--stage-tmp", action="store_true",
                    help="copy ckpt to node-local /tmp + strip to weights-only for fast loads "
                         "(GPFS small-read fix, ~266s->~5s); original untouched, staged once per node")
    ap.add_argument("--config", default="mri_volume_diffusion", help="for the model card")
    ap.add_argument("--note", default="")
    ap.add_argument("--subjects", nargs="*", default=None, help="default: all built subjects")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing arm produced by a DIFFERENT run (ckpt/regime/"
                         "z_mode); without it, run_vggt refuses to clobber a named arm.")
    args = ap.parse_args()
    assert args.ckpt, "no gather05 ckpt found; pass --ckpt"

    method = paths.canonical_arm(args.model_name, date=args.date, continuous_z=args.continuous_z)
    # Slug in name, scheme in registry: --date omitted -> vggt_<model_name>; passing --date keeps the
    # legacy vggt_<date>_<model> form. canonical_arm de-doubles _contz (fixes the historical bug), so a
    # NEW contz OOD run is named vggt_..._contz (single), unlike the existing doubled dirs (left as-is).
    ds = args.dataset
    root = paths.dataset_root(ds)
    subjects = args.subjects or paths.subjects(ds)
    if not subjects:
        sys.exit(f"no built subjects under {root}/*/manifest.json")

    check_overwrite(ds, subjects, method, _run_identity(args), args.overwrite)  # fail fast before model load

    load_ckpt = stage_ckpt_to_tmp(args.ckpt) if args.stage_tmp else args.ckpt  # metadata still uses args.ckpt
    device = torch.device("cuda")
    t0 = time.perf_counter()
    model = load_rtfb_model_reference(load_ckpt, refiner=args.refiner, device=device)
    model_load_s = time.perf_counter() - t0
    print(f"[run_vggt] method={method}  regime={args.regime}  z={'contz' if args.continuous_z else 'snap'}  "
          f"subjects={len(subjects)}  (model load {model_load_s:.0f}s)", flush=True)
    metadata = build_metadata(args, args.ckpt, method)

    for si, subject in enumerate(subjects):
        subj_dir = str(paths.subject_dir(ds, subject))
        T = json.load(open(paths.manifest(ds, subject)))["T"]
        prep_by_ds = {"cmrxrecon": lambda: prep_cmrxrecon(subj_dir, T, args.continuous_z),
                      "miitt": lambda: prep_miitt(subj_dir, subject, T, args.continuous_z),
                      "ocmr": lambda: prep_ocmr(subj_dir, subject, T, args.continuous_z),
                      "acdc": lambda: prep_acdc(subj_dir, subject, T, args.continuous_z)}
        prep_fn = prep_by_ds[args.dataset]()
        if not prep_fn[1]:                                                      # no in-FOV planes -> skip, don't crash
            print(f"  [{subject}] SKIP: no in-FOV planes", flush=True); continue
        md = str(paths.arm_dir(ds, subject, method)); os.makedirs(md, exist_ok=True)

        timing, rdiag = {"model_load_sec": model_load_s}, {}
        for breathing, var in [(False, "clean"), (True, "breath")]:
            rv = str(paths.recon_dir(ds, subject, method, var)); os.makedirs(rv, exist_ok=True)
            ts = time.perf_counter()
            pred_vols, per_phase_ms, ed_pack = reconstruct(
                model, prep_fn, breathing, args.regime, args.frames_per_slice, si, device, GRID_SHAPE)
            wall = time.perf_counter() - ts
            for t in range(T):
                _save_dhw_to_xyz(pred_vols[t], str(paths.recon(ds, subject, method, var, t)))
            timing[var] = {"per_phase_ms": per_phase_ms, "total_sec": wall}
            rdiag[var] = resp_diag(ed_pack, breathing)
            if breathing:                                                       # ED Δ field: the SVR .dof analog
                np.savez_compressed(os.path.join(md, "ed_dvf.npz"),
                                    delta=ed_pack["delta"].astype(np.float16),
                                    slot_z=np.array([ed_pack["entries"][k]["z_plane"] for k, _ in ed_pack["slots"]]),
                                    slot_t=np.array([ph for _, ph in ed_pack["slots"]]),
                                    applied_disp_mm=ed_pack["applied"])
            print(f"  [{subject} {var}] T={T} wall={wall:.1f}s  {np.mean(per_phase_ms):.0f}ms/phase", flush=True)

        json.dump(timing, open(os.path.join(md, "timing.json"), "w"), indent=2)
        json.dump(rdiag, open(os.path.join(md, "resp_diag.json"), "w"), indent=2)
        json.dump(metadata, open(os.path.join(md, "metadata.json"), "w"), indent=2)
        write_provenance(os.path.join(md, "provenance.txt"), args, args.ckpt, method, model_load_s)

    print(f"DONE -> {root}/*/{method}/  (score: assemble_and_gif.py <subj> {method})", flush=True)


if __name__ == "__main__":
    main()
