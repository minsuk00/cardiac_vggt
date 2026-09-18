"""Build the frozen input bundle for any pooled source: GT + clean + breathing-corrupted stacks.

One builder for all seven sources. It replaces the four per-dataset builders (`cmrxrecon.py`,
`acdc.py`, `miitt.py`, `ocmr.py` — the last three archived), which existed only because three
sources reached the model through hand-written adapters. Every source now has an `MRIDataset`
entry, so the bundle is just that subject's own phases plus one breathing realization.

    python evaluation/src/engine/build_inputs/pooled.py --source cmrx2024
    python evaluation/src/engine/build_inputs/pooled.py --source acdc --subjects ACDC_patient006
    python evaluation/src/engine/build_inputs/pooled.py --source ocmr --split-file <f> --split val

Output per subject, under `evaluation/volumes/<source>/out/<subject>/`:

    gt/gt_t{00..T-1}.nii.gz        clean, unshifted — what every method is scored against
    clean/stack_t{00..T-1}.nii.gz  == GT planes (the no-breathing upper bound)
    breath/stack_t{00..T-1}.nii.gz each plane z resliced by its OWN breath displacement
    mask.nii.gz                    content (FOV) mask
    mask_heart.nii.gz              heart ROI, canonical grid — copied when the source has one
    heart_seg.nii.gz               per-phase LV/MYO/RV, canonical grid — same
    manifest.json                  geometry + the full breathing realization
    scatter/stack_t{00..T-1}.nii.gz  same-input stack VGGT sees (add_scatter, --add-scatter)
    rolled/stack_t{00..T-1}.nii.gz   breath/ with each plane's phases circularly rolled by a frozen
                                     random offset: UNKNOWN per-slice phase, for the self-gating
                                     Fetal CMR 4D arm (add_rolled, --add-rolled; docs/105)

## Why the bundle is frozen

The classical SVR baselines (SVRTK, NeSVoR, NiftyMIC) take minutes to hours per subject, so they
run ONCE and their recons are cached. That only yields a fair head-to-head if every method — the
baselines and our model, today and in six months — is handed a **byte-identical** corruption.
Hence: write the corrupted stacks to disk once, and have every scorer read them.

## What is deliberately NOT in the bundle

The input **slot draw** (which z, which t per slot). The bundle holds whole volumes, so it is
draw-independent; the draw belongs to scoring time and is recorded by `run_vggt.py` per subject.
This is what makes adding a val subject cheap: bundles are keyed on the subject NAME, never on its
position in the split, so appending to the split file cannot invalidate an existing bundle.

## Native z (docs/58)

`D` and `dz` come from the subject, never from a constant. There is no 12-plane cube and no
snapping: `dz` is read per subject from `data["dz_mm"]`, the affine is stamped with it, and the
respiratory reslice is passed `spacing=(dz, 1.4, 1.4)`. `reslice_volume_vec` has no default
spacing any more, so a missed call site raises instead of silently breathing a 10 mm subject at
12 mm (the pitch-keyed mis-grading of docs/58 §8.1a).

## Breathing determinism

Seeded by `sha256("<source>/<subject>")`, not by the positional `seq_index` — so a subject's
breathing is identical no matter where it sits in the split file, or whether other subjects were
added before it.

`group_ids=arange(D)` and `n_planes=D` are BOTH passed, which the old builders never did
(`n_planes` is inert unless `group_ids` is also given). **Measured, because the mechanism is
misleading here: the old form produced bit-identical breathing anyway.** With `S == D == n_planes`
the burst branch's `gather(rand((B,P)), arange(D))` consumes the generator in exactly the same
order as the per-slot branch's `rand((B,S))`, and with `per_slot=False` the amplitude draw
`rand((B,1))` matches too — so the two agree exactly. The match rests on BOTH coincidences, and
breaks the moment either goes: leaving `per_slot` at its config default (True) diverges, and so
does `S != D`. Passing the grouping explicitly, and leaving `per_slot` alone, makes this bundle
match the trainer *structurally* instead of numerically-by-accident — the old builders' own
`rcfg.per_slot = False` line was in fact the one thing that would have diverged from training had
`group_by_burst` ever been turned off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import nibabel as nib
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, ROOT)

from data.datasets.mri_dataset import MRIDataset                               # noqa: E402
from data.respiratory import RespiratoryConfig, sample_resp_disp, reslice_volume_vec  # noqa: E402
import evaluation.paths as paths                                               # noqa: E402

DATA_ROOT = os.path.join(ROOT, "scratch/data")
DEFAULT_SPLIT_FILE = os.path.join(ROOT, "training/splits/pooled_curated_v2.txt")  # docs/97; miitt/ocmr use their *_eval.txt
DEFAULT_CFG = os.path.join(ROOT, "training/config/default.yaml")
INPLANE_MM = 1.4          # canonical in-plane spacing; must match preprocess.TARGET_SPACING

# source name -> the split-file path prefix that identifies it. The split lines are
# "<prefix>/.../<subject>", so the first path component is the source.
SOURCE_PREFIX = {
    "cmrx2023": "CMRxRecon2023",
    "cmrx2024": "CMRxRecon2024",
    "cmrx2025": "CMRxRecon2025",
    "acdc": "ACDC_sax",
    "mnms": "MNMs_sax",
    "miitt": "MIITT_sax",
    "ocmr": "OCMR_sax",
}

# Canonical-grid siblings copied verbatim into the bundle when the source ships them.
SIBLINGS = {"mask_heart.nii.gz": "heart_roi_canonical.nii.gz",
            "heart_seg.nii.gz": "heart_seg_canonical.nii.gz"}


def name_seed(source, name):
    """Stable, split-order-robust breath seed = hash of '<source>/<subject>'."""
    return int(hashlib.sha256(f"{source}/{name}".encode()).hexdigest(), 16) % (2 ** 31)


def load_respiratory_config(cfg_path):
    """RespiratoryConfig from a training config's `data.augmentation.respiratory` block.

    Plain `OmegaConf.load`, not Hydra `compose`: the block is all literals, so this needs no
    resolver registration and cannot be broken by an unrelated interpolation elsewhere in the
    config (the `backbone_tag`/`aug_tag` resolvers, for one, are defined only in launch.py).
    """
    cfg = OmegaConf.load(cfg_path)
    node = cfg.data.augmentation.respiratory
    rcfg = RespiratoryConfig.from_cfg(node)
    if not rcfg.enable:
        raise SystemExit(f"respiratory sim disabled in {cfg_path} — the breath arm would equal clean")
    return rcfg, OmegaConf.to_container(node, resolve=True)


def read_split(split_file, split, prefix):
    """Subject dir names (relative to data_root) in `split` whose first path component is `prefix`."""
    out, cur = [], None
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                cur = line[1:-1].lower()
            elif cur == split.lower() and line.split("/")[0] == prefix:
                out.append(line)
    return out


def draw_scatter(rel_path, split, seed, D, T):
    """VGGT's one-frame-per-slice slot draw for this subject -> (phase_per_plane (D,), ref_plane).

    Reproduces EXACTLY what run_vggt.py's make_dataset + get_data draw: a ONE-subject MRIDataset
    (so `seq_index` is only the RNG seed), reference_slot + one_frame_per_slice + integer z, seeded
    with the same name hash. The draw depends only on (seed, D, T) — every plane appears once, slot 0
    is the mid-ventricular reference at the target phase, every other plane gets an independent
    random phase. Frozen into manifest["scatter"] so every consumer (VGGT arms AND the classical
    baselines' scatter input) sees the same scattered acquisition."""
    import tempfile
    common = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True,
                               "rescale_aug": False, "landscape_check": False,
                               "augs": {"scales": [1.0, 1.0]}})
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "one_subject.txt")
        with open(sf, "w") as f:
            f.write(f"[{split}]\n{rel_path}\n")
        ds = MRIDataset(common, DATA_ROOT, split=split, split_file=sf, mode="dynamic",
                        mri_mode="axial", num_slices=D, target_size=518, t_target_fixed=0,
                        reference_slot=True, one_frame_per_slice=True, continuous_z=False)
        b = ds.get_data(seq_index=seed, img_per_seq=D)
    z = [int(round(float(v))) for v in b["slice_indices"]]
    t = [int(v) for v in b["timesteps"]]
    if sorted(z) != list(range(D)) or len(z) != D:
        raise RuntimeError(f"{rel_path}: scatter draw covers planes {sorted(z)}, expected 0..{D-1}")
    if any(v < 0 or v >= T for v in t):
        raise RuntimeError(f"{rel_path}: scatter draw phase out of range: {t}")
    phase_per_plane = [None] * D
    for zi, ti in zip(z, t):
        phase_per_plane[zi] = ti
    return phase_per_plane, z[0]


def add_scatter(subj_dir, source, name, split, rel_path):
    """Write <subj_dir>/scatter/stack_t{k}: the SAME-INPUT stack for the classical baselines —
    what VGGT actually sees when queried at phase k. Plane z = breath/stack_t{p_z}[z] with p_z
    the frozen random phase for that plane; the reference plane = breath/stack_t{k}[ref]. Pure
    re-assembly of the breath bundle (breathing is per plane and phase-independent), so it is
    deterministic and can be added to an existing bundle. Records the draw in manifest.json."""
    mpath = os.path.join(subj_dir, "manifest.json")
    man = json.load(open(mpath))
    T, D = man["T"], man["D"]
    seed = man["seed"]
    assert seed == name_seed(source, name), (seed, name_seed(source, name))
    phase_per_plane, ref = draw_scatter(rel_path, split, seed, D, T)
    breath = [nib.load(os.path.join(subj_dir, "breath", f"stack_t{t:02d}.nii.gz")) for t in range(T)]
    arrs = [np.asarray(im.dataobj, dtype=np.float32) for im in breath]     # (X,Y,Z) each
    affine = breath[0].affine
    os.makedirs(os.path.join(subj_dir, "scatter"), exist_ok=True)
    for k in range(T):
        out = np.empty_like(arrs[0])
        for z in range(D):
            out[:, :, z] = arrs[k if z == ref else phase_per_plane[z]][:, :, z]
        nib.save(nib.Nifti1Image(out, affine), os.path.join(subj_dir, "scatter", f"stack_t{k:02d}.nii.gz"))
    man["scatter"] = {"phase_per_plane": phase_per_plane, "ref_plane": int(ref),
                      "draw": "MRIDataset one_frame_per_slice+reference_slot, integer z, seq_index=seed",
                      "note": "scatter/stack_t{k}: plane z from breath/stack_t{phase_per_plane[z]}, "
                              "ref_plane from breath/stack_t{k}"}
    tmp = f"{mpath}.tmp{os.getpid()}"
    json.dump(man, open(tmp, "w"), indent=1)
    os.replace(tmp, mpath)
    return phase_per_plane, ref


def add_rolled(subj_dir, source, name):
    """Write <subj_dir>/rolled/stack_t{k}: the breath bundle with each plane's T phases circularly
    shifted by its own frozen random roll r_z, i.e. rolled/stack_t{k}[z] = breath/stack_t{(k - r_z) mod T}[z].
    Frame k of plane z therefore shows cardiac phase (k - r_z) mod T, and the stored ED (phase 0)
    sits at frame r_z — unknown to the consumer. This is the input for the self-gating Fetal CMR 4D
    arm (docs/105): the stored cine is ED-first and cross-slice synchronised, so on breath/ the
    self-gater would merely read the index back (measured: period forced to T, ED at frame 0 in
    112/167 slices); the roll removes the per-slice start phase, which is exactly the unknown the
    method's inter-slice synchronisation exists to recover. Breathing stays one state per plane
    (a 12-frame real-time burst is ~1 s, well inside a breathing cycle). Deterministic: the roll RNG
    is seeded from the bundle's own subject seed (+1 so it never shares a stream with the breathing
    draw). Records the roll in manifest["rolled"] — for auditing, never read by the arm."""
    mpath = os.path.join(subj_dir, "manifest.json")
    man = json.load(open(mpath))
    T, D = man["T"], man["D"]
    seed = man["seed"]
    assert seed == name_seed(source, name), (seed, name_seed(source, name))
    roll = np.random.default_rng(seed + 1).integers(0, T, size=D).tolist()
    breath = [nib.load(os.path.join(subj_dir, "breath", f"stack_t{t:02d}.nii.gz")) for t in range(T)]
    arrs = [np.asarray(im.dataobj, dtype=np.float32) for im in breath]     # (X,Y,Z) each
    affine = breath[0].affine
    os.makedirs(os.path.join(subj_dir, "rolled"), exist_ok=True)
    for k in range(T):
        out = np.empty_like(arrs[0])
        for z in range(D):
            out[:, :, z] = arrs[(k - roll[z]) % T][:, :, z]
        nib.save(nib.Nifti1Image(out, affine), os.path.join(subj_dir, "rolled", f"stack_t{k:02d}.nii.gz"))
    man["rolled"] = {"roll_per_plane": roll, "rng": "np.random.default_rng(seed + 1).integers(0, T, D)",
                     "note": "rolled/stack_t{k}: plane z from breath/stack_t{(k - roll_per_plane[z]) mod T}; "
                             "stored ED (phase 0) of plane z is at frame roll_per_plane[z]"}
    tmp = f"{mpath}.tmp{os.getpid()}"
    json.dump(man, open(tmp, "w"), indent=1)
    os.replace(tmp, mpath)
    return roll


def build_subject(rel_path, source, rcfg, rcfg_dump, out_root, split_file, split, overwrite):
    """-> (name, status) where status is 'built' | 'skipped'."""
    name = os.path.basename(rel_path)
    subj_dir = os.path.join(out_root, name)
    if os.path.exists(os.path.join(subj_dir, "manifest.json")) and not overwrite:
        return name, "skipped"

    # One dataset per subject: `--subjects` and incremental adds must not depend on how many
    # other subjects are in the split, and get_data reads everything it needs from the subject.
    # img_per_seq=1 / num_slices=1 because the bundle holds whole VOLUMES — the slot draw is
    # scoring-time state and is recorded by run_vggt.py, not here.
    common = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True,
                               "rescale_aug": False, "landscape_check": False,
                               "augs": {"scales": [1.0, 1.0]}})
    ds = MRIDataset(common, DATA_ROOT, split=split, split_file=split_file,
                    mode="dynamic", mri_mode="axial", num_slices=1, target_size=518,
                    t_target_fixed=0)
    idx = [i for i, p in enumerate(ds.subjects)
           if os.path.basename(os.path.dirname(p)) == name]
    if not idx:
        raise KeyError(f"{name} not found in {split_file} [{split}]")
    data = ds.get_data(seq_index=idx[0], img_per_seq=1)

    phases = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32))   # (T,D,H,W) splat order
    content_mask = np.asarray(data["content_mask"]).astype(np.uint8)           # (D,H,W)
    T, D, H, W = phases.shape
    dz = float(np.asarray(data["dz_mm"]).reshape(-1)[0])
    spacing_dhw = (dz, INPLANE_MM, INPLANE_MM)          # (D,H,W) mm, the respiratory code's order
    affine = np.diag([INPLANE_MM, INPLANE_MM, dz, 1.0])  # (X,Y,Z) on disk

    seed = name_seed(source, name)
    # group_ids AND n_planes: group_by_burst is a no-op without group_ids (respiratory.py), which
    # is exactly how every previous builder silently fell to the per-slot branch.
    disp, r = sample_resp_disp(
        1, D, rcfg, "cpu", train=False,
        seq_index=torch.tensor([[seed]], dtype=torch.int64),
        group_ids=torch.arange(D, dtype=torch.long).unsqueeze(0), n_planes=D)
    disp0, r0 = disp[0], r[0]                            # (D,3) mm ; (D,)

    for sub in ("gt", "clean", "breath"):
        os.makedirs(os.path.join(subj_dir, sub), exist_ok=True)

    def save_xyz(dhw, path):
        nib.save(nib.Nifti1Image(np.ascontiguousarray(np.asarray(dhw).transpose(2, 1, 0)), affine), path)

    save_xyz(content_mask.astype(np.float32), os.path.join(subj_dir, "mask.nii.gz"))

    # Canonical heart siblings: copied when present, WARNED and skipped when not. The old builder
    # hard-copied and would crash — and absence is real (docs/78 reports a missing ROI hard-raises
    # in training when heart_weight>0, so some sources genuinely lack it).
    sax = os.path.join(DATA_ROOT, rel_path, "sax")
    missing_siblings = []
    for dst, src in SIBLINGS.items():
        p = os.path.join(sax, src)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(subj_dir, dst))
        else:
            missing_siblings.append(src)

    for t in range(T):
        Vt = phases[t]                                   # (D,H,W)
        save_xyz(Vt, os.path.join(subj_dir, "gt", f"gt_t{t:02d}.nii.gz"))
        save_xyz(Vt, os.path.join(subj_dir, "clean", f"stack_t{t:02d}.nii.gz"))
        # Each plane z keeps only its OWN breath-displaced reslice — the acquisition model:
        # slice z was acquired during breath r[z], so that is the only plane it observes.
        breathed = torch.stack(
            [reslice_volume_vec(Vt, disp0[z], spacing=spacing_dhw)[z] for z in range(D)], dim=0)
        save_xyz(breathed, os.path.join(subj_dir, "breath", f"stack_t{t:02d}.nii.gz"))

    manifest = {
        "source": source, "subject": name, "rel_path": rel_path,
        "split_file": os.path.relpath(split_file, ROOT), "split": split,
        "seed": seed, "seed_basis": f"sha256('{source}/{name}')",
        "T": T, "D": D, "H": H, "W": W,
        "spacing_xyz_mm": [INPLANE_MM, INPLANE_MM, dz], "dz_mm": dz,
        "content_mask_frac": float(content_mask.mean()),
        "heart_siblings_missing": missing_siblings,
        "builder": os.path.basename(__file__),
        "breath": {
            "mean_abs_disp_mm": float(disp0.norm(dim=-1).mean()),
            "max_abs_disp_mm": float(disp0.norm(dim=-1).max()),
            "disp_dhw_mm": disp0.tolist(),               # (D,3) per z-plane
            "r_per_plane": r0.tolist(),                  # (D,)
            "config": rcfg_dump,                         # every parameter, for the scorer to check
        },
    }
    json.dump(manifest, open(os.path.join(subj_dir, "manifest.json"), "w"), indent=1)
    add_scatter(subj_dir, source, name, split, rel_path)
    return name, "built"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCE_PREFIX))
    ap.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    ap.add_argument("--split", default="val")
    ap.add_argument("--subjects", default="", help="comma-separated subject names (default: all "
                                                  "of --source in --split)")
    ap.add_argument("--config", default=DEFAULT_CFG,
                    help="training config supplying the respiratory block (frozen into the bundle)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild subjects that already have a manifest.json (default: skip them, "
                         "which is what makes adding val subjects an incremental no-op)")
    ap.add_argument("--add-scatter", action="store_true",
                    help="only add scatter/ (+ manifest['scatter']) to EXISTING bundles that lack it; "
                         "gt/clean/breath untouched")
    ap.add_argument("--add-rolled", action="store_true",
                    help="only add rolled/ (+ manifest['rolled']) to EXISTING bundles that lack it; "
                         "gt/clean/breath/scatter untouched (input for the self-gating fetal_cmr_4d arm)")
    a = ap.parse_args()

    rcfg, rcfg_dump = load_respiratory_config(a.config)
    rels = read_split(a.split_file, a.split, SOURCE_PREFIX[a.source])
    if a.subjects:
        want = {s.strip() for s in a.subjects.split(",") if s.strip()}
        rels = [r for r in rels if os.path.basename(r) in want]
        unknown = want - {os.path.basename(r) for r in rels}
        if unknown:
            raise SystemExit(f"not in {a.split_file} [{a.split}] for source {a.source}: {sorted(unknown)}")
    if a.limit:
        rels = rels[: a.limit]
    if not rels:
        raise SystemExit(f"no {a.source} subjects in {a.split_file} [{a.split}]")

    out_root = str(paths.dataset_root(a.source))
    os.makedirs(out_root, exist_ok=True)
    print(f"{a.source}: {len(rels)} subjects [{a.split}] -> {out_root}")
    if a.add_scatter:
        n = {"added": 0, "skipped": 0, "failed": 0}
        for rel in rels:
            name = os.path.basename(rel); sd = os.path.join(out_root, name)
            if not os.path.exists(os.path.join(sd, "manifest.json")):
                print(f"  no bundle for {name} — build it first", flush=True); n["failed"] += 1; continue
            if "scatter" in json.load(open(os.path.join(sd, "manifest.json"))):
                n["skipped"] += 1; continue
            try:
                add_scatter(sd, a.source, name, a.split, rel); n["added"] += 1
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {name}: {type(e).__name__}: {e}", flush=True); n["failed"] += 1
        print(f"add-scatter: {n}")
        return
    if a.add_rolled:
        n = {"added": 0, "skipped": 0, "failed": 0}
        for rel in rels:
            name = os.path.basename(rel); sd = os.path.join(out_root, name)
            if not os.path.exists(os.path.join(sd, "manifest.json")):
                print(f"  no bundle for {name} — build it first", flush=True); n["failed"] += 1; continue
            if "rolled" in json.load(open(os.path.join(sd, "manifest.json"))):
                n["skipped"] += 1; continue
            try:
                add_rolled(sd, a.source, name); n["added"] += 1
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {name}: {type(e).__name__}: {e}", flush=True); n["failed"] += 1
        print(f"add-rolled: {n}")
        return
    print(f"  breathing from {os.path.relpath(a.config, ROOT)}: amp={rcfg.amplitude_mm}"
          f"+/-{rcfg.amplitude_jitter}mm ap={rcfg.ap_ratio} burst={rcfg.group_by_burst} "
          f"tilt=({rcfg.tilt_min_deg},{rcfg.tilt_max_deg})", flush=True)

    n = {"built": 0, "skipped": 0, "failed": 0}
    warned = []
    for rel in rels:
        try:
            name, status = build_subject(rel, a.source, rcfg, rcfg_dump, out_root,
                                         a.split_file, a.split, a.overwrite)
        except Exception as e:  # noqa: BLE001 — one bad subject must not hide the rest
            print(f"  FAIL {os.path.basename(rel)}: {type(e).__name__}: {e}", flush=True)
            n["failed"] += 1
            continue
        n[status] += 1
        if status == "built":
            m = json.load(open(os.path.join(out_root, name, "manifest.json")))
            if m["heart_siblings_missing"]:
                warned.append(name)
            print(f"  {name:<44} T={m['T']} D={m['D']:>2} dz={m['dz_mm']:>5.2f} "
                  f"mean|disp|={m['breath']['mean_abs_disp_mm']:.2f}mm seed={m['seed']}"
                  + ("  [no heart ROI]" if m["heart_siblings_missing"] else ""), flush=True)
    print(f"built={n['built']} skipped={n['skipped']} failed={n['failed']}")
    if warned:
        print(f"WARNING: {len(warned)} subjects have no canonical heart ROI/seg — heart-ROI "
              f"metrics will be unavailable for them: {warned[:5]}")


if __name__ == "__main__":
    main()
