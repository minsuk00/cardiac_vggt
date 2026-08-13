"""CMRx24 adapter and Stage-1 train/evaluation operations."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from cardiac import (
    augment_reference_inplane,
    augment_slice_intensity,
    compensate_motion,
    extract_respiratory_slices_256,
    motion_metrics,
    native_to_stage1,
    released_coarse_motion_metrics,
    reconstruct_native,
    remove_stage1_padding,
    stage1_flow_to_native_mm,
)


RESPIRATORY_DEFAULTS = dict(
    enable=True,
    amplitude_mm=18.8,
    amplitude_jitter=7.35,
    cos2n=3,
    ap_ratio=0.35,
    ap_axis="H",
    group_by_burst=True,
    tilt_min_deg=0.0,
    tilt_max_deg=45.0,
    amplitude_breath_jitter=0.0,
)


def make_dataset(data_root: str, split_file: str, split: str):
    """Construct the read-only VGGT CMR cache adapter in fixed-ED/static mode."""
    from data import MRIDataset

    return MRIDataset(
        common_conf={}, data_root=data_root, split=split, split_file=split_file,
        mode="static", num_slices=20, t_target_fixed=0, reference_slot=False,
        continuous_z=False, one_frame_per_slice=True, defer_input_images=True,
    )


def prepare_sample(dataset, index: int, device: torch.device, *, seed: int):
    """Load native ED, draw one current-simulator displacement per z-plane, corrupt it."""
    from data.respiratory import RespiratoryConfig, sample_resp_disp

    item = dataset.get_data(seq_index=index, img_per_seq=20)
    dz_mm = float(item["dz_mm"][0])
    if abs(dz_mm - 12.0) > 1e-4:
        raise ValueError(
            f"FC-SVR CMR24 protocol requires dz=12 mm, got {dz_mm}; "
            "variable-pitch slab scaling is not implemented yet"
        )
    phases = torch.as_tensor(item["phases"], device=device).float().unsqueeze(0)
    mask = torch.as_tensor(item["content_mask"], device=device).float()
    if "heart_roi_canonical" not in item:
        raise RuntimeError(f"{item['seq_name']} has no shape-valid heart_roi_canonical")
    heart_mask = torch.as_tensor(item["heart_roi_canonical"], device=device).float() * mask
    if not heart_mask.any():
        raise RuntimeError(f"{item['seq_name']} has an empty heart_roi_canonical")
    generator = torch.Generator(device=device).manual_seed(seed)
    is_train = getattr(dataset, "split", None) == "train"
    if is_train:
        phases_aug, mask, heart_mask = augment_reference_inplane(
            phases[0], mask, heart_mask, generator
        )
        phases = phases_aug.unsqueeze(0)
    clean = phases[0, 0]
    depth = clean.shape[0]
    z = torch.arange(depth, device=device).view(1, depth)
    t = torch.zeros_like(z)
    cfg = RespiratoryConfig(**RESPIRATORY_DEFAULTS)
    disp, _ = sample_resp_disp(
        # Use the simulator's train branch with our private seeded generator for
        # both splits. This gives the same distribution and deterministic val
        # without coupling to MRIDataset's seq_index wrapper.
        1, depth, cfg, device, train=True, generator=generator,
        group_ids=z, n_planes=depth,
    )
    corrupt = extract_respiratory_slices_256(
        phases, t, z, disp, spacing_mm=(dz_mm, 1.4, 1.4)
    )[0]
    corrupt_heart_mask = extract_respiratory_slices_256(
        heart_mask[None, None], t, z, disp,
        spacing_mm=(dz_mm, 1.4, 1.4),
    )[0].gt(0.5).float()
    if is_train:
        corrupt = augment_slice_intensity(corrupt, generator)
    inputs, target, meta = native_to_stage1(
        corrupt * corrupt_heart_mask,
        corrupt_heart_mask,
        disp[0],
        corrupt_heart_mask,
    )
    return {
        "inputs": inputs.unsqueeze(0), "target": target.unsqueeze(0), "meta": meta,
        "corrupt": corrupt.unsqueeze(0), "clean": clean.unsqueeze(0),
        "corrupt_mask": corrupt_heart_mask.unsqueeze(0),
        "heart_mask": heart_mask.unsqueeze(0),
        "gt_motion_dhw_mm": disp,
        "bbox": torch.as_tensor(item["anatomy_bbox"], device=device),
        "spacing": (dz_mm, 1.4, 1.4),
        "name": item["seq_name"],
    }


def predict_native_motion(model, inputs, target, meta, *, compensate: bool):
    coarse = model(inputs.float())
    return coarse_to_native_motion(model, coarse, target, meta, compensate=compensate)


def coarse_to_native_motion(model, coarse, target, meta, *, compensate: bool):
    if compensate:
        coarse = compensate_motion(coarse, target)
    coarse = remove_stage1_padding(coarse, meta)
    high = model.upsample_flow(coarse)
    return stage1_flow_to_native_mm(high, meta)


def volume_metrics(
    pred: torch.Tensor, target: torch.Tensor, bbox: torch.Tensor,
    heart_mask: torch.Tensor | None = None,
):
    """Match VGGT's unit-range full and geometric-bbox MAE/MSE/PSNR definitions."""
    error = pred - target
    mse = error.square().mean()
    z0, z1, y0, y1, x0, x1 = [int(x) for x in bbox]
    cropped = error[:, z0:z1, y0:y1, x0:x1]
    mse_bbox = cropped.square().mean()
    metrics = {
        "metric_mae_3d_full": error.abs().mean().item(),
        "metric_mse_3d_full": mse.item(),
        "metric_psnr_3d_full": (10 * torch.log10(mse.clamp_min(1e-10).reciprocal())).item(),
        "metric_mae_3d_bbox": cropped.abs().mean().item(),
        "metric_mse_3d_bbox": mse_bbox.item(),
        "metric_psnr_3d_bbox": (10 * torch.log10(mse_bbox.clamp_min(1e-10).reciprocal())).item(),
    }
    if heart_mask is not None:
        selected = error.masked_select(heart_mask.bool())
        mse_heart = selected.square().mean()
        metrics.update({
            "metric_mae_3d_heartseg": selected.abs().mean().item(),
            "metric_mse_3d_heartseg": mse_heart.item(),
            "metric_psnr_3d_heartseg": (
                10 * torch.log10(mse_heart.clamp_min(1e-10).reciprocal())
            ).item(),
        })
    return metrics


@torch.no_grad()
def validate(model, dataset, device, *, seed: int, limit: int | None = None):
    """Evaluate raw and official paper-compensated Stage-1 reconstructions."""
    was_training = model.training
    model.eval()
    rows = []
    count = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(count):
        sample = prepare_sample(dataset, index, device, seed=seed + index)
        row = {"subject": sample["name"], "label": "FC-SVR Stage 1, GT-pose-normalized"}
        raw_coarse = model(sample["inputs"].float())
        for label, compensated in (("raw", False), ("compensated", True)):
            coarse = compensate_motion(raw_coarse, sample["target"]) if compensated else raw_coarse
            motion = coarse_to_native_motion(
                model, coarse, sample["target"], sample["meta"], compensate=False
            )
            recon, _ = reconstruct_native(
                sample["corrupt"], motion, spacing_mm=sample["spacing"],
                foreground_mask=sample["corrupt_mask"],
            )
            row[label] = volume_metrics(
                recon, sample["clean"], sample["bbox"], sample["heart_mask"]
            )
            row[label].update(released_coarse_motion_metrics(coarse, sample["target"]))
            row[label].update(motion_metrics(
                motion,
                sample["gt_motion_dhw_mm"],
                sample["corrupt_mask"],
                slab_spacing_mm=sample["meta"].slab_spacing_mm[0],
            ))
        rows.append(row)
    model.train(was_training)
    return rows


def append_jsonl(path: Path, record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
