# Archived tools (non-functional)

Archived 2026-09-30, when `VGGT(...)`, `compute_volume_intensity_loss(...)` and `MultitaskLoss(...)`
stopped accepting unknown kwargs. Every script here either passed a removed argument
(`enable_camera`, `use_z_pose_embedding`, `depth`, `grid_shape`, `tv_weight`, ...) or imports one
that did. Before that change those arguments were silently ignored, so these scripts built the
current model, not the variant they were written for.

Archived 2026-10-01: 7 more that used the removed dataloader stack (`get_loader`,
`.base_dataset.datasets[0]`, `dataset_configs[0]`), removed aggregator kwargs or the old
`GradientClipper(model)` call, or import one that did.

Kept for provenance only (docs cite some of them). Paths mirror their old place under `tools/`.
To see one as it last ran, `git log --follow -- tools/_archive/<path>`.
