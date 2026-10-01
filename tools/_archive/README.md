# Archived tools (non-functional)

Archived 2026-09-30, when `VGGT(...)`, `compute_volume_intensity_loss(...)` and `MultitaskLoss(...)`
stopped accepting unknown kwargs. Every script here either passed a removed argument
(`enable_camera`, `use_z_pose_embedding`, `depth`, `grid_shape`, `tv_weight`, ...) or imports one
that did. Before that change those arguments were silently ignored, so these scripts built the
current model, not the variant they were written for.

Kept for provenance only (docs cite some of them). Paths mirror their old place under `tools/`.
To see one as it last ran, `git log --follow -- tools/_archive/<path>`.
