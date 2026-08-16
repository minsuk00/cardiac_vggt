# Model provenance (arms under `evaluation/volumes/`)

One row per arm dir name (same name in `volumes/<ds>/out/<subj>/<arm>` and, when a
durable copy exists, `checkpoints/<arm>/`). VGGT arms are harvested from their
`metadata.json`; classical baselines have none. Regenerate with
`python evaluation/build_models_table.py`.

Config deltas (gather05/contz/lowdiff100/…) are NOT in a resolved yaml — they live in
the named base config (`training/config/<config>.yaml`) + the training sbatch
(`sbatch/_archive/oneframe_*.sh`). `ckpt (source)` is the original path; `copied`=✓ means a
durable copy sits in `checkpoints/<arm>/checkpoint.pt`.

| arm | type | ep | config | regime | fps | z_mode | date | wandb | commit | copied | datasets | ckpt (source) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nesvor | baseline | — | — | — | — | — | — | — | — |  | cmrxrecon miitt | — | — |
| svrtk3d | baseline | — | — | — | — | — | — | — | — |  | cmrxrecon miitt | — | — |
| vggt_20260713_gather05 | vggt | — | mri_volume_diffusion | onef | 5 | snapped | 20260713 | 81li618p | 92fd739 |  | cmrxrecon miitt | `/home/minsukc/vggt/scratch/logs/216539845_mri_volume_diffusion_ftgather05_1frame_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt` |  |
| vggt_20260713_gather05_contz | vggt | — | mri_volume_diffusion | onef | 5 | continuous | 20260713 | 81li618p | 92fd739 |  | miitt | `/home/minsukc/vggt/scratch/logs/216539845_mri_volume_diffusion_ftgather05_1frame_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt` |  |
| vggt_20260715_1f_aug_moderate | vggt | 39 | mri_volume_diffusion | onef | 5 | snapped | 20260715 | unknown | 4a98b3a |  | cmrxrecon miitt | `/home/minsukc/vggt/scratch/checkpoints/20260715_1frame_aug_moderate_ep39.pt` | 1frame ablation ckpt ep39 |
| vggt_20260715_1f_contz | vggt | 39 | mri_volume_diffusion | onef | 5 | snapped | 20260715 | unknown | 4a98b3a |  | cmrxrecon | `/home/minsukc/vggt/scratch/checkpoints/20260715_1frame_contz_ep39.pt` | 1frame ablation ckpt ep39 |
| vggt_20260715_1f_contz_contz | vggt | 39 | mri_volume_diffusion | onef | 5 | continuous | 20260715 | unknown | 4a98b3a |  | miitt | `/home/minsukc/vggt/scratch/checkpoints/20260715_1frame_contz_ep39.pt` | 1frame ablation ckpt ep39 |
| vggt_20260715_1f_dino_ft | vggt | 33 | mri_volume_diffusion | onef | 5 | snapped | 20260715 | unknown | 4a98b3a |  | cmrxrecon miitt | `/home/minsukc/vggt/scratch/checkpoints/20260715_1frame_dino_ft_ep33.pt` | 1frame ablation ckpt ep33 |
| vggt_20260715_1f_gather05 | vggt | 39 | mri_volume_diffusion | onef | 5 | snapped | 20260715 | unknown | 4a98b3a |  | cmrxrecon miitt | `/home/minsukc/vggt/scratch/checkpoints/20260715_1frame_gather05_ep39.pt` | 1frame ablation ckpt ep39 |
| vggt_20260715_1f_lowdiff100 | vggt | 25 | mri_volume_diffusion | onef | 5 | snapped | 20260715 | unknown | 4a98b3a |  | cmrxrecon miitt | `/home/minsukc/vggt/scratch/checkpoints/20260715_1frame_lowdiff100_ep25.pt` | 1frame ablation ckpt ep25 |
| vggt_20260715_1f_no_gather | vggt | 37 | mri_volume_diffusion | onef | 5 | snapped | 20260715 | unknown | 4a98b3a |  | cmrxrecon miitt | `/home/minsukc/vggt/scratch/checkpoints/20260715_1frame_no_gather_ep37.pt` | 1frame ablation ckpt ep37 |
| vggt_20260716_1f_aug_moderate_ep59 | vggt | 59 | mri_volume_diffusion | onef | 5 | snapped | 20260716 | unknown | 4a98b3a |  | cmrxrecon miitt | `/home/minsukc/vggt/scratch/checkpoints/20260716_1frame_aug_moderate_ep59.pt` | 1frame ablation RESUMED ckpt ep59 |
| vggt_20260716_1f_contz_ep59 | vggt | 59 | mri_volume_diffusion | onef | 5 | snapped | 20260716 | unknown | 4a98b3a |  | cmrxrecon | `/home/minsukc/vggt/scratch/checkpoints/20260716_1frame_contz_ep59.pt` | 1frame ablation RESUMED ckpt ep59 |
| vggt_20260716_1f_contz_ep59_contz | vggt | 59 | mri_volume_diffusion | onef | 5 | continuous | 20260716 | unknown | 4a98b3a |  | miitt | `/home/minsukc/vggt/scratch/checkpoints/20260716_1frame_contz_ep59.pt` | 1frame ablation RESUMED ckpt ep59 |
| vggt_20260716_1f_dino_ft_ep50 | vggt | 50 | mri_volume_diffusion | onef | 5 | snapped | 20260716 | unknown | 4a98b3a |  | cmrxrecon miitt | `/home/minsukc/vggt/scratch/checkpoints/20260716_1frame_dino_ft_ep50.pt` | 1frame ablation RESUMED ckpt ep50 |
| vggt_20260716_1f_gather05_ep60 | vggt | 60 | mri_volume_diffusion | onef | 5 | snapped | 20260716 | unknown | 4a98b3a |  | cmrxrecon miitt | `/home/minsukc/vggt/scratch/checkpoints/20260716_1frame_gather05_ep60.pt` | 1frame ablation RESUMED ckpt ep60 |
| vggt_20260716_1f_lowdiff100_ep44 | vggt | 44 | mri_volume_diffusion | onef | 5 | snapped | 20260716 | unknown | 4a98b3a |  | cmrxrecon miitt | `/home/minsukc/vggt/scratch/checkpoints/20260716_1frame_lowdiff100_ep44.pt` | 1frame ablation RESUMED ckpt ep44 |
| vggt_20260716_1f_no_gather_ep57 | vggt | 57 | mri_volume_diffusion | onef | 5 | snapped | 20260716 | unknown | 4a98b3a |  | cmrxrecon miitt | `/home/minsukc/vggt/scratch/checkpoints/20260716_1frame_no_gather_ep57.pt` | 1frame ablation RESUMED ckpt ep57 |
| vggt_20260719_1f_aug_moderate_ep99 | vggt | 99 | mri_volume_diffusion | onef | 5 | snapped | 20260719 | unknown | 4a98b3a | ✓ | acdc cmrxrecon miitt ocmr | `/home/minsukc/vggt/scratch/checkpoints/20260719_1frame_aug_moderate_ep99.pt` | 1frame ablation FINAL ckpt ep99 |
| vggt_20260719_1f_contz_ep99 | vggt | 99 | mri_volume_diffusion | onef | 5 | snapped | 20260719 | unknown | 4a98b3a | ✓ | cmrxrecon | `/home/minsukc/vggt/scratch/checkpoints/20260719_1frame_contz_ep99.pt` | 1frame ablation FINAL ckpt ep99 |
| vggt_20260719_1f_contz_ep99_contz | vggt | 99 | mri_volume_diffusion | onef | 5 | continuous | 20260719 | unknown | 4a98b3a | ✓ | acdc miitt ocmr | `/home/minsukc/vggt/scratch/checkpoints/20260719_1frame_contz_ep99.pt` | 1frame ablation FINAL ckpt ep99 |
| vggt_20260719_1f_dino_ft_ep99 | vggt | 99 | mri_volume_diffusion | onef | 5 | snapped | 20260719 | unknown | 4a98b3a | ✓ | acdc cmrxrecon miitt ocmr | `/home/minsukc/vggt/scratch/checkpoints/20260719_1frame_dino_ft_ep99.pt` | 1frame ablation FINAL ckpt ep99 |
| vggt_20260719_1f_gather05_ep99 | vggt | 99 | mri_volume_diffusion | onef | 5 | snapped | 20260719 | unknown | 4a98b3a | ✓ | acdc cmrxrecon miitt ocmr | `/home/minsukc/vggt/scratch/checkpoints/20260719_1frame_gather05_ep99.pt` | 1frame ablation FINAL ckpt ep99 |
| vggt_20260719_1f_lowdiff100_ep99 | vggt | 99 | mri_volume_diffusion | onef | 5 | snapped | 20260719 | unknown | 4a98b3a | ✓ | acdc cmrxrecon miitt ocmr | `/home/minsukc/vggt/scratch/checkpoints/20260719_1frame_lowdiff100_ep99.pt` | 1frame ablation FINAL ckpt ep99 |
| vggt_20260719_1f_no_gather_ep99 | vggt | 99 | mri_volume_diffusion | onef | 5 | snapped | 20260719 | unknown | 4a98b3a | ✓ | acdc cmrxrecon miitt ocmr | `/home/minsukc/vggt/scratch/checkpoints/20260719_1frame_no_gather_ep99.pt` | 1frame ablation FINAL ckpt ep99 |
