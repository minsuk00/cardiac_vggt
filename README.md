# Cardiac 4D MRI Slice-to-Volume Reconstruction

Adapts [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025) for **unsupervised slice-to-volume reconstruction of cardiac cine MRI**. Given S=12 scattered 2D slices at arbitrary `(cardiac phase t, z-position)` pairs, the model reconstructs the full 3D volume at any chosen target phase. Trained on CMRxRecon2024 (`Cine_combined`, 301 subjects).

## Setup

```bash
micromamba activate svr
pip install -r requirements.txt
```

This repo is not installed as a package. Run everything **from the repo root** with
`PYTHONPATH=training:.` — both entries matter. Python puts the *script's* directory on the
import path, never your working directory, so `.` is what makes `import vggt` resolve when you
run `training/launch.py`; `training` is what lets the Hydra configs resolve their short
`_target_` names such as `loss.MultitaskLoss`.

## Training

Entry point: `training/launch.py` (Hydra). Active config: `mri_volume`.

```bash
PYTHONPATH=training:. torchrun --nproc_per_node=1 training/launch.py --config mri_volume
```

Cluster: `bash sbatch/train_mri_volume_reference.sh`.

## Acknowledgements

Built on top of [VGGT](https://github.com/facebookresearch/vggt) (Wang et al., CVPR 2025). Thanks to its authors for open-sourcing the model and pretrained weights, which we adapted for the cardiac MRI setting.

## License

See [LICENSE.txt](./LICENSE.txt).
