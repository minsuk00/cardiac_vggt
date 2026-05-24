# Cardiac 4D MRI Slice-to-Volume Reconstruction

Adapts [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025) for **unsupervised slice-to-volume reconstruction of cardiac cine MRI**. Given S=12 scattered 2D slices at arbitrary `(cardiac phase t, z-position)` pairs, the model reconstructs the full 3D volume at any chosen target phase. Trained on CMRxRecon2024 (`Cine_combined`, 301 subjects).

## Setup

```bash
micromamba activate svr
pip install -e .
pip install -r requirements.txt
```

## Training

Entry point: `training/launch.py` (Hydra). Active config: `mri_volume`.

```bash
PYTHONPATH=training:. torchrun --nproc_per_node=1 training/launch.py --config mri_volume
```

Cluster: `bash sbatch/train_mri_volume.sh`.

## Acknowledgements

Built on top of [VGGT](https://github.com/facebookresearch/vggt) (Wang et al., CVPR 2025). Thanks to its authors for open-sourcing the model and pretrained weights, which we adapted for the cardiac MRI setting.

## License

See [LICENSE.txt](./LICENSE.txt).
