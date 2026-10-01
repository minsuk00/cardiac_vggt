# Cardiac 4D MRI Slice-to-Volume Reconstruction

Adapts [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025) for **unsupervised slice-to-volume reconstruction of cardiac cine MRI**. The input is one 2D slice per short-axis plane, each from an arbitrary cardiac phase, plus one reference slice at the target phase. The model reconstructs the full 3D volume at that target phase. Trained on the pooled curated-v2 split (`training/splits/pooled_curated_v2.txt`: 628 / 90 / 180 train / val / test subjects).

## Setup

```bash
micromamba activate svr
pip install -r requirements.txt
pip install --no-deps -e .     # the vggt, training and inference packages (editable)
```

`batchaug` and `fused_ssim` are not on PyPI; see the comments at the bottom of `requirements.txt`.
`--no-deps` keeps pip from re-resolving the pinned torch stack that `requirements.txt` installed.

## Training

Entry point: `training.launch` (Hydra, single GPU). Config: `training/config/default.yaml` = the paper recipe (the `diff1000` arm); ablation arms are `training/config/ablation_<arm>.yaml`.

```bash
torchrun --nproc_per_node=1 -m training.launch --config default
```

Cluster (paper recipe, self-submitting, auto-requeue): `ARM=diff1000 bash sbatch/train_final_518.sh`.
`ARM` is one of `base | diff1000 | hw2 | hw0 | nogather | diff1000_nogather`; `diff1000` is the paper model.

## Evaluation

`sbatch sbatch/eval_pooled_val.sh` scores a checkpoint on the frozen gated + breathing-simulated bundles. See `evaluation/README.md`.

## Acknowledgements

Built on top of [VGGT](https://github.com/facebookresearch/vggt) (Wang et al., CVPR 2025). Thanks to its authors for open-sourcing the model and pretrained weights, which we adapted for the cardiac MRI setting.

## License

See [LICENSE.txt](./LICENSE.txt).
