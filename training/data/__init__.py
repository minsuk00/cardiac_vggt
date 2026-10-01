from training.resolvers import register_all

from .datasets.mri_dataset import MRIDataset

# Registered here too (imported by every consumer that instantiates datasets/models from the
# config) so `patch_size: ${backbone_ps:${backbone}}` resolves in standalone compose()
# scripts, not just training/launch.py.
register_all()
