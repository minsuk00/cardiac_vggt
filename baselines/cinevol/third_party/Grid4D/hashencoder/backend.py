from distutils.command.build import build
import os
from torch.utils.cpp_extension import load
from pathlib import Path

# VGGT-MRI patch (see baselines/cinevol/DEVIATIONS.md): build dir is configurable so array shards
# load one prebuilt .so instead of each compiling into its own cwd. Upstream: './tmp_build/'.
_build_dir = os.environ.get('GRID4D_BUILD_DIR', './tmp_build/')
Path(_build_dir).mkdir(parents=True, exist_ok=True)

_src_path = os.path.dirname(os.path.abspath(__file__))

_backend = load(name='_hash_encoder',
                extra_cflags=['-O3', '-std=c++17'],          # patch: upstream c++14; torch>=2.1 headers need c++17
                extra_cuda_cflags=[
                    '-O3', '-std=c++17', '-allow-unsupported-compiler',
                    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
                ],
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'hashencoder.cu',
                    'bindings.cpp',
                ]],
                build_directory=_build_dir,
                verbose=True,
                )

__all__ = ['_backend']