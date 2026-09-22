# source baselines/cinevol/env.sh        (from the repo root; runs CiNeVol under the svr env, nothing installed)
#
# The Grid4D hash encoder is a torch JIT CUDA extension: the first import compiles ONE .so into
# $GRID4D_BUILD_DIR (~80 s), later imports just load it. Compile it ONCE before an array job
# (python tools/run_cinevol.py --build-only) so shards never race on the build directory.
source /etc/profile.d/modules.sh 2>/dev/null
module load gcc/11.2.0 cuda/13.1.0                       # nvcc matching svr's torch 2.13+cu130
export TORCH_CUDA_ARCH_LIST="8.6;8.9"                    # A40 (spgpu) + L40S (spgpu2) in one .so
export GRID4D_BUILD_DIR="${GRID4D_BUILD_DIR:-$PWD/scratch/cinevol/grid4d_build}"
export CINEVOL_PY="${CINEVOL_PY:-/home/minsukc/micromamba/envs/svr/bin/python}"
export PYTHONPATH="$PWD/baselines/cinevol:$PWD/training:$PWD"
export PYTHONDONTWRITEBYTECODE=1
