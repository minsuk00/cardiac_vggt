#!/bin/bash
# Run a fiss_recon python script in the `fiss-recon` env with cupy's CUDA
# headers + libs on the path (cupy 13 needs nvrtc/cufft from the pip nvidia
# wheels). Usage: bash tools/fiss_recon/run.sh cs_recon.py <args...>
ENV=/home/minsukc/micromamba/envs/fiss-recon
export CUDA_PATH=$ENV/cuda_headers
export LD_LIBRARY_PATH=$ENV/cuda_headers/lib:$LD_LIBRARY_PATH
cd "$(dirname "$0")"
exec micromamba run -n fiss-recon python "$@"
