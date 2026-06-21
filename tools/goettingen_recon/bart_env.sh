#!/bin/bash
# Source this to put our locally-built BART (CUDA) on PATH with its FFTW/OpenBLAS deps.
# Built against: gcc/13.2.0 + cuda/12.8.1 modules + conda env `bart-build` (fftw, openblas).
module load gcc/13.2.0 cuda/12.8.1 2>/dev/null

export BART_TOOLBOX_PATH=/home/minsukc/vggt/scratch/bart
export PATH=$BART_TOOLBOX_PATH:$PATH
export LD_LIBRARY_PATH=/home/minsukc/micromamba/envs/bart-build/lib:$LD_LIBRARY_PATH
# nlinv-net scripts were written against BART v0.9.00; we built v1.0.00.
export BART_COMPAT_VERSION="v0.9.00"
