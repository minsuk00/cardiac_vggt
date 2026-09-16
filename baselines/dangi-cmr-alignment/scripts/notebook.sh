#!/bin/bash
# Run from a shell with access to this repository (GPU allocation for local training).
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export JUPYTER_CONFIG_DIR="${JUPYTER_CONFIG_DIR:-$repo_dir/.jupyter/config}"
export JUPYTER_RUNTIME_DIR="${JUPYTER_RUNTIME_DIR:-$repo_dir/.jupyter/runtime}"
export IPYTHONDIR="${IPYTHONDIR:-$repo_dir/.jupyter/ipython}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$repo_dir/.jupyter/matplotlib}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python -m ipykernel install --sys-prefix --name dangi --display-name 'Python (Dangi)'
exec .venv/bin/jupyter lab --no-browser Dangi_Workbench.ipynb "$@"
