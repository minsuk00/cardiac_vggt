#!/bin/bash
# Cross-architecture equivalence check: 2023/2025 were reconstructed on A40 (cc8.6),
# 2024 will land on L40S (cc8.9). Re-reconstruct one ALREADY-DONE 2025 subject here and
# compare against its A40 output. Expect ~135 dB (the documented same-arch float32
# reduction-order noise floor); anything materially worse means arch is confounded with year.
# Short walltime so it backfills ahead of the 4 h array.
#SBATCH --job-name=arch_equiv
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/%j_arch_equiv.log
set -eo pipefail
cd /home/minsukc/vggt
eval "$(micromamba shell hook --shell bash)"
micromamba activate svr
nvidia-smi --query-gpu=name,compute_cap --format=csv
CID=CMRx25_R1test_Center003_UIH_15T_umr670_P001
python tools/reconstruct_cmrx2025.py --subjects $CID --force \
    --out-root /tmp/archcheck --report /tmp/archcheck_report.json --stage-dir /tmp/archcheck_stage
python - <<PY
import nibabel as nib, numpy as np
a=np.asarray(nib.load("/tmp/archcheck/$CID/sax/4d_recon.nii.gz").dataobj).astype(np.float64)   # L40S
b=np.asarray(nib.load("scratch/data/CMRxRecon2025/Cine_combined/$CID/sax/4d_recon.nii.gz").dataobj).astype(np.float64)  # A40
assert a.shape==b.shape, (a.shape,b.shape)
mse=((a-b)**2).mean(); rng=b.max()-b.min()
print(f"shape {a.shape}")
print(f"max|diff| {np.abs(a-b).max():.3e}   rel {np.abs(a-b).max()/rng:.2e}")
print(f"corr {np.corrcoef(a.ravel(),b.ravel())[0,1]:.12f}")
print(f"PSNR {10*np.log10(rng**2/mse) if mse>0 else float('inf'):.2f} dB   -> {'PASS (arch irrelevant)' if mse==0 or 10*np.log10(rng**2/mse)>120 else 'INVESTIGATE'}")
PY
