#!/bin/bash
#SBATCH --job-name=e0_score_followup
#SBATCH --account=jjparkcv0
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/%x_%j.log
# Full 5-arm read at 30k+ (docs/70 §1d follow-up): heart arms at ckpt_150 (~35.3k),
# corseg arms at ckpt_last (~30k). NOTE: corseg arms TRAIN on CorSeg, so their
# CorSeg-derived amp_ratio is biased UP — a flat score is a strong negative verdict,
# a positive score needs nnU-Net re-scoring before being believed.

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate svr

cd /home/minsukc/vggt
run_arm () {  # name tree ckpt port
  local name=$1 tree=$2 ckpt=$3 port=$4
  local out=result/e0_dumps/$name
  python tools/e0_dump_phase_sweep.py \
    --tree "$tree" --config default \
    --ckpt "$ckpt" --out "$out" --limit-val-batches 29 --master-port $port \
    --override split_file=training/splits/cmrx24only.txt \
    --override dataset_name=cmrx24only \
    --override ef_val_sweep=false --override logging.ef_eval_enable=false || return 1
  python tools/e0_score_volumes.py --dump "$out" --gt-seg-dir result/e0_dumps/_gt_segs_cmrx24val
}
H=/home/minsukc/vggt-arm-heart
C=/home/minsukc/vggt-arm-corseg
run_arm arm_heartl1_w000_35k $H scratch/logs/213530039_mri_volume_heartl1_w000_dynamic_axial_cmrx24only/ckpts/checkpoint_150.pt 29693
run_arm arm_heartl1_w010_35k $H scratch/logs/213520194_mri_volume_heartl1_w010_dynamic_axial_cmrx24only/ckpts/checkpoint_150.pt 29694
run_arm arm_heartl1_w050_35k $H scratch/logs/213520194_mri_volume_heartl1_w050_dynamic_axial_cmrx24only/ckpts/checkpoint_150.pt 29695
run_arm arm_corseg_w002_30k $C scratch/logs/213515736_mri_volume_corsegdice_w002_dynamic_axial_cmrx24only/ckpts/checkpoint_last.pt 29696
run_arm arm_corseg_w100_30k $C scratch/logs/213515736_mri_volume_corsegdice_w100_dynamic_axial_cmrx24only/ckpts/checkpoint_last.pt 29697
echo ALL_DONE
