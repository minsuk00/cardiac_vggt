#!/bin/bash
# EF pipeline for 4wokxzov (217720691): dump V_canon+V_gt per (subj,target_t) -> nnU-Net Task114 seg
# -> pred-vs-true EF slope + ES-timing. Confirms whether 4wok recovers per-patient contraction amplitude
# or (like the reference ckpt, slope -0.026) regresses to the cohort mean.
set -e
cd /home/minsukc/vggt
CK=scratch/logs/217720691_mri_volume_diffusion_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt
VOLS=scratch/analysis/phase_analysis/4wok_vols
SEG=scratch/analysis/phase_analysis/4wok_segs
mkdir -p "$VOLS" "$SEG"

echo "[1/3] dump volumes (svr)"
micromamba run -n svr python tools/cmrxrecon_phase_analysis/measure_model_contraction.py \
    --out_dir "$VOLS" --ckpt "$CK" --n_subjects 30 --warp_head_type dpt

echo "[2/3] nnU-Net Task114 seg (nnunet, 2d)"
micromamba run -n nnunet bash -c "source tools/nnunet_mnms_eval/env.sh && \
    nnUNet_predict -i '$VOLS' -o '$SEG' -t 114 -m 2d -f 0 --disable_tta"

echo "[3/3] analyze EF"
micromamba run -n svr python tools/cmrxrecon_phase_analysis/analyze_model_contraction.py \
    --seg_dir "$SEG" --out_json "$VOLS/ef_4wok.json"
echo "EF_4WOK_DONE"
