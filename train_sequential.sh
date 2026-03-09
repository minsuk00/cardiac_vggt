#!/bin/bash

# Sequential Training Script for MRI Reconstruction
# 1. Static Oblique
# 2. Dynamic Oblique
# 3. Dynamic Axial
# 4. Dynamic Mixed

RUNS=(
    "static oblique"
    "dynamic oblique"
    "dynamic axial"
    "dynamic mixed"
)

for RUN in "${RUNS[@]}"
do
    read -r DATA_MODE MRI_MODE <<< "$RUN"
    
    echo "=========================================================="
    echo "STARTING PHASE: DATA_MODE=$DATA_MODE, MRI_MODE=$MRI_MODE"
    echo "=========================================================="
    
    # Run the training with Hydra overrides
    PYTHONUNBUFFERED=1 PYTHONPATH=training:. torchrun \
        --nproc_per_node=1 \
        --master_port=29507 \
        training/launch.py \
        --config mri_finetune \
        mri_data_mode="$DATA_MODE" \
        mri_mode="$MRI_MODE"

    echo "PHASE $DATA_MODE/$MRI_MODE COMPLETED."
    echo "----------------------------------------------------------"
    sleep 5
done

echo "ALL SEQUENTIAL TRAINING PHASES FINISHED."
