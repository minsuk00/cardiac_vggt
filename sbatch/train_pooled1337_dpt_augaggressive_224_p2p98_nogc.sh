#!/bin/bash
export EXPERIMENT_OVERRIDES="intensity_percentiles=[2,98] model.gradient_checkpointing=false"
export VARIANT_SUFFIX="_p2p98_nogc"
exec bash /home/minsukc/vggt/sbatch/train_pooled1337_dpt_augaggressive_224.sh
