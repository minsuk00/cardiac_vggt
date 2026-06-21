# Source this to run nnU-Net v1 M&Ms (Task114) inference in the isolated `nnunet` env.
# Never activate svr alongside this. All paths on GPFS.
export NNUNET_ROOT=/home/minsukc/vggt/scratch/data/nnunet_mnms
export RESULTS_FOLDER=$NNUNET_ROOT/results
export nnUNet_raw_data_base=$NNUNET_ROOT/raw
export nnUNet_preprocessed=$NNUNET_ROOT/preprocessed
mkdir -p "$nnUNet_raw_data_base" "$nnUNet_preprocessed"
# usage: micromamba run -n nnunet bash -c 'source tools/nnunet_mnms_eval/env.sh && nnUNet_predict ...'
