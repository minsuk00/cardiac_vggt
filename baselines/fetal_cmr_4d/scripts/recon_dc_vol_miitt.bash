#!/usr/bin/env bash


# RECON DC VOLUME

# e.g., bash recon_dc_vol.bash ~/path/to/top/level/recon/directory/ dc_vol


# Input 

RECONDIR=$1
VOLDESC=$2


# Check that Recon Directory Exists

if [ ! -d "$RECONDIR" ]; then
  echo directory $RECONDIR does not exist
  exit 1
else


# Manage Paths to Allow Queueing of Jobs using TaskSpooler

ORIG_PATH=$(pwd)
SCRIPT_PATH=$(dirname `which $0`)

RECONVOLDIR=$RECONDIR/$VOLDESC
mkdir -p $RECONVOLDIR
cd $RECONVOLDIR

echo RECON DC VOLUME
echo $RECONVOLDIR


# Variables 

RECON=$VOLDESC.nii.gz
STACKS="../data/s*_dc_ab.nii.gz"
THICKNESS=$(cat ../data/slice_thickness.txt)
MASKDCVOL="../mask/mask_chest.nii.gz"
TGTSTACKNO=$(cat ../data/tgt_stack_no.txt)
EXCLUDESTACKFILE="../data/force_exclude_stack.txt"
EXCLUDESLICEFILE="../data/force_exclude_slice.txt"
RESOLUTION=1.25
NMC=6
NSR=10
NSRLAST=20
NUMCARDPHASE=1
STACKDOFDIR="stack_transformations"
DOFOUTDIR="slice_transformations"

echo reconstructing DC volume: $RECONVOLDIR/$RECON


# Setup

ITER=$(($NMC+1))
NUMSTACK=$(ls -1 ../data/s*_dc_ab.nii.gz | wc -l);
EXCLUDESTACK=$(cat $EXCLUDESTACKFILE)
NUMEXCLUDESTACK=$(eval "wc -w $EXCLUDESTACKFILE | awk -F' ' '{print \$1}'" )
EXCLUDESLICE=$(cat $EXCLUDESLICEFILE)
NUMEXCLUDESLICE=$(eval "wc -w $EXCLUDESLICEFILE | awk -F' ' '{print \$1}'" )
echo "   target stack no.: "$TGTSTACKNO


# Reconstruct DC Volume

# MIITT single-stack patch: DROP -stack_registration (there is no second stack to
# register to; with one stack it hits a degenerate "average volume weight is 0"
# path in reconstructCardiac and segfaults). The slice->volume rigid registration
# (the actual static motion correction) still runs and still emits the slice dofs.
CMD="mirtk reconstructCardiac $RECON $NUMSTACK $STACKS -thickness $THICKNESS -mask $MASKDCVOL -iterations $ITER -rec_iterations $NSR -rec_iterations_last $NSRLAST -resolution $RESOLUTION -force_exclude_stack $NUMEXCLUDESTACK $EXCLUDESTACK -force_exclude_sliceloc $NUMEXCLUDESLICE $EXCLUDESLICE -numcardphase $NUMCARDPHASE -no_robust_statistics -debug > log-main.txt"
echo reconstructing DC volume: $CMD
echo $CMD > recon.bash
eval $CMD


# Clean Up

# MIITT single-stack patch: no -stack_registration means no stack-transformation
# dofs are produced, so synthesize an identity one (a single stack sits at the
# origin of its own world frame). slice_cine/cine_vol glob for this dof.
CMD="mkdir -p $STACKDOFDIR; mirtk init-dof $STACKDOFDIR/stack-transformation000.dof;"
echo $CMD >> recon.bash
eval $CMD

# MIITT patch: this SVRTK version names slice dofs transformation0.dof..transformationN.dof
# (single/double digit), so the authors' `transformation0*.dof` glob only catches #0.
# Use [0-9]* to capture all per-slice transforms.
CMD="mkdir -p $DOFOUTDIR; mv transformation[0-9]*.dof $DOFOUTDIR;"
echo $CMD >> recon.bash
eval $CMD

CMD="mkdir -p sr_iterations; mv *_mc*sr* sr_iterations;"
echo $CMD >> recon.bash
eval $CMD


# Finish

echo "volume reconstruction complete"

cd $ORIG_PATH


fi

