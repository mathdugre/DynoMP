#!/bin/sh
set -eu

source ./code/setup.sh

echo "launching sbatch jobs..."
#########################
# ANTs Brain Extraction #
#########################
export SIF_IMG=${ANTS_BASE_SIF}
brain_extraction=$(sbatch --parsable --array=0-$(( ${#SUBJECTS[@]} - 1 )) ${SLURM_OPTS} ./code/antsBrainExtraction.sbatch)

#####################
# ANTs (Binary 64) #
####################
export SIF_IMG=${ANTS_BASE_SIF}
export EXPERIMENT_NAME="fp64"
sbatch --dependency=$brain_extraction \
    ${SLURM_OPTS} \
    --array=0-$(( ${#SUBJECTS[@]} - 1 )) \
    ./code/antsRegistration.sbatch

###############
# ANTs (DynoMP) #
##############
export SIF_IMG=${ANTS_DYNOMP_SIF}
export EXPERIMENT_NAME="dynomp"
sbatch --dependency=$brain_extraction \
    ${SLURM_OPTS} \
    --array=0-$(( ${#SUBJECTS[@]} - 1 )) \
    ./code/antsRegistration.sbatch

