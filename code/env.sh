export WORK_DIR=$SCRATCH/ants-amp-paper
export TEMPLATEFLOW_DIR=${PROJECT_HOME}/templateflow
export SIF_DIR=${PROJECT_HOME}/containers
export DATA_DIR=${PROJECT_HOME}/datasets/corr/RawDataBIDS/BMB_1
export DATALAD_URL="https://datasets.datalad.org/corr/RawDataBIDS/BMB_1"
export SUBJECTS=($(awk 'NR>1 {print $1}' ${DATA_DIR}/participants.tsv))
export SLURM_OPTS="--account=rrg-glatard"

# SIF Images
export ANTS_BASE_SIF=${SIF_DIR}/ants-baseline.sif
export ANTS_AMP_SIF=${SIF_DIR}/ants-amp-miccai2026.sif
