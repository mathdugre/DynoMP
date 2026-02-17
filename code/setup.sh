#!/bin/bash
set -eu

module load apptainer
module load git-annex

source ./code/env.sh

#####################
# DataLad downloads #
#####################
datalad install -gr -J4 \
    --source ${DATALAD_URL} \
    ${DATA_DIR}
datalad install -J4 -r \
    --source ///templateflow \
    ${TEMPLATEFLOW_DIR}
datalad get -d ${TEMPLATEFLOW_DIR} \
    ${TEMPLATEFLOW_DIR}/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz

#####################
# Apptainer images #
#####################
if [[ ! -f ${ANTS_BASE_SIF} ]]; then
    echo "Pulling ANTs base SIF image..."
    apptainer pull ${ANTS_BASE_SIF} docker://mathdugre/ants:p2-baseline
fi

if [[ ! -f ${ANTS_AMP_SIF} ]]; then
    echo "Pulling ANTs AMP SIF image..."
    apptainer pull ${ANTS_AMP_SIF} docker://mathdugre/ants:amp-miccai2026
fi

echo "Setup completed"
