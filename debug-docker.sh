OUTPUT_DIR=/output/debug
FIXED_IMG=/data/template.nii.gz
INPUT_IMG=/data/sub-s007.nii.gz

# Registration
docker run --rm -it \
    -v $PWD/data:/data \
    -v $PWD/output:/output \
    ants:p3 \
    antsRegistration \
    --verbose 1 \
    --dimensionality 3 \
    --collapse-output-transforms 1 \
    --use-histogram-matching 0 \
    --winsorize-image-intensities [0.005,0.995] \
    --interpolation Linear \
    --random-seed 1 \
    --output [${OUTPUT_DIR}/,${OUTPUT_DIR}/Warped.nii.gz] \
    --initial-moving-transform [${FIXED_IMG},${INPUT_IMG},1] \
    --transform Rigid[0.1] \
    --metric MI[${FIXED_IMG},${INPUT_IMG},1,32,Regular,0.25] \
    --convergence [1000x500x250,1e-6,10] \
    --shrink-factors 8x4x2 \
    --smoothing-sigmas 3x2x1vox \
    --transform Affine[0.1] \
    --metric MI[${FIXED_IMG},${INPUT_IMG},1,32,Regular,0.25] \
    --convergence [1000x500x250,1e-6,10] \
    --shrink-factors 8x4x2 \
    --smoothing-sigmas 3x2x1vox \
    --transform SyN[ 0.1,3,0 ] \
    --metric CC[${FIXED_IMG},${INPUT_IMG},1,4] \
    --convergence [100x70,1e-6,10] \
    --shrink-factors 8x4 \
    --smoothing-sigmas 3x2vox \
    > log-debug.txt 2>&1