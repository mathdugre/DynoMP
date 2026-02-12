INPUT_DIR=$HOME/datasets/OASIS-2d
OUTPUT_DIR=$PWD/models
mkdir -p $OUTPUT_DIR

uv run ./code/voxelmorph_train.py \
    --input $INPUT_DIR \
    --output $OUTPUT_DIR/2d-default.pt \
    --dim 2 \
    --epochs 100 \
    --save-every 100\
    --strategy default

uv run ./code/voxelmorph_train.py \
    --input $INPUT_DIR \
    --output $OUTPUT_DIR/2d-dmp.pt \
    --dim 2 \
    --epochs 100 \
    --save-every 100\
    --strategy dmp

uv run ./code/voxelmorph_train.py \
    --input $INPUT_DIR \
    --output $OUTPUT_DIR/2d-bf16.pt \
    --dim 2 \
    --epochs 100 \
    --save-every 100\
    --strategy bf16

uv run ./code/voxelmorph_train.py \
    --input $INPUT_DIR \
    --output $OUTPUT_DIR/2d-amp.pt \
    --dim 2 \
    --epochs 100 \
    --save-every 100\
    --strategy amp
