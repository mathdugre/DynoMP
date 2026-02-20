DATASET_DIR=$HOME/datasets
OUTPUT_DIR=$PWD/models
mkdir -p $OUTPUT_DIR

uv run ./code/mnist_train.py \
    --input $DATASET_DIR \
    --output $OUTPUT_DIR/mnist.pt \
    --epochs 14 \
    --strategy default

uv run ./code/mnist_train.py \
    --input $DATASET_DIR \
    --output $OUTPUT_DIR/mnist-dynomp.pt \
    --epochs 14 \
    --strategy dynomp

uv run ./code/mnist_train.py \
    --input $DATASET_DIR \
    --output $OUTPUT_DIR/mnist-fp16.pt \
    --epochs 14 \
    --strategy fp16

uv run ./code/mnist_train.py \
    --input $DATASET_DIR \
    --output $OUTPUT_DIR/mnist-bf16.pt \
    --epochs 14 \
    --strategy bf16

uv run ./code/mnist_train.py \
    --input $DATASET_DIR \
    --output $OUTPUT_DIR/mnist-amp.pt \
    --epochs 14 \
    --strategy amp