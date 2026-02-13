#!/bin/bash
#SBATCH --job-name=vm3d-amp
#SBATCH --gpus=h100_20gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/vm3d-amp-%A_%a.out
#SBATCH --account=def-glatard
set -eu

INPUT_DIR=$HOME/datasets/OASIS
OUTPUT_DIR=$PWD/models
mkdir -p $OUTPUT_DIR

export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run ./code/voxelmorph_train.py \
    --input $INPUT_DIR \
    --output $OUTPUT_DIR/3d-amp.pt \
    --dim 3 \
    --epochs 100 \
    --save-every 10 \
    --strategy amp
