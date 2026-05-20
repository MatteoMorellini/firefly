#!/bin/bash
#SBATCH --job-name=clevr4-clip
#SBATCH --output=my_output_%j.out
#SBATCH --error=my_error_%j.err
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

module purge
module load CUDA/12.3.2
module load Python/3.12.3-GCCcore-13.3.0

uv run python train_feature.py --feature color --images ./clevr4/images --statements ./data/declarations/train.json \
    --eval ./data/declarations/eval.json --limit 10000 --epochs 100 --batch-size 128
