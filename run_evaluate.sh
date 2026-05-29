#!/bin/bash
#SBATCH --job-name=val-lora
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

uv run python evaluate_declaration.py --images ./clevr_4/images --statements \
    ./data/declarations/test.json --output ./result_lora_count_5.json --lora runs/ViT-L-14_count/5.pt
