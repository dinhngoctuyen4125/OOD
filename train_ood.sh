#!/bin/bash

#SBATCH --job-name=ood
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

for SEED in 0
do
    /home/ritsu/miniconda3/envs/ood/bin/python train_ood.py \
        --unlearn_dataset "codellama_all" \
        --ood_dataset "ood_codellama" \
        --base_unlearn_path "../Data-Collection/codellama/D_forget.json" \
        --base_ood_path "../Data-Collection/codellama/D_forget.json" \
        --model_name_or_path "microsoft/codebert-base" \
        --seed ${SEED}
done