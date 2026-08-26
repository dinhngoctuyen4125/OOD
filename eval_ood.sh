#!/bin/bash

#SBATCH --job-name=eval_ood
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
    /home/ritsu/miniconda3/envs/ood/bin/python eval_ood.py \
        --data_path "../Data-Collection/codellama/D_forget.json" \
        --text_field "function" \
        --ood_weights "./ood_checkpoints_codellama_${SEED}/" \
        --ood_base_model "microsoft/codebert-base" \
        --ood_setting_name "codellama" \
        --ood_type "_all" \
        --batch_size 32 \
        --seed ${SEED}
done
