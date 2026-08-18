#!/usr/bin/env bash
#SBATCH --job-name=fcp-cov
#SBATCH --array=0-23
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/covariate_%A_%a.log

set -euo pipefail
mkdir -p logs

datasets=(adult bank_marketing electricity miniboone fashion_mnist year diamonds allstate)
weights=(exponential quadratic mahalanobis)
dataset_index=$((SLURM_ARRAY_TASK_ID / ${#weights[@]}))
weight_index=$((SLURM_ARRAY_TASK_ID % ${#weights[@]}))

python -m fcp_shift.cli run \
  --config configs/main/covariate_shift.yaml \
  --dataset "${datasets[$dataset_index]}" \
  --weight "${weights[$weight_index]}"

