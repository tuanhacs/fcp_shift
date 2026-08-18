#!/usr/bin/env bash
#SBATCH --job-name=fcp-transport
#SBATCH --array=0-119
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/transport_%A_%a.log

set -euo pipefail
mkdir -p logs

datasets=(adult bank_marketing electricity miniboone fashion_mnist year diamonds allstate)
weights=(exponential quadratic mahalanobis)
rhos=(0.0 0.25 0.5 0.75 1.0)

rho_index=$((SLURM_ARRAY_TASK_ID % ${#rhos[@]}))
quotient=$((SLURM_ARRAY_TASK_ID / ${#rhos[@]}))
weight_index=$((quotient % ${#weights[@]}))
dataset_index=$((quotient / ${#weights[@]}))

python -m fcp_shift.cli run \
  --config configs/main/transport_shift.yaml \
  --dataset "${datasets[$dataset_index]}" \
  --weight "${weights[$weight_index]}" \
  --rho "${rhos[$rho_index]}"

