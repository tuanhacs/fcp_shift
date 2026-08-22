#!/usr/bin/env bash
#SBATCH --job-name=fcp-ablation
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ablation_%A_%a.log

set -euo pipefail
mkdir -p logs

configs=(
  configs/ablation/corollary.yaml
  configs/ablation/delta.yaml
  configs/ablation/models.yaml
  configs/ablation/timing.yaml
  configs/ablation/weights.yaml
  configs/ablation/baselines.yaml
)

python -m fcp_shift.cli run --config "${configs[$SLURM_ARRAY_TASK_ID]}"
