#!/usr/bin/env bash
#SBATCH --job-name=fcp-asymptotic
#SBATCH --array=0-0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/asymptotic_%A_%a.log

set -euo pipefail
mkdir -p logs

python -m fcp_shift.cli run --config configs/main/asymptotic.yaml

