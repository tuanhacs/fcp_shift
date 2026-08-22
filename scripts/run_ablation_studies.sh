#!/usr/bin/env bash
set -euo pipefail

for config in \
  configs/ablation/corollary.yaml \
  configs/ablation/delta.yaml \
  configs/ablation/models.yaml \
  configs/ablation/timing.yaml \
  configs/ablation/weights.yaml \
  configs/ablation/baselines.yaml
do
  python -m fcp_shift.cli run --config "$config" "$@"
done

