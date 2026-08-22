#!/usr/bin/env bash
set -euo pipefail

for config in configs/smoke/ablation_*_smoke.yaml
do
  python -m fcp_shift.cli run --config "$config" --force
done

