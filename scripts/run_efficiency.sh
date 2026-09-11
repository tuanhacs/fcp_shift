#!/usr/bin/env bash
set -euo pipefail

python -m fcp_shift.cli run --config configs/ablation/efficiency.yaml "$@"
