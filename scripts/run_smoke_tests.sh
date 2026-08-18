#!/usr/bin/env bash
set -euo pipefail

python -m fcp_shift.cli run --config configs/smoke/covariate_shift_smoke.yaml --force
python -m fcp_shift.cli run --config configs/smoke/transport_shift_smoke.yaml --force
python -m fcp_shift.cli run --config configs/smoke/asymptotic_smoke.yaml --force

