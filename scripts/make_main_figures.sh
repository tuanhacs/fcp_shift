#!/usr/bin/env bash
set -euo pipefail

python -m fcp_shift.cli figures --config configs/main/covariate_shift.yaml
python -m fcp_shift.cli figures --config configs/main/transport_shift.yaml
