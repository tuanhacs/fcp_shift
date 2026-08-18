#!/usr/bin/env bash
set -euo pipefail

python -m fcp_shift.cli run --config configs/main/transport_shift.yaml "$@"

