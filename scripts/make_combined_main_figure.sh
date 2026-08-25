#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <weight> <rho> [dataset ...]" >&2
  echo "Example: $0 exponential 0.50 year adult" >&2
  exit 2
fi

weight=$1
rho=$2
shift 2

args=(
  python -m fcp_shift.cli main-figure
  --covariate-config configs/main/covariate_shift.yaml
  --transport-config configs/main/transport_shift.yaml
  --weight "$weight"
  --rho "$rho"
)

if [[ $# -gt 0 ]]; then
  args+=(--datasets "$@")
fi

"${args[@]}"
