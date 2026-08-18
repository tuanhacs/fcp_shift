#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_covariate_shift.sh
bash scripts/run_transport_shift.sh
bash scripts/run_asymptotic.sh

