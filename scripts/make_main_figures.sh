#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d outputs ]]; then
  echo "No outputs directory exists. Run the experiments first." >&2
  exit 1
fi

find outputs -type f -name '*.pdf' -print | sort

