#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
for stage in 1 2 3 4; do
    python reproduce_rf_equal4.py --stages "$stage" | tee "logs/Stage${stage}_rerun.log"
done
python verify_all.py | tee logs/All80_Verification.log
