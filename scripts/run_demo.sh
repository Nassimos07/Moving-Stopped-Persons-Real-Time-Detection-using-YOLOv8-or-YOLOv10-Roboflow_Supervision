#!/usr/bin/env bash
set -euo pipefail

python3 -m movement_detection run \
  --config configs/default.yaml \
  --mode relative \
  --filter_mode all
