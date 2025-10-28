#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # or set to your repo root
cd "$ROOT_DIR"

DATASETS=(vqa-v2 gqa vizwiz text-vqa refcoco ocid-ref tally-qa pope vsr)

for ds in "${DATASETS[@]}"; do
  echo "==> Downloading: $ds"
  if python scripts/datasets/prepare.py --dataset_family "$ds"; then
    echo "✓ Done: $ds"
  else
    echo "✗ Failed: $ds" >&2
  fi
done
