#!/usr/bin/env bash
set -e
CONFIG=${1:-configs/srgan.yaml}
echo "Starting training with config: $CONFIG"
python scripts/train.py --config "$CONFIG"
