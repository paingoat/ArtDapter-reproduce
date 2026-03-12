#!/usr/bin/env bash
# ==============================================================================
# sanity_check.sh — Wrapper để chạy sanity check
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-configs/pod_train_config.yaml}"

cd "$PROJECT_DIR"

echo "Chạy sanity check với config: $CONFIG"
echo ""

PYTHONPATH=. python control/sanity_check.py --config "$CONFIG"
