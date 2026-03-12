#!/usr/bin/env bash
# ==============================================================================
# sanity_check.sh — Wrapper để chạy sanity check
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-configs/pod_train_config.yaml}"

cd "$PROJECT_DIR"

# --- Thiết lập HuggingFace cache (phải khớp với train.sh) ---
export HF_DATASETS_CACHE="/workspace/hf_cache/datasets"
export HF_HOME="/tmp/hf_cache"
export TRANSFORMERS_CACHE="/tmp/hf_cache/transformers"

echo "Chạy sanity check với config: $CONFIG"
echo ""

PYTHONPATH=. python control/sanity_check.py --config "$CONFIG"
