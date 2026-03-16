#!/usr/bin/env bash
# ==============================================================================
# download_dataset.sh — Wrapper để tải dataset từ HuggingFace
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-configs/pod_train_config.yaml}"

cd "$PROJECT_DIR"

# --- Thiết lập HuggingFace cache (phải khớp với train.sh) ---
export HF_DATASETS_CACHE="/backup/data/art-gen/hf_cache/datasets"
export HF_HOME="/backup/data/art-gen/hf_cache"
export TRANSFORMERS_CACHE="/backup/data/art-gen/hf_cache/transformers"

echo "Chạy download dataset với config: $CONFIG"
echo ""

PYTHONPATH=. python control/download_dataset.py --config "$CONFIG"
