#!/usr/bin/env bash
# ==============================================================================
# train.sh — Chạy training ArtDapter
#
# Usage:
#   bash control/train.sh                      # train từ đầu
#   bash control/train.sh --resume             # tự tìm checkpoint mới nhất để resume
#   bash control/train.sh --resume_from PATH   # resume từ checkpoint cụ thể
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="configs/pod_train_config.yaml"
CKPT_DIR="$PROJECT_DIR/ckpt/trained"
WANDB_CACHE="/backup/data/art-gen/wandb"

cd "$PROJECT_DIR"

# --- Parse arguments ---
RESUME_ARG=""
if [ "${1:-}" = "--resume" ]; then
    # Tự tìm checkpoint mới nhất
    LATEST=$(find "$CKPT_DIR" -name "*.ckpt" ! -name "*EXCEPTION*" 2>/dev/null | sort | tail -1)
    if [ -z "$LATEST" ]; then
        # Fallback: tìm exception checkpoint
        LATEST=$(find "$CKPT_DIR" -name "*EXCEPTION*.ckpt" 2>/dev/null | sort | tail -1)
    fi
    if [ -n "$LATEST" ]; then
        echo "Resume từ: $LATEST"
        RESUME_ARG="--resume_from $LATEST"
    else
        echo "Không tìm thấy checkpoint. Train từ đầu."
    fi
elif [ "${1:-}" = "--resume_from" ] && [ -n "${2:-}" ]; then
    RESUME_ARG="--resume_from $2"
    echo "Resume từ: $2"
fi

# --- Hiển thị info ---
echo "=========================================="
echo "  ArtDapter — Training"
echo "=========================================="
echo "Config:    $CONFIG"
echo "CKPT dir:  $CKPT_DIR"
echo "WandB dir: $WANDB_CACHE"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# --- Kiểm tra init.ckpt ---
INIT_CKPT=$(grep "init_path" "$CONFIG" | head -1 | awk '{print $2}')
if [ ! -f "$INIT_CKPT" ]; then
    echo "LỖI: Thiếu $INIT_CKPT"
    echo "Chạy trước: bash control/prepare_weights.sh"
    exit 1
fi

# --- Thiết lập HuggingFace cache ---
# Dataset cache → persist (tránh tải lại ~5GB)
export HF_DATASETS_CACHE="/backup/data/art-gen/hf_cache/datasets"
# Model/tokenizer cache
export HF_HOME="/backup/data/art-gen/hf_cache"
export TRANSFORMERS_CACHE="/backup/data/art-gen/hf_cache/transformers"

mkdir -p "$WANDB_CACHE"

# --- Chạy training ---
echo "Bắt đầu training..."
echo "  (Ctrl+C để dừng — checkpoint đã lưu tự động)"
echo ""

PYTHONPATH=. python train.py \
    --config_filepath "$CONFIG" \
    --gpus 0 \
    --wandb_dir "$WANDB_CACHE" \
    $RESUME_ARG

echo ""
echo "Training hoàn tất!"
echo "Checkpoints:"
ls -lhtr "$CKPT_DIR"/*.ckpt 2>/dev/null || echo "  (không có)"
