#!/usr/bin/env bash
# ==============================================================================
# prepare_weights.sh — Gộp weights SD v1.5 + ELLA → init.ckpt
# Chạy 1 lần. Nếu init.ckpt đã tồn tại thì bỏ qua.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INIT_DIR="$PROJECT_DIR/ckpt/init"
INIT_CKPT="$INIT_DIR/init.ckpt"
CONFIG="configs/pod_train_config.yaml"

cd "$PROJECT_DIR"

echo "=========================================="
echo "  ArtDapter — Prepare Initial Weights"
echo "=========================================="

# Kiểm tra raw weights
if [ ! -f "$INIT_DIR/v1-5-pruned.ckpt" ]; then
    echo "LỖI: Thiếu v1-5-pruned.ckpt. Chạy trước: bash control/download_weights.sh"
    exit 1
fi
if [ ! -f "$INIT_DIR/ella-sd1.5-tsc-t5xl.safetensors" ]; then
    echo "LỖI: Thiếu ella-sd1.5-tsc-t5xl.safetensors. Chạy trước: bash control/download_weights.sh"
    exit 1
fi

if [ -f "$INIT_CKPT" ]; then
    echo "init.ckpt đã tồn tại ($(du -h "$INIT_CKPT" | cut -f1)). Bỏ qua."
    echo "  (Xóa để tạo lại: rm $INIT_CKPT)"
else
    echo "Đang gộp weights..."
    PYTHONPATH=. python prepare_weights.py \
        --init_dir "$INIT_DIR" \
        --output init.ckpt \
        --config "$CONFIG" \
        --precision bf16-mixed
    echo ""
    echo "init.ckpt đã tạo: $(du -h "$INIT_CKPT" | cut -f1)"
fi

echo ""
echo "Bước tiếp theo:"
echo "  bash control/download_dataset.sh"
