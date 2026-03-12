#!/usr/bin/env bash
# ==============================================================================
# inference.sh — Chạy Streamlit inference app
#
# Usage:
#   bash control/inference.sh                             # dùng pod_inference_config.yaml
#   bash control/inference.sh --ckpt path/to/model.ckpt   # chỉ định checkpoint
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CKPT_DIR="$PROJECT_DIR/ckpt/trained"

cd "$PROJECT_DIR"

# --- Parse arguments ---
CKPT_PATH=""
if [ "${1:-}" = "--ckpt" ] && [ -n "${2:-}" ]; then
    CKPT_PATH="$2"
fi

# --- Nếu --ckpt không chỉ định, tự tìm checkpoint mới nhất ---
if [ -z "$CKPT_PATH" ]; then
    CKPT_PATH=$(find "$CKPT_DIR" -name "*.ckpt" ! -name "*EXCEPTION*" 2>/dev/null | sort | tail -1)
    if [ -z "$CKPT_PATH" ]; then
        echo "LỖI: Không tìm thấy checkpoint trained nào trong $CKPT_DIR"
        echo "Hãy train trước hoặc chỉ định: bash control/inference.sh --ckpt path/to/model.ckpt"
        exit 1
    fi
fi

echo "=========================================="
echo "  ArtDapter — Inference (Streamlit)"
echo "=========================================="
echo "Checkpoint: $CKPT_PATH"
echo ""

# --- Cập nhật pod_inference_config.yaml với checkpoint path ---
# app.py luôn đọc configs/inference_config.yaml, nên ta tạo symlink/copy
# Cách an toàn nhất: dùng sed thay path trong config rồi restore
CONFIG="configs/pod_inference_config.yaml"
ORIG_CONFIG="configs/inference_config.yaml"
BACKUP="configs/.inference_config.yaml.bak"

# Backup config gốc nếu chưa có
if [ ! -f "$BACKUP" ]; then
    cp "$ORIG_CONFIG" "$BACKUP" 2>/dev/null || true
fi

# Copy pod config vào inference_config.yaml và cập nhật checkpoint path
cp "$CONFIG" "$ORIG_CONFIG"
sed -i "s|init_path:.*|init_path: $CKPT_PATH|" "$ORIG_CONFIG"

echo "Đã cập nhật $ORIG_CONFIG với checkpoint: $CKPT_PATH"
echo "Khởi động Streamlit..."
echo "  → Mở browser tại http://localhost:8501"
echo "  → Ctrl+C để dừng"
echo ""

# --- Chạy Streamlit ---
PYTHONPATH=. streamlit run inference/app.py --server.port 8501 --server.address 0.0.0.0

# --- Restore config gốc ---
if [ -f "$BACKUP" ]; then
    mv "$BACKUP" "$ORIG_CONFIG"
    echo "Đã restore $ORIG_CONFIG"
fi
