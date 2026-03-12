#!/usr/bin/env bash
# ==============================================================================
# download_weights.sh — Tải pre-trained weights (SD v1.5 + ELLA)
# Chạy 1 lần. Nếu đã có file thì tự bỏ qua.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INIT_DIR="$PROJECT_DIR/ckpt/init"

mkdir -p "$INIT_DIR"

echo "=========================================="
echo "  ArtDapter — Tải Pre-trained Weights"
echo "=========================================="

# --- Stable Diffusion v1.5 (~4GB) ---
SD_FILE="$INIT_DIR/v1-5-pruned.ckpt"
if [ -f "$SD_FILE" ]; then
    echo "[1/2] SD v1.5 đã có: $SD_FILE ($(du -h "$SD_FILE" | cut -f1))"
else
    echo "[1/2] Đang tải Stable Diffusion v1.5 (~4GB)..."
    wget -q --show-progress -O "$SD_FILE" \
        https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
    echo "  Đã tải: $SD_FILE"
fi

# --- ELLA weights (~200MB) ---
ELLA_FILE="$INIT_DIR/ella-sd1.5-tsc-t5xl.safetensors"
if [ -f "$ELLA_FILE" ]; then
    echo "[2/2] ELLA đã có: $ELLA_FILE ($(du -h "$ELLA_FILE" | cut -f1))"
else
    echo "[2/2] Đang tải ELLA weights (~200MB)..."
    wget -q --show-progress -O "$ELLA_FILE" \
        https://huggingface.co/QQGYLab/ELLA/resolve/main/ella-sd1.5-tsc-t5xl.safetensors
    echo "  Đã tải: $ELLA_FILE"
fi

echo ""
echo "Weights đã sẵn sàng:"
ls -lh "$INIT_DIR"/
echo ""
echo "Bước tiếp theo:"
echo "  bash control/prepare_weights.sh"
