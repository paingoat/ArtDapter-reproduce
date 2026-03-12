#!/usr/bin/env bash
# ==============================================================================
# setup_env.sh — Cài đặt môi trường conda + font cho ArtDapter trên RunPod
# Chạy 1 lần duy nhất khi tạo pod mới.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  ArtDapter — Setup Môi Trường"
echo "=========================================="

# --- 1. Kiểm tra GPU ---
echo ""
echo "[1/4] Kiểm tra GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# --- 2. Kiểm tra miniconda ---
echo "[2/4] Kiểm tra Miniconda..."
if ! command -v conda &> /dev/null; then
    echo "Miniconda chưa cài. Đang cài đặt..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    echo "Miniconda đã cài xong. Source lại shell:"
    echo "  source ~/.bashrc"
    echo "Sau đó chạy lại script này."
    exit 0
else
    echo "Conda: $(conda --version)"
fi

# --- 3. Tạo conda env ---
echo ""
echo "[3/4] Tạo conda environment 'artgen'..."
ENV_YAML="$PROJECT_DIR/environment.yaml"

# Tự động accept ToS để không bị block trong môi trường non-interactive
conda config --set auto_accept_tos yes 2>/dev/null || true

if conda env list | grep -q "^artgen "; then
    echo "Environment 'artgen' đã tồn tại. Bỏ qua."
    echo "  (Muốn tạo lại: conda env remove -n artgen --yes && bash $0)"
else
    conda env create -f "$ENV_YAML" --yes
    echo "Environment 'artgen' đã tạo xong."
fi

# --- 4. Cài font DejaVuSans cho visualization ---
echo ""
echo "[4/4] Cài font DejaVuSans..."
FONT_DIR="$PROJECT_DIR/font"
FONT_FILE="$FONT_DIR/DejaVuSans.ttf"
if [ -f "$FONT_FILE" ]; then
    echo "Font đã có."
else
    mkdir -p "$FONT_DIR"
    # Thử symlink từ hệ thống trước
    SYS_FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if [ -f "$SYS_FONT" ]; then
        ln -sf "$SYS_FONT" "$FONT_FILE"
        echo "Symlink từ $SYS_FONT"
    else
        # Cài package font
        sudo apt-get update -qq && sudo apt-get install -y -qq fonts-dejavu-ttf > /dev/null 2>&1
        if [ -f "$SYS_FONT" ]; then
            ln -sf "$SYS_FONT" "$FONT_FILE"
            echo "Đã cài font và tạo symlink."
        else
            echo "CẢNH BÁO: Không tìm thấy font DejaVuSans. Visualization có thể bị lỗi."
        fi
    fi
fi

# --- Tạo thư mục cần thiết ---
mkdir -p "$PROJECT_DIR/ckpt/init"
mkdir -p "$PROJECT_DIR/ckpt/trained"

echo ""
echo "=========================================="
echo "  Setup hoàn tất!"
echo "=========================================="
echo ""
echo "Bước tiếp theo:"
echo "  conda activate artgen"
echo "  bash control/download_weights.sh"
