# ArtDapter — Hướng dẫn chạy trên RunPod (RTX 5090)

> **Hardware mục tiêu**: NVIDIA RTX 5090 · 32 GB VRAM · 60 GB RAM · 12 vCPU  
> **Template**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`  
> **Chi phí tham khảo**: ~$0.76–0.89/giờ

---

## Mục lục

1. [Chuẩn bị Pod](#1-chuẩn-bị-pod)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Tải & Gộp weights](#3-tải--gộp-weights)
4. [Sanity Check](#4-sanity-check)
5. [Training](#5-training)
6. [Inference (Streamlit)](#6-inference-streamlit)
7. [Evaluation](#7-evaluation)
8. [tmux — Chạy nền an toàn](#8-tmux--chạy-nền-an-toàn)
9. [Quản lý dung lượng](#9-quản-lý-dung-lượng)
10. [Cheatsheet](#10-cheatsheet)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Chuẩn bị Pod

### 1.1 Tạo Pod trên RunPod

- Chọn GPU: **RTX 5090** (32 GB)
- Template: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Disk: ≥ **50 GB** (container volume), ≥ **20 GB** (pod volume nếu cần persist)
- Expose port **8501** nếu muốn dùng Streamlit inference qua web

### 1.2 Kết nối SSH

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/your_key
```

### 1.3 Clone repo

```bash
cd /workspace
git clone <your-repo-url> ArtDapter
cd ArtDapter
```

### 1.4 Kiểm tra hardware nhanh

```bash
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB')"
```

---

## 2. Cài đặt môi trường

### Cách nhanh (script tự động)

```bash
bash control/setup_env.sh
```

Script này sẽ:
1. Kiểm tra GPU
2. Cài Miniconda (nếu chưa có)
3. Tạo conda env `artgen` từ `environment.yaml`
4. Cài font DejaVuSans cho visualization

### Cách thủ công

```bash
# Cài Miniconda (nếu chưa có)
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc

# Tạo env
conda env create -f environment.yaml
conda activate artgen

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Kích hoạt env

Mỗi lần mở terminal mới (hoặc tmux session mới):

```bash
conda activate artgen
```

---

## 3. Tải & Gộp weights

### Bước 1: Tải pre-trained weights

```bash
conda activate artgen
bash control/download_weights.sh
```

Tải 2 file:
| File | Kích thước | Nguồn |
|------|-----------|-------|
| `v1-5-pruned.ckpt` | ~4 GB | Stable Diffusion v1.5 |
| `ella-sd1.5-tsc-t5xl.safetensors` | ~200 MB | ELLA adapter |

### Bước 2: Gộp weights

```bash
bash control/prepare_weights.sh
```

Gộp SD v1.5 + ELLA → `ckpt/init/init.ckpt` (~5 GB).

> **Lưu ý**: Trên máy local, script **KHÔNG** xóa raw weights sau khi gộp (khác Kaggle).

### Kiểm tra kết quả

```bash
ls -lh ckpt/init/
# Sẽ thấy:
#   init.ckpt                          (~5 GB)
#   v1-5-pruned.ckpt                   (~4 GB)
#   ella-sd1.5-tsc-t5xl.safetensors   (~200 MB)
```

---

## 4. Sanity Check

Kiểm tra toàn bộ pipeline trước khi train (~2 phút).

```bash
conda activate artgen
bash control/sanity_check.sh
```

Script chạy 6 bước:
1. **Import modules** — tất cả dependencies + project modules + xformers
2. **GPU & CUDA** — CUDA available, bf16 support
3. **Config** — load & validate `pod_train_config.yaml`, in effective batch size
4. **Weights** — `init.ckpt` tồn tại & kích thước hợp lý
5. **Dataset** — load 1 batch từ CompArt qua HuggingFace
6. **Forward + Backward** — 1 training step, báo loss + peak VRAM

Kết quả mong đợi:
```
  [PASS] Core dependencies
  [PASS] Project modules
  [PASS] xformers (optional)
  [PASS] CUDA available
         GPU: NVIDIA GeForce RTX 5090 | VRAM: 32.0 GB | CUDA: 12.8
  [PASS] bf16 support
  [PASS] Load config: configs/pod_train_config.yaml
         Effective batch size: 11 x 2 = 22
         Precision: bf16-mixed | Steps: 5000
  [PASS] init.ckpt exists
  [PASS] Dataset load
  [PASS] Forward + backward pass
         Loss: x.xxxx
         Peak VRAM (batch=1): ~xx.xx GB

  Result: 10 passed, 0 failed
```

> Lần đầu chạy dataset sẽ tải CompArt từ HuggingFace (~vài GB).

---

## 5. Training

### Chạy training

```bash
conda activate artgen

# Train từ đầu
bash control/train.sh

# Resume từ checkpoint mới nhất (tự tìm trong ckpt/trained/)
bash control/train.sh --resume

# Resume từ checkpoint cụ thể
bash control/train.sh --resume_from ckpt/trained/step=2000-loss=0.05.ckpt
```

### Config chính (`configs/pod_train_config.yaml`)

| Param | Value | Ghi chú |
|-------|-------|---------|
| `precision` | `bf16-mixed` | RTX 5090 native bf16 |
| `batch_size` | `11` | Mỗi GPU step |
| `accumulate_grad_batches` | `2` | Effective batch = 22 |
| `training_steps` | `5000` | Tổng steps |
| `save_top_k` | `3` | Giữ 3 checkpoint gần nhất |
| `log_frequency` | `500` | WandB log mỗi 500 steps |
| `learning_rate` | `0.0001` | AdamW |
| `ddim_steps` | `20` | Sampling cho visualization |
| `use_checkpoint` | `False` | Gradient checkpointing (tắt vì 32GB đủ) |

### Monitoring

- **WandB**: Mở link WandB in ra terminal khi bắt đầu train
- **nvidia-smi**: Mở terminal thứ 2 để theo dõi VRAM

```bash
# Terminal 2
watch -n 5 nvidia-smi
```

### Checkpoints

Checkpoints lưu tại `ckpt/trained/`. Với `save_top_k=3`, chỉ giữ 3 file mới nhất (~7 GB mỗi file = ~21 GB max).

```bash
ls -lhtr ckpt/trained/*.ckpt
```

### Train trong tmux (khuyến nghị)

Xem [mục 8](#8-tmux--chạy-nền-an-toàn) để chạy training an toàn trong tmux, tránh mất progress khi SSH bị ngắt.

---

## 6. Inference (Streamlit)

### Chạy inference app

```bash
conda activate artgen

# Tự tìm checkpoint mới nhất
bash control/inference.sh

# Chỉ định checkpoint
bash control/inference.sh --ckpt ckpt/trained/step=5000-loss=0.03.ckpt
```

### Truy cập giao diện

- **Local**: http://localhost:8501
- **RunPod**: Dùng port forwarding SSH hoặc expose port 8501 khi tạo pod:
  ```bash
  ssh -L 8501:localhost:8501 root@<pod-ip> -p <port>
  ssh -L 8501:localhost:8501 root@82.221.170.234 -p 23863
  ```
  Mở http://localhost:8501 trên máy local.

---

## 7. Evaluation

```bash
conda activate artgen
cd /workspace/ArtDapter

PYTHONPATH=. python eval.py \
    --config_filepath configs/pod_eval_config.yaml \
    --gpus 0 \
    --ckpt_path ckpt/trained/<your_checkpoint>.ckpt
```

> **Lưu ý**: Cần kiểm tra lại `eval.py` nếu có trong project. Script evaluation sử dụng FID/CLIP scores.

---

## 8. tmux — Chạy nền an toàn

tmux giúp giữ process chạy kể cả khi SSH bị ngắt.

### Cheat sheet tmux

```bash
# Tạo session mới
tmux new -s train

# Trong tmux, chạy training
conda activate artgen
bash control/train.sh

# Detach (thoát mà không tắt): Ctrl+B rồi nhấn D

# Quay lại session
tmux attach -t train

# Xem các session đang chạy
tmux ls

# Kill session
tmux kill-session -t train
```

### Workflow khuyến nghị

```bash
# Terminal 1: Training
tmux new -s train
conda activate artgen
bash control/train.sh
# Ctrl+B, D để detach

# Terminal 2: Monitor
tmux new -s monitor
watch -n 5 nvidia-smi
# Ctrl+B, D để detach

# Quay lại bất cứ lúc nào
tmux attach -t train
tmux attach -t monitor
```

---

## 9. Quản lý dung lượng

### Ước tính dung lượng

| Item | Kích thước |
|------|-----------|
| Code + configs | ~50 MB |
| Raw weights (SD + ELLA) | ~4.2 GB |
| init.ckpt | ~5 GB |
| Checkpoints (3 × 7 GB) | ~21 GB |
| CompArt dataset (HF cache) | ~5–10 GB |
| WandB cache | ~2–5 GB |
| Conda env | ~10 GB |
| **Tổng** | **~47–55 GB** |

### Kiểm soát HuggingFace cache

Mặc định HF cache nằm ở `~/.cache/huggingface/`. Trên RunPod nên chuyển sang `/tmp`:

```bash
export HF_HOME="/tmp/hf_cache"
export HF_DATASETS_CACHE="/tmp/hf_cache/datasets"
export TRANSFORMERS_CACHE="/tmp/hf_cache/transformers"
```

> `train.sh` đã tự set các biến này.

### WandB cache

Training script đặt WandB dir ở `/tmp/wandb` để tránh chiếm disk chính:

```bash
# WandB media cache có thể lớn dần
du -sh /tmp/wandb/
# Dọn nếu cần
rm -rf /tmp/wandb/wandb/
```

### Dọn dẹp nhanh

```bash
# Xem dung lượng tổng quan
df -h /workspace

# Dung lượng từng thư mục
du -sh /workspace/ArtDapter/ckpt/*
du -sh /tmp/hf_cache/
du -sh /tmp/wandb/

# Xóa conda cache (packages đã cài)
conda clean --all -y
```

---

## 10. Cheatsheet

### Monitoring

```bash
nvidia-smi                           # VRAM, GPU util
watch -n 5 nvidia-smi                # auto-refresh mỗi 5s
df -h                                # disk usage (all mounts)
du -sh <folder>                      # folder size
htop                                 # CPU, RAM
free -h                              # RAM summary
```

### Conda

```bash
conda activate artgen                # kích hoạt env
conda deactivate                     # thoát env
conda env list                       # danh sách env
conda env remove -n artgen           # xóa env (cài lại)
conda clean --all -y                 # dọn cache
```

### tmux

```bash
tmux new -s <name>                   # tạo session mới
tmux attach -t <name>                # quay lại session
tmux ls                              # list sessions
tmux kill-session -t <name>          # kill session
# Ctrl+B, D                         # detach
# Ctrl+B, [                         # scroll mode (q để thoát)
```

### Training pipeline

```bash
# Full pipeline từ đầu
bash control/setup_env.sh            # 1 lần
conda activate artgen
bash control/download_weights.sh     # 1 lần
bash control/prepare_weights.sh      # 1 lần
bash control/sanity_check.sh         # kiểm tra
bash control/train.sh                # train
bash control/inference.sh            # inference

# Resume training
bash control/train.sh --resume
```

### WandB

```bash
wandb login                          # login bằng API key
wandb sync <run_dir>                 # sync offline run
```

### Useful

```bash
# Kiểm tra VRAM real-time
nvidia-smi dmon -s um -d 5

# Số file checkpoint
ls -1 ckpt/trained/*.ckpt | wc -l

# Tổng kích thước checkpoints
du -sh ckpt/trained/

# Kill training (nếu chạy background)
pkill -f "python train.py"
```

---

## 11. Troubleshooting

### `CUDA out of memory`

- Giảm `batch_size` trong `pod_train_config.yaml` (ví dụ: 11 → 8)
- Bật gradient checkpointing: đổi `use_checkpoint: True` trong unet_config
- Kiểm tra có process khác chiếm VRAM: `nvidia-smi`

### `ModuleNotFoundError`

- Đảm bảo đã `conda activate artgen`
- Kiểm tra `PYTHONPATH=.` khi chạy scripts

### Dataset download chậm / timeout

```bash
# Set HF mirror nếu cần
export HF_ENDPOINT="https://hf-mirror.com"
```

### SSH bị ngắt giữa chừ

- Dùng tmux (mục 8). Training vẫn chạy nền.
- Quay lại: `tmux attach -t train`

### WandB không log

- Kiểm tra `wandb login`
- Nếu bị firewall, dùng offline mode:
  ```bash
  export WANDB_MODE=offline
  # Sau training, sync:
  wandb sync /tmp/wandb/wandb/latest-run
  ```

### Checkpoint quá lớn, hết disk

- `save_top_k: 3` tự xóa checkpoint cũ
- Nếu vẫn hết: giảm `save_top_k` hoặc dọn manual:
  ```bash
  ls -lhtr ckpt/trained/*.ckpt
  rm ckpt/trained/<old_checkpoint>.ckpt
  ```

---

## Pipeline tổng quan

```
┌──────────────────────────────────────────────────┐
│  1. setup_env.sh        → Conda env + font       │
│  2. download_weights.sh → SD v1.5 + ELLA         │
│  3. prepare_weights.sh  → init.ckpt              │
│  4. sanity_check.sh     → Verify toàn bộ         │
│  5. train.sh            → Training (WandB log)    │
│  6. inference.sh        → Streamlit app           │
└──────────────────────────────────────────────────┘
```
