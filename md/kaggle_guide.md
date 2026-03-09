# 🎨 Hướng Dẫn Train & Inference ArtDapter trên Kaggle (GPU P100)

> **GPU**: NVIDIA Tesla P100 — 16GB HBM2 VRAM  
> **Repo**: `https://github.com/paingoat/ArtDapter-reproduce.git`  
> **Config**: `configs/train_config_kaggle.yaml`  
> **Dataset**: `https://www.kaggle.com/datasets/thoandanh/compart`

---

## 📋 Mục Lục

1. [Tổng quan & Lưu ý quan trọng](#1-tổng-quan--lưu-ý-quan-trọng)
2. [Chuẩn bị trước khi bắt đầu](#2-chuẩn-bị-trước-khi-bắt-đầu)
3. [Các cell cho Training](#3-các-cell-cho-training)
4. [Các cell cho Inference](#4-các-cell-cho-inference)
5. [Cheat Sheet: Session mới](#5-cheat-sheet-session-mới)
6. [FAQ & Troubleshooting](#6-faq--troubleshooting)

---

## 1. Tổng quan & Lưu ý quan trọng

### Bộ nhớ trên Kaggle

| Loại | Giới hạn | Đường dẫn | Ghi chú |
|---|---|---|---|
| **GPU VRAM** | 16 GB | — | Chứa model + tensors khi tính toán |
| **RAM** | 29 GB | — | Chứa dataset, biến Python |
| **Disk** (tổng) | ~58 GB | `/`, `~/.cache/` | Hệ thống + packages + cache |
| **Output** | ~20 GB | `/kaggle/working/` | File output, checkpoint |
| **Input** | **Không giới hạn** | `/kaggle/input/` | ⭐ Kaggle Dataset — **MIỄN PHÍ** |
| `/tmp/` | Dùng chung Disk | `/tmp/` | **Không tính vào quota** |

> ⚠️ **Nguyên nhân tràn Disk phổ biến nhất**: HuggingFace tự tải T5-XL (~10GB) vào `~/.cache/huggingface/` → chiếm disk nhanh chóng.

### Chiến lược tối ưu (đã áp dụng trong guide này)

1. ✅ CompArt dataset → đã upload lên **Kaggle Dataset** (`/kaggle/input/`) → config trỏ thẳng, **0 download, 0 byte disk**
2. ✅ HuggingFace cache (T5-XL model) → chuyển sang `/tmp/` → **không tính quota**
3. ✅ Pre-trained weights (SD v1.5, ELLA) → tải vào `/tmp/`, dùng xong xóa
4. ✅ Checkpoint training → lưu ở `/kaggle/working/` (~4GB/file)

### Cấu hình training

| Setting | Giá trị | Lý do |
|---|---|---|
| `training_steps` | 5,000 | Survey reproduction (ước tính ~2-3 giờ) |
| `batch_size` | 4 | P100 16GB VRAM |
| `accumulate_grad_batches` | 6 | Effective batch = 4 × 6 = 24 ≈ 22 gốc |
| `precision` | `16-mixed` | P100 **không** hỗ trợ bf16 |
| `use_checkpoint` | True | Gradient checkpointing tiết kiệm VRAM |
| `log_frequency` | 200 | Checkpoint thường xuyên |
| `dataset_path` | `/kaggle/input/compart` | Load trực tiếp từ Kaggle Input |

---

## 2. Chuẩn bị trước khi bắt đầu

### Tạo Kaggle Notebook

1. Vào [kaggle.com](https://www.kaggle.com) → **"+ Create"** → **"New Notebook"**
2. Đổi tên: `ArtDapter-Training-P100`

### Cài đặt Notebook (Settings ⚙️)

| Setting | Giá trị |
|---|---|
| **Accelerator** | GPU P100 |
| **Persistence** | Variables and Files |
| **Internet** | ON |

### Add Data

1. Click **"+ Add Data"** ở thanh bên phải
2. Tìm `thoandanh/compart` → **Add**
3. Dataset sẽ ở `/kaggle/input/compart/`

---

## 3. Các Cell cho Training

Copy-paste từng cell bên dưới vào notebook Kaggle theo đúng thứ tự.

---

### Cell 0 — ⚡ Setup Môi Trường (CHẠY ĐẦU TIÊN)

> **Cell quan trọng nhất!** Phải chạy trước mọi cell khác để tránh tràn disk.

```python
import os

# ====== CHUYỂN TẤT CẢ CACHE SANG /tmp/ (KHÔNG TÍNH VÀO DISK QUOTA) ======
# T5-XL model (~10GB) sẽ cache ở đây thay vì ~/.cache/ (tốn disk quota)
os.environ['HF_HOME'] = '/tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache/transformers'
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_cache/datasets'

# Kiểm tra dataset CompArt đã được add vào notebook chưa
if os.path.exists('/kaggle/input/compart'):
    print('✅ Dataset CompArt đã có tại /kaggle/input/compart/')
else:
    print('⚠️ Chưa add dataset "thoandanh/compart" vào notebook!')
    print('   → Click "+ Add Data" → tìm "thoandanh/compart" → Add')

print(f'\n📁 HF_HOME = {os.environ["HF_HOME"]}')
```

**Mục đích**: Chuyển HuggingFace cache (T5-XL ~10GB) sang `/tmp/` để không tốn disk quota. Dataset CompArt load trực tiếp từ `/kaggle/input/compart/` qua config — không cần symlink.

---

### Cell 1 — 🔍 Kiểm tra GPU

```python
import torch
!nvidia-smi
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f"\n✅ GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB | CUDA: {torch.version.cuda}")

if "P100" in gpu_name:
    print("✅ Đúng GPU P100")
else:
    print(f"⚠️ GPU không phải P100: {gpu_name} — vẫn chạy được nếu VRAM >= 16GB")
```

---

### Cell 2 — 📥 Clone Repo

```python
import os

GITHUB_REPO = "https://github.com/paingoat/ArtDapter-reproduce.git"

if not os.path.exists('/kaggle/working/ArtDapter'):
    !git clone {GITHUB_REPO} /kaggle/working/ArtDapter

%cd /kaggle/working/ArtDapter

# Tạo thư mục checkpoint
!mkdir -p /kaggle/working/ckpt/trained
!mkdir -p /kaggle/working/ArtDapter/ckpt/init
```

---

### Cell 3 — 📦 Cài Dependencies

```python
!pip install -q pytorch-lightning lightning transformers diffusers \
    omegaconf einops wandb datasets safetensors open-clip-torch tqdm Pillow
```

---

### Cell 4 — ⬇️ Tải Pre-trained Weights

> Tải vào `/tmp/` rồi copy vào repo. **Mỗi session mới phải chạy lại** (vì `/tmp/` bị xóa khi session kết thúc).

```python
import os

LOCAL_INIT = '/kaggle/working/ArtDapter/ckpt/init'
TMP_WEIGHTS = '/tmp/pretrained_weights'
os.makedirs(TMP_WEIGHTS, exist_ok=True)

# --- Stable Diffusion v1.5 weights (~4GB) ---
sd_path = f'{TMP_WEIGHTS}/v1-5-pruned.ckpt'
sd_link = f'{LOCAL_INIT}/v1-5-pruned.ckpt'
if not os.path.exists(sd_link):
    if not os.path.exists(sd_path):
        print('📥 Downloading Stable Diffusion v1.5 weights (~4GB)...')
        !wget -q --show-progress -O {sd_path} \
            https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
    !ln -sf {sd_path} {sd_link}
    print('✅ SD v1.5 linked')

# --- ELLA weights (~200MB) ---
ella_path = f'{TMP_WEIGHTS}/ella-sd1.5-tsc-t5xl.safetensors'
ella_link = f'{LOCAL_INIT}/ella-sd1.5-tsc-t5xl.safetensors'
if not os.path.exists(ella_link):
    if not os.path.exists(ella_path):
        print('📥 Downloading ELLA weights (~200MB)...')
        !wget -q --show-progress -O {ella_path} \
            https://huggingface.co/QQGYLab/ELLA/resolve/main/ella-sd1.5-tsc-t5xl.safetensors
    !ln -sf {ella_path} {ella_link}
    print('✅ ELLA linked')

# Kiểm tra disk
!echo ""; df -h / | head -2; echo ""; du -sh /tmp/pretrained_weights/
```

**Mục đích**: Tải weights vào `/tmp/` (không tốn disk quota), symlink vào repo.

---

### Cell 5 — 🔧 Prepare Initial Weights

> Chỉ cần chạy **1 lần đầu tiên**. Nếu đã có `init.ckpt` (từ session trước hoặc Kaggle Dataset), bỏ qua cell này.

```python
import os

init_ckpt = '/kaggle/working/ArtDapter/ckpt/init/init.ckpt'

if not os.path.exists(init_ckpt):
    print('🔧 Preparing initial weights...')
    !python prepare_weights.py \
        --init_dir /kaggle/working/ArtDapter/ckpt/init \
        --output init.ckpt \
        --config configs/train_config_kaggle.yaml
    print('✅ init.ckpt đã tạo xong')

    # Xóa raw weights (đã merge vào init.ckpt rồi) để tiết kiệm disk
    !rm -f /kaggle/working/ArtDapter/ckpt/init/v1-5-pruned.ckpt
    !rm -f /kaggle/working/ArtDapter/ckpt/init/ella-sd1.5-tsc-t5xl.safetensors
    print('🗑️ Đã xóa raw weights (đã merge vào init.ckpt)')
else:
    print('✅ init.ckpt đã tồn tại')
```

> 💡 **Mẹo**: Sau lần đầu, upload `init.ckpt` lên **Kaggle Dataset** rồi symlink:
> ```python
> !ln -sf /kaggle/input/artdapter-weights/init.ckpt /kaggle/working/ArtDapter/ckpt/init/init.ckpt
> ```

---

### Cell 6 — 🔑 Login WandB

```python
import wandb

# Cách 1: Login tương tác
wandb.login()

# Cách 2: Dùng Kaggle Secret (khuyến nghị)
# Settings → Secrets → Add "WANDB_API_KEY" → Enable
# from kaggle_secrets import UserSecretsClient
# wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
```

---

### Cell 7 — 🔎 Tìm Checkpoint (Resume)

```python
import glob

CKPT_DIR = '/kaggle/working/ckpt/trained'
ckpts = sorted(glob.glob(f'{CKPT_DIR}/*.ckpt'))

# Nếu có checkpoint trên Kaggle Dataset, thêm ở đây:
# ckpts += sorted(glob.glob('/kaggle/input/artdapter-checkpoints/*.ckpt'))

normal_ckpts = [c for c in ckpts if 'EXCEPTION' not in c]
exception_ckpts = [c for c in ckpts if 'EXCEPTION' in c]

if normal_ckpts:
    RESUME_CKPT = normal_ckpts[-1]
    print(f'📌 Checkpoint: {RESUME_CKPT}')
    print(f'   ({len(normal_ckpts)} normal, {len(exception_ckpts)} exception)')
elif exception_ckpts:
    RESUME_CKPT = exception_ckpts[-1]
    print(f'⚠️ Chỉ có exception checkpoint: {RESUME_CKPT}')
else:
    RESUME_CKPT = None
    print('🆕 Không có checkpoint — train từ đầu')
```

---

### Cell 8 — 🚀 Training

```python
if RESUME_CKPT:
    print(f'▶️ Resuming from: {RESUME_CKPT}')
    !python train.py \
        --config_filepath configs/train_config_kaggle.yaml \
        --gpus 0 \
        --resume_from "{RESUME_CKPT}"
else:
    print('▶️ Starting training from scratch...')
    !python train.py \
        --config_filepath configs/train_config_kaggle.yaml \
        --gpus 0
```

#### Ước tính thời gian

| Metric | Giá trị |
|---|---|
| Tốc độ ước tính | ~1.5-2.0 s/step |
| Tổng steps | 5,000 (survey reproduction) |
| Thời gian ước tính | **~2-3 giờ** (1 session) |

---

### Cell 9 — 📊 Kiểm tra Checkpoint & Disk

```python
import os

# Checkpoints
ckpt_dir = '/kaggle/working/ckpt/trained'
if os.path.exists(ckpt_dir):
    !ls -lh {ckpt_dir}
else:
    print('❌ Chưa có checkpoint')

# Disk usage
print('\n📊 Disk usage:')
!df -h / | head -2
!echo "Output: $(du -sh /kaggle/working/ 2>/dev/null | cut -f1)"
!echo "Tmp: $(du -sh /tmp/ 2>/dev/null | cut -f1)"
```

---

## 4. Các Cell cho Inference

> Chạy trong **cùng notebook** (sau khi train) hoặc **notebook riêng**.

---

### Cell Inference 1 — Setup (nếu notebook mới)

> Bỏ qua nếu đang cùng notebook đã train.

```python
import os

# Setup cache (PHẢI chạy đầu tiên)
os.environ['HF_HOME'] = '/tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf_cache/transformers'
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_cache/datasets'

GITHUB_REPO = "https://github.com/paingoat/ArtDapter-reproduce.git"
if not os.path.exists('/kaggle/working/ArtDapter'):
    !git clone {GITHUB_REPO} /kaggle/working/ArtDapter
%cd /kaggle/working/ArtDapter

!pip install -q pytorch-lightning lightning transformers diffusers \
    omegaconf einops wandb datasets safetensors open-clip-torch tqdm Pillow
```

---

### Cell Inference 2 — Load Model & Generate

```python
import torch
import einops
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from models import load_state_dict
from ldm.models.diffusion.custom_ddim import CustomDDIMSampler

# ====== CẤU HÌNH ======
CHECKPOINT_PATH = "/kaggle/working/ckpt/trained/YOUR_CHECKPOINT.ckpt"  # ← Đổi tên
SEED = 42
NUM_SAMPLES = 4
DDIM_STEPS = 50
CFG_SCALE = 7.5
RESOLUTION = 512
# =======================

seed_everything(SEED)
config = OmegaConf.load("configs/inference_config_kaggle.yaml")

print(f'📥 Loading model...')
model = instantiate_from_config(config.model)
model.load_state_dict(load_state_dict(CHECKPOINT_PATH, location='cpu'))
model = model.cuda().eval()
print('✅ Model loaded')

ddim_sampler = CustomDDIMSampler(model)
```

---

### Cell Inference 3 — Nhập Prompt & Generate

```python
# ====== NHẬP PROMPT ======
prompt = "A serene landscape with mountains reflected in a calm lake"
art_style = "Impressionism"
PoA = [
    "Asymmetric balance with the mountain on the left balanced by open sky on the right.",
    "Cool blues and greens create a harmonious color palette.",
    "Variety in brush strokes from smooth water to textured mountains.",
    "", "", "", "", 
    "Gentle movement suggested by ripples in the water.",
    "", ""
]
# =========================

caption = model.apply_prompt_template(
    [prompt] * NUM_SAMPLES, [art_style] * NUM_SAMPLES, [PoA] * NUM_SAMPLES
)

with torch.no_grad():
    cond = dict(c_crossattn=[model.get_learned_conditioning(caption)])
    un_cond = dict(c_crossattn=[model.get_unconditional_conditioning(NUM_SAMPLES)])

    print(f'🎨 Generating {NUM_SAMPLES} images...')
    z_samples, _ = ddim_sampler.sample(
        conditioning=cond, S=DDIM_STEPS, batch_size=NUM_SAMPLES,
        shape=(4, RESOLUTION // 8, RESOLUTION // 8), verbose=False,
        eta=0.0, unconditional_guidance_scale=CFG_SCALE,
        unconditional_conditioning=un_cond,
    )
    x_samples = model.decode_first_stage(z_samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    print('✅ Done!')
```

---

### Cell Inference 4 — Hiển Thị Kết Quả

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

fig, axes = plt.subplots(1, NUM_SAMPLES, figsize=(5 * NUM_SAMPLES, 5))
if NUM_SAMPLES == 1: axes = [axes]

for i, ax in enumerate(axes):
    ax.imshow(x_samples[i])
    ax.set_title(f'Sample {i+1}')
    ax.axis('off')

plt.suptitle(f'"{prompt}" | {art_style}', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('/kaggle/working/inference_output.png', dpi=150, bbox_inches='tight')
plt.show()

# Lưu ảnh riêng
output_dir = '/kaggle/working/generated_images'
os.makedirs(output_dir, exist_ok=True)
for i, s in enumerate(x_samples):
    Image.fromarray((s * 255).astype(np.uint8)).save(f'{output_dir}/sample_{i+1}.png')
print(f'💾 Saved {NUM_SAMPLES} images to {output_dir}/')
```

---

## 5. Cheat Sheet: Session Mới

### Workflow

```
Lần đầu tiên:
  Cell 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

Các lần sau (resume):
  Cell 0 → 1 → 2 → 3 → (bỏ 4, 5 nếu đã upload init.ckpt lên Dataset) → 6 → 7 → 8

Inference:
  Inference Cell 1 → 2 → 3 → 4
```

| Lần đầu | Các lần sau |
|---|---|
| Cell 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 | Cell 0 → 1 → 2 → 3 → **(bỏ 4, 5)** → 6 → 7 → 8 |

> ⚠️ **Cell 0 LUÔN PHẢI chạy đầu tiên** mỗi session để setup HF cache.

### Cách giữ checkpoint giữa các sessions

1. **Quick Save**: Click "Save Version" → chọn "Quick Save"
2. **Kaggle Dataset** (khuyến nghị):
   - Save Version → vào output → "New Dataset" → dùng cho session sau

---

## 6. FAQ & Troubleshooting

### Q: Lỗi "No space left on device" / Disk tràn?
**A**: Chạy lệnh dọn dẹp:
```python
!rm -rf ~/.cache/huggingface
!pip cache purge
!rm -rf /kaggle/working/ArtDapter/ckpt/init/v1-5-pruned.ckpt
!rm -rf /kaggle/working/ArtDapter/ckpt/init/ella-sd1.5-tsc-t5xl.safetensors
```

### Q: Lỗi "CUDA out of memory"?
**A**: Giảm `batch_size` xuống 2 trong `train_config_kaggle.yaml`, tăng `accumulate_grad_batches` lên 12.

### Q: Training bị ngắt giữa chừng?
**A**: Checkpoint tự lưu mỗi 200 steps + exception checkpoint khi crash. Chạy lại từ Cell 0 → ... → Cell 7 (tìm checkpoint) → Cell 8.

### Q: Dataset CompArt không load được từ Kaggle Input?
**A**: Kiểm tra đã Add Data chưa (`"+ Add Data" → tìm "thoandanh/compart" → Add`). Config `train_config_kaggle.yaml` đã trỏ `dataset_path` thẳng đến `/kaggle/input/compart` — không cần symlink.

### Q: Tốc độ training bao nhiêu?
**A**: P100 ≈ 1.5-2.0 s/step. Với 5K steps (survey) ≈ **2-3 giờ**, dư sức trong 1 session 12h.
