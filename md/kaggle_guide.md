# 🎨 Hướng Dẫn Train & Inference ArtDapter trên Kaggle (GPU P100)

> **GPU**: NVIDIA Tesla P100 — 16GB HBM2 VRAM  
> **Repo**: `https://github.com/YOUR_USERNAME/ArtDapter.git` ← Thay bằng repo của bạn  
> **Config**: `configs/train_config_kaggle.yaml`

---

## 📋 Mục Lục

1. [Tổng quan & Lưu ý quan trọng](#1-tổng-quan--lưu-ý-quan-trọng)
2. [Tạo Kaggle Notebook](#2-tạo-kaggle-notebook)
3. [Các cell cho Training](#3-các-cell-cho-training)
4. [Các cell cho Inference](#4-các-cell-cho-inference)
5. [Cheat Sheet: Session mới](#5-cheat-sheet-session-mới)
6. [FAQ & Troubleshooting](#6-faq--troubleshooting)

---

## 1. Tổng quan & Lưu ý quan trọng

### Specs GPU P100 trên Kaggle

| Thông số | Giá trị |
|---|---|
| GPU | NVIDIA Tesla P100 |
| VRAM | 16GB HBM2 |
| Memory Bandwidth | 732 GB/s |
| Compute Capability | 6.0 |
| FP16 | ✅ Hỗ trợ (18.7 TFLOPS) |
| BF16 | ❌ **Không hỗ trợ** |
| Precision dùng | `16-mixed` (FP16) |

### Cấu hình training đã tối ưu

| Setting | Giá trị | Lý do |
|---|---|---|
| `batch_size` | 4 | P100 16GB VRAM |
| `accumulate_grad_batches` | 6 | Effective batch = 4 × 6 = 24 ≈ 22 gốc |
| `precision` | `16-mixed` | P100 chỉ hỗ trợ FP16, **không** bf16 |
| `use_checkpoint` | True | Gradient checkpointing tiết kiệm VRAM |
| `log_frequency` | 200 | Checkpoint thường xuyên (phòng session ngắt) |
| `num_workers` | 2 | Kaggle ít CPU cores |

### Giới hạn Kaggle

- **GPU quota**: 30 giờ/tuần (P100)
- **Session tối đa**: ~12 giờ liên tục
- **Output storage**: 20GB tại `/kaggle/working/` — **file ở đây được giữ lại** khi Save Version
- **Kaggle Datasets**: Dùng để lưu checkpoint lâu dài giữa các notebooks

> ⚠️ **QUAN TRỌNG**: Khác với Google Colab (dùng Google Drive), Kaggle lưu output tại `/kaggle/working/`. Để giữ checkpoint qua nhiều sessions, bạn cần **Save Version** hoặc upload checkpoint lên **Kaggle Dataset**.

---

## 2. Tạo Kaggle Notebook

### Bước 1: Tạo Notebook mới

1. Vào [kaggle.com](https://www.kaggle.com) → đăng nhập
2. Click **"+ Create"** → **"New Notebook"**
3. Đổi tên notebook: `ArtDapter-Training-P100`

### Bước 2: Bật GPU P100

1. Click **⚙️ Settings** (góc phải) hoặc **Session Options**
2. **Accelerator**: chọn **GPU P100**
3. **Persistence**: bật **ON** (giữ output qua sessions)
4. **Internet**: bật **ON** (cần để tải weights và clone repo)

### Bước 3: Thêm Kaggle Dataset (cho resume training lần sau)

> Lần đầu tiên bỏ qua bước này. Sau khi train xong session đầu, Save Version rồi tạo Dataset từ output.

1. Vào **Kaggle** → **Datasets** → **New Dataset**
2. Upload checkpoint files từ output của notebook trước
3. Quay lại notebook → **Add Data** → chọn Dataset vừa tạo
4. Dataset sẽ ở `/kaggle/input/your-dataset-name/`

---

## 3. Các Cell cho Training

Copy-paste từng cell bên dưới vào notebook Kaggle theo đúng thứ tự.

---

### Cell 1 — 🔍 Kiểm tra GPU

```python
!nvidia-smi
import torch
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f"\n✅ GPU: {gpu_name}")
print(f"✅ VRAM: {vram_gb:.1f} GB")
print(f"✅ CUDA: {torch.version.cuda}")
print(f"✅ PyTorch: {torch.__version__}")

if "P100" in gpu_name:
    print("✅ Đúng GPU P100 — config đã tối ưu cho GPU này")
elif "T4" in gpu_name:
    print("⚠️ Đang dùng T4, không phải P100 — vẫn chạy được nhưng nên chọn P100 trong Settings")
else:
    print(f"⚠️ GPU không phải P100: {gpu_name}")
```

**Mục đích**: Xác nhận GPU P100 được cấp phát. Nếu thấy T4 hoặc GPU khác, vào Settings đổi lại.

---

### Cell 2 — 📥 Clone Repo & Setup

```python
import os

# === SỬA URL GITHUB CỦA BẠN Ở ĐÂY ===
GITHUB_REPO = "https://github.com/YOUR_USERNAME/ArtDapter.git"

# Clone repo
if not os.path.exists('/kaggle/working/ArtDapter'):
    !git clone {GITHUB_REPO} /kaggle/working/ArtDapter

%cd /kaggle/working/ArtDapter

# Tạo thư mục checkpoint
!mkdir -p /kaggle/working/ckpt/trained
!mkdir -p /kaggle/working/ArtDapter/ckpt/init
```

**Mục đích**: Clone source code từ GitHub vào Kaggle. Nếu notebook đã có repo (từ session trước), sẽ bỏ qua.

---

### Cell 3 — 📦 Cài Dependencies

```python
!pip install -q pytorch-lightning lightning transformers diffusers \
    omegaconf einops wandb datasets safetensors open-clip-torch tqdm Pillow
```

**Mục đích**: Cài tất cả thư viện cần thiết. PyTorch đã có sẵn trên Kaggle.

---

### Cell 4 — ⬇️ Tải Pre-trained Weights

> **Lần đầu**: chạy cell này để tải (~4.2GB). Các session sau: nếu đã upload weights lên Kaggle Dataset thì bỏ qua cell này.

```python
import os

LOCAL_INIT = '/kaggle/working/ArtDapter/ckpt/init'

# --- Stable Diffusion v1.5 weights (~4GB) ---
sd_path = f'{LOCAL_INIT}/v1-5-pruned.ckpt'
if not os.path.exists(sd_path):
    print('📥 Downloading Stable Diffusion v1.5 weights (~4GB)...')
    !wget -q --show-progress -O {sd_path} \
        https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
else:
    print('✅ SD v1.5 weights đã có')

# --- ELLA weights (~200MB) ---
ella_path = f'{LOCAL_INIT}/ella-sd1.5-tsc-t5xl.safetensors'
if not os.path.exists(ella_path):
    print('📥 Downloading ELLA weights (~200MB)...')
    !wget -q --show-progress -O {ella_path} \
        https://huggingface.co/QQGYLab/ELLA/resolve/main/ella-sd1.5-tsc-t5xl.safetensors
else:
    print('✅ ELLA weights đã có')

print('\n✅ Tất cả pre-trained weights đã sẵn sàng')
```

**Mục đích**: Tải Stable Diffusion v1.5 và ELLA weights. Nếu đã có (do session trước chưa bị xóa) thì bỏ qua.

> 💡 **Mẹo tiết kiệm thời gian**: Sau lần tải đầu, upload folder `ckpt/init/` lên **Kaggle Dataset**. Các session sau chỉ cần symlink:
> ```python
> !ln -sf /kaggle/input/artdapter-weights/v1-5-pruned.ckpt /kaggle/working/ArtDapter/ckpt/init/v1-5-pruned.ckpt
> !ln -sf /kaggle/input/artdapter-weights/ella-sd1.5-tsc-t5xl.safetensors /kaggle/working/ArtDapter/ckpt/init/ella-sd1.5-tsc-t5xl.safetensors
> ```

---

### Cell 5 — 🔧 Prepare Initial Weights

> Chạy **1 lần duy nhất**. Merge SD v1.5 pre-trained weights + random ArtDapter weights → `init.ckpt`.

```python
import os

init_ckpt = '/kaggle/working/ArtDapter/ckpt/init/init.ckpt'

if not os.path.exists(init_ckpt):
    print('🔧 Preparing initial weights (merge SD + random ArtDapter)...')
    !python prepare_weights.py \
        --init_dir /kaggle/working/ArtDapter/ckpt/init \
        --output init.ckpt \
        --config configs/train_config_kaggle.yaml
    print('✅ init.ckpt đã tạo xong')
else:
    print('✅ init.ckpt đã tồn tại')
```

**Mục đích**: Tạo checkpoint khởi tạo gồm SD backbone (pre-trained) + ArtDapter module (random weights).

> 💡 **Lưu lại init.ckpt**: Upload `init.ckpt` lên Kaggle Dataset để không cần tạo lại.

---

### Cell 6 — 🔑 Login WandB (Tùy chọn)

```python
import wandb

# Cách 1: Login tương tác (paste API key)
wandb.login()

# Cách 2: Dùng Kaggle Secret (khuyến nghị)
# 1. Vào Settings → Add-ons → Secrets
# 2. Thêm secret tên "WANDB_API_KEY" với giá trị là API key từ wandb.ai/authorize
# 3. Bật "Attach to this notebook"
# Sau đó dùng:
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# wandb_key = user_secrets.get_secret("WANDB_API_KEY")
# wandb.login(key=wandb_key)
```

**Mục đích**: Xác thực WandB để log training metrics. Nếu không dùng WandB, cần sửa code `train.py` để bỏ WandB logger.

---

### Cell 7 — 🔎 Tìm Checkpoint (cho Resume Training)

```python
import glob

# Tìm trong output của notebook hiện tại
CKPT_DIR = '/kaggle/working/ckpt/trained'
ckpts = sorted(glob.glob(f'{CKPT_DIR}/*.ckpt'))

# Cũng tìm trong Kaggle Dataset (nếu đã upload checkpoint từ session trước)
# Uncomment dòng dưới và đổi tên dataset cho đúng:
# DATASET_CKPT_DIR = '/kaggle/input/artdapter-checkpoints'
# ckpts += sorted(glob.glob(f'{DATASET_CKPT_DIR}/*.ckpt'))

# Phân loại checkpoints
normal_ckpts = [c for c in ckpts if 'EXCEPTION' not in c]
exception_ckpts = [c for c in ckpts if 'EXCEPTION' in c]

if normal_ckpts:
    RESUME_CKPT = normal_ckpts[-1]
    print(f'📌 Checkpoint tìm thấy: {RESUME_CKPT}')
    print(f'   Tổng: {len(normal_ckpts)} normal, {len(exception_ckpts)} exception')
elif exception_ckpts:
    RESUME_CKPT = exception_ckpts[-1]
    print(f'⚠️ Chỉ có exception checkpoint: {RESUME_CKPT}')
else:
    RESUME_CKPT = None
    print('🆕 Không có checkpoint — sẽ train từ đầu')
```

**Mục đích**: Auto-detect checkpoint mới nhất. Nếu đã upload checkpoint lên Kaggle Dataset, uncomment phần `DATASET_CKPT_DIR`.

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

**Mục đích**: Chạy training loop. Checkpoint tự động lưu tại `/kaggle/working/ckpt/trained/` mỗi 200 steps.

### Ước tính thời gian training trên P100

| Metric | Giá trị |
|---|---|
| Effective batch size | 24 (4 × 6) |
| Tốc độ ước tính | ~1.5-2.0 s/step |
| Steps/session (12h) | ~21,600-28,800 steps |
| Tổng steps cần | 280,000 |
| Tổng sessions ≈ | **10-13 sessions** |
| Tổng giờ GPU ≈ | **~120-160 giờ** (~4-5 tuần Kaggle quota) |

---

### Cell 9 — 📊 Kiểm tra Checkpoint

```python
import os

ckpt_dir = '/kaggle/working/ckpt/trained'
if os.path.exists(ckpt_dir):
    !ls -lh {ckpt_dir}
else:
    print('❌ Chưa có checkpoint nào')
```

**Mục đích**: Xem danh sách checkpoints đã lưu.

---

### Cell 10 — 💾 Lưu Checkpoint cho Session Sau

> **QUAN TRỌNG**: Trước khi session kết thúc, chạy cell này để đảm bảo checkpoint được lưu.

```python
# Cách 1: Save Version (đơn giản nhất)
# Click "Save Version" (góc phải trên) → chọn "Quick Save"
# Output folder sẽ được giữ lại

# Cách 2: Copy checkpoint vào output (nếu cần)
import shutil
import glob

ckpt_dir = '/kaggle/working/ckpt/trained'
ckpts = sorted(glob.glob(f'{ckpt_dir}/*.ckpt'))

if ckpts:
    latest = ckpts[-1]
    print(f'📌 Latest checkpoint: {latest}')
    print(f'📏 Size: {os.path.getsize(latest) / 1024**3:.2f} GB')
    print('\n💡 Để giữ checkpoint cho session sau:')
    print('   1. Click "Save Version" → "Quick Save"')
    print('   2. Hoặc tạo Kaggle Dataset từ output để dùng lâu dài')
else:
    print('❌ Chưa có checkpoint')
```

---

## 4. Các Cell cho Inference

> Inference có thể chạy trong **cùng notebook** (sau khi train xong) hoặc **notebook riêng**.

---

### Cell Inference 1 — Setup (nếu notebook mới)

> Bỏ qua cell này nếu đang ở cùng notebook đã train.

```python
import os

# Clone repo (nếu chưa có)
GITHUB_REPO = "https://github.com/YOUR_USERNAME/ArtDapter.git"
if not os.path.exists('/kaggle/working/ArtDapter'):
    !git clone {GITHUB_REPO} /kaggle/working/ArtDapter

%cd /kaggle/working/ArtDapter

# Cài dependencies
!pip install -q pytorch-lightning lightning transformers diffusers \
    omegaconf einops wandb datasets safetensors open-clip-torch tqdm Pillow
```

---

### Cell Inference 2 — Load Model & Generate

```python
import torch
import einops
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything

from dataset import CompArt
from ldm.util import instantiate_from_config
from models import load_state_dict
from ldm.models.diffusion.custom_ddim import CustomDDIMSampler

# ====== CẤU HÌNH ======
CHECKPOINT_PATH = "/kaggle/working/ckpt/trained/YOUR_CHECKPOINT.ckpt"  # ← Đổi tên checkpoint
CONFIG_PATH = "configs/inference_config_kaggle.yaml"
SEED = 42
NUM_SAMPLES = 4          # Số ảnh generate
DDIM_STEPS = 50           # Số bước diffusion (nhiều hơn = chi tiết hơn, chậm hơn)
CFG_SCALE = 7.5           # Guidance scale (cao hơn = bám prompt hơn)
RESOLUTION = 512
# =======================

seed_everything(SEED)
config = OmegaConf.load(CONFIG_PATH)

# Load model
print(f'📥 Loading model from {CHECKPOINT_PATH}...')
model = instantiate_from_config(config.model)
state_dict = load_state_dict(CHECKPOINT_PATH, location='cpu')
model.load_state_dict(state_dict)
model = model.cuda().eval()
print('✅ Model loaded')

# Tạo DDIM sampler
ddim_sampler = CustomDDIMSampler(model)
```

---

### Cell Inference 3 — Nhập Prompt & Generate

```python
# ====== NHẬP PROMPT Ở ĐÂY ======
prompt = "A serene landscape with mountains reflected in a calm lake"
art_style = "Impressionism"

# Principles of Art (để trống "" nếu không muốn chỉ định)
PoA = [
    "Asymmetric balance with the mountain on the left balanced by open sky on the right.",  # Balance
    "Cool blues and greens create a harmonious color palette.",                              # Harmony
    "Variety in brush strokes from smooth water to textured mountains.",                     # Variety
    "",  # Unity
    "Strong contrast between dark mountains and bright sky.",                                # Contrast
    "",  # Emphasis
    "",  # Proportion
    "Gentle movement suggested by ripples in the water.",                                    # Movement
    "",  # Rhythm
    ""   # Pattern
]
# =================================

# Tạo conditioning
caption = model.apply_prompt_template(
    [prompt] * NUM_SAMPLES,
    [art_style] * NUM_SAMPLES,
    [PoA] * NUM_SAMPLES
)

with torch.no_grad():
    cond = dict(c_crossattn=[model.get_learned_conditioning(caption)])
    un_cond = dict(c_crossattn=[model.get_unconditional_conditioning(NUM_SAMPLES)])

    print(f'🎨 Generating {NUM_SAMPLES} images...')
    z_samples, _ = ddim_sampler.sample(
        conditioning=cond,
        S=DDIM_STEPS,
        batch_size=NUM_SAMPLES,
        shape=(4, RESOLUTION // 8, RESOLUTION // 8),
        verbose=False,
        eta=0.0,
        unconditional_guidance_scale=CFG_SCALE,
        unconditional_conditioning=un_cond,
    )

    x_samples = model.decode_first_stage(z_samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 0.5 + 0.5).clamp(0, 1)
    x_samples = x_samples.cpu().numpy()
    print('✅ Done!')
```

---

### Cell Inference 4 — Hiển Thị Kết Quả

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, NUM_SAMPLES, figsize=(5 * NUM_SAMPLES, 5))
if NUM_SAMPLES == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    ax.imshow(x_samples[i])
    ax.set_title(f'Sample {i+1}')
    ax.axis('off')

plt.suptitle(f'Prompt: "{prompt}" | Style: {art_style}', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('/kaggle/working/inference_output.png', dpi=150, bbox_inches='tight')
plt.show()
print('💾 Saved to /kaggle/working/inference_output.png')
```

---

### Cell Inference 5 — Lưu Ảnh Riêng Lẻ

```python
from PIL import Image
import numpy as np

output_dir = '/kaggle/working/generated_images'
os.makedirs(output_dir, exist_ok=True)

for i, sample in enumerate(x_samples):
    img = Image.fromarray((sample * 255).astype(np.uint8))
    img_path = f'{output_dir}/sample_{i+1}.png'
    img.save(img_path)
    print(f'💾 Saved: {img_path}')

print(f'\n✅ {NUM_SAMPLES} images saved to {output_dir}/')
```

---

## 5. Cheat Sheet: Session Mới

### Workflow tổng quan

```
Session 1 (lần đầu):
  Cell 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
  (Kiểm tra GPU → Clone → Cài lib → Tải weights → Prepare → WandB → Tìm ckpt → Train)

Session 2+ (resume):
  Cell 1 → 2 → 3 → (bỏ 4,5) → 6 → 7 → 8
  (Nếu đã upload weights/init.ckpt lên Kaggle Dataset)

Inference (sau khi train xong):
  Inference Cell 1 (nếu notebook mới) → 2 → 3 → 4 → 5
```

### Bảng tóm tắt

| Lần đầu | Các lần sau (resume) |
|---|---|
| Cell 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 | Cell 1 → 2 → 3 → **(bỏ 4, 5)** → 6 → 7 → 8 |

> Cell 4 (tải weights) và Cell 5 (prepare init) chỉ cần chạy **lần đầu**.
> Nếu muốn bỏ qua các lần sau, upload `ckpt/init/` lên **Kaggle Dataset**.

### Cách giữ checkpoint giữa các sessions

1. **Quick Save**: Click "Save Version" → Output tại `/kaggle/working/` được lưu
2. **Kaggle Dataset** (khuyến nghị cho lâu dài):
   - Sau khi train → click "Save Version"
   - Vào version đó → **"New Dataset"** từ output
   - Session sau: Add Data → chọn Dataset → checkpoint ở `/kaggle/input/...`

---

## 6. FAQ & Troubleshooting

### Q: Lỗi "CUDA out of memory"?

**A**: Giảm `batch_size` trong `configs/train_config_kaggle.yaml`:
```yaml
dataloader:
  batch_size: 2  # giảm từ 4 xuống 2
```
Và tăng `accumulate_grad_batches` để giữ effective batch size:
```yaml
accumulate_grad_batches: 12  # 2 × 12 = 24
```

### Q: Kaggle không cho chọn GPU P100?

**A**: P100 có thể hết quota (30h/tuần). Chờ quota reset hàng tuần hoặc dùng T4 (cũng 16GB VRAM, config tương thích).

### Q: Lỗi "No module named ..."?

**A**: Chạy lại Cell 3 (cài dependencies). Nếu vẫn lỗi, thử:
```python
!pip install --force-reinstall pytorch-lightning lightning transformers
```

### Q: WandB không login được?

**A**: Dùng Kaggle Secrets:
1. **Settings** → **Secrets** → **Add Secret**
2. Name: `WANDB_API_KEY`, Value: lấy từ [wandb.ai/authorize](https://wandb.ai/authorize)
3. Enable cho notebook hiện tại

### Q: Training bị ngắt giữa chừng?

**A**: Checkpoint tự động lưu mỗi 200 steps. Chạy lại từ Cell 7 (tìm checkpoint) → Cell 8 (resume training).

### Q: Muốn dùng T4 thay P100?

**A**: Config `train_config_kaggle.yaml` tương thích với cả T4 (cùng 16GB VRAM, cùng hỗ trợ FP16). Chỉ cần đổi GPU type trong Settings.

### Q: Tốc độ training bao nhiêu?

**A**: P100 ước tính ~1.5-2.0 s/step (với batch=4, accumulate=6). Mỗi session 12h ≈ 21,000-28,000 steps. Tổng 280K steps cần khoảng 10-13 sessions.
