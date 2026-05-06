# ArtDapter — Hướng dẫn setup inference từ đầu

Tài liệu này mô tả **chỉ luồng inference** (không train): clone repo, tạo môi trường, tải weights đã huấn luyện từ Hugging Face, chạy giao diện Streamlit.

**Nguồn weights khuyến nghị:** [paingoat/artdapter-v1](https://huggingface.co/paingoat/artdapter-v1/tree/main) — trong repo HF có thư mục `trained/` chứa file checkpoint `.ckpt` (ví dụ `genial-silence-15-epoch=34-step=124000.ckpt`; tên file có thể thay đổi theo phiên bản, hãy chọn file `.ckpt` mới nhất trên trang Files).

---

## 1. Yêu cầu hệ thống

| Hạng mục | Ghi chú |
|----------|---------|
| **GPU NVIDIA** | App dùng `pytorch_lightning.Trainer` với `accelerator='gpu'`. Cần CUDA và driver tương thích với PyTorch trong `environment.yaml` (CUDA 12.8 wheel). |
| **RAM / VRAM** | Checkpoint + T5/CLIP tải về có thể chiếm hàng chục GB ổ đĩa; VRAM tối thiểu thực tế thường từ ~8–12 GB trở lên tùy độ phân giải và batch. |
| **Python** | `environment.yaml` cố định **Python 3.11** (PyTorch 2.8). |
| **Hệ điều hành** | Linux / WSL2 khớp với các script `control/*.sh`. Windows: dùng **Anaconda Prompt** hoặc **PowerShell** cho conda + Streamlit; script bash có thể chạy trong Git Bash/WSL hoặc làm thủ công các bước tương đương bên dưới. |

---

## 2. Lấy mã nguồn

```bash
git clone <URL-repo-ArtDapter-của-bạn>
cd ArtDapter
```

---

## 3. Đường dẫn lưu dữ liệu mặc định: `/workspace/data`

Toàn bộ dữ liệu “nặng” (cache Hugging Face, checkpoint tải về, snapshot HF) nên gom về **`/workspace/data`** để dễ backup và tránh đầy ổ gốc repo (đặc biệt trên RunPod / container có volume `/workspace`).

```bash
export DATA_ROOT=/workspace/data
mkdir -p "$DATA_ROOT"/{hf_hub,hf_downloads,ckpt/trained}
```

| Vị trí | Mục đích |
|--------|----------|
| `$DATA_ROOT/hf_hub` | Cache model/tokenizer (đặt `HF_HOME` trỏ vào đây, xem mục 7). |
| `$DATA_ROOT/hf_downloads` | Thư mục `--local-dir` khi gọi `huggingface-cli download` (mục 5). |
| `$DATA_ROOT/ckpt/trained` | Lưu file `.ckpt` inference. |

**Liên kết tới repo** để Streamlit vẫn đọc được `ckpt/trained/*.ckpt` trong project (đường dẫn tương đối cố định trong app):

```bash
# Trong thư mục gốc ArtDapter, sau khi đã mkdir ckpt trong repo (hoặc chỉ cần thư mục cha)
rm -rf ckpt/trained 2>/dev/null || true
mkdir -p ckpt
ln -sfn "$DATA_ROOT/ckpt/trained" ckpt/trained
```

Trên máy **không** có `/workspace` (ví dụ Windows local), có thể đặt `DATA_ROOT` tới ổ dữ liệu của bạn (ví dụ `D:\ArtDapterData`) và giữ nguyên logic symlink / biến môi trường tương tự.

---

## 4. Môi trường Conda (khuyến nghị)

File gốc: [`environment.yaml`](../environment.yaml) — tạo env tên `artgen`:

```bash
conda env create -f environment.yaml --yes
conda activate artgen
```

Nếu conda báo lỗi ToS kênh mặc định, tham khảo [`control/setup_env.sh`](../control/setup_env.sh) (phần `conda tos accept`).

### Gói pip bổ sung cho inference

- **`openai`**: chế độ **CTF** gọi API phân rã prompt (`PromptDecomposer`). Nếu không cài, code fallback (chất lượng kém hơn).
- **`huggingface_hub`**: CLI `hf` (tải model) và thư viện Hub.
- **`python-dotenv`**: đọc file `.env` ở thư mục gốc repo khi chạy Streamlit (đã khai báo trong [`environment.yaml`](../environment.yaml)).

```bash
pip install "openai>=1.0.0" "huggingface_hub>=0.20.0" "python-dotenv>=1.0.0"
```

---

## 5. Tải weights từ Hugging Face

Repo model: **[paingoat/artdapter-v1](https://huggingface.co/paingoat/artdapter-v1/tree/main)**.

### Cách A — Hugging Face CLI (khuyến nghị)

Đăng nhập **không bắt buộc** nếu repo public; nếu cần token:

```bash
hf auth login
```

(`huggingface-cli` đã deprecated; dùng lệnh `hf`.)

Tải file checkpoint trong `trained/` về **`$DATA_ROOT/hf_downloads`** rồi copy vào **`$DATA_ROOT/ckpt/trained`** (đã symlink `ckpt/trained` trong repo như mục 3):

```bash
export DATA_ROOT=/workspace/data
mkdir -p "$DATA_ROOT/hf_downloads" "$DATA_ROOT/ckpt/trained"

# Ví dụ: tải snapshot (có thể ~15 GB — kiểm tra dung lượng trên trang HF)
hf download paingoat/artdapter-v1 \
  --include "trained/*.ckpt" \
  --local-dir "$DATA_ROOT/hf_downloads/hf_artdapter_v1"

cp "$DATA_ROOT/hf_downloads/hf_artdapter_v1/trained/"*.ckpt "$DATA_ROOT/ckpt/trained/"
```

Nếu có **nhiều** file `.ckpt`, giữ ít nhất một file; sidebar Streamlit liệt kê mọi file trong `ckpt/trained/*.ckpt`.

### Cách B — Tải tay trên trình duyệt

1. Mở [Files · paingoat/artdapter-v1](https://huggingface.co/paingoat/artdapter-v1/tree/main).
2. Vào thư mục `trained/`, tải file `.ckpt`.
3. Đặt file vào **`/workspace/data/ckpt/trained/`** (hoặc `ArtDapter/ckpt/trained/` nếu bạn không dùng symlink — cùng cấp với `configs/`, `inference/`, …).

---

## 6. Cấu hình YAML (tuỳ chọn nhưng nên đồng bộ)

Streamlit **nạp trọng số** từ đường dẫn bạn chọn trong sidebar (**Checkpoint**), không đọc `model.init_path` trong YAML để load file. Tuy nhiên `init_path` trong config vẫn nên trỏ tới checkpoint thật để tránh nhầm khi dùng tool khác.

- **Regular:** [`configs/inference_config.yaml`](../configs/inference_config.yaml) — `model.target: models.ArtDaptedModel`
- **CTF:** [`configs/ctf_inference_config.yaml`](../configs/ctf_inference_config.yaml) — `model.target: models.ctf_model.ArtDaptedModelCTF`

Sửa dòng:

```yaml
model:
  init_path: ./ckpt/trained/<tên-file.ckpt-của-bạn>
```

ở **cả hai** file nếu bạn dùng cả hai chế độ.

---

## 7. Biến môi trường

| Biến | Khi nào cần |
|------|-------------|
| `DATA_ROOT` | Quy ước gốc dữ liệu; mặc định tài liệu này dùng **`/workspace/data`** (mục 3). |
| `HF_HOME` | **Khuyến nghị:** `$DATA_ROOT/hf_hub` để cache T5/CLIP và model HF nằm trên volume dữ liệu. |
| `HF_DATASETS_CACHE` / `TRANSFORMERS_CACHE` | Tuỳ chọn: con của `HF_HOME` (ví dụ `$HF_HOME/datasets`, `$HF_HOME/transformers`) để tách thư mục. |
| `OPENAI_API_KEY` | Bật **Pipeline Mode → ctf** và muốn phân rã prompt qua GPT (không dùng fallback). |

Ví dụ Linux/macOS (mở shell trước khi tải weights và trước khi chạy Streamlit):

```bash
export DATA_ROOT=/workspace/data
export HF_HOME="$DATA_ROOT/hf_hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

export OPENAI_API_KEY="sk-..."
```

PowerShell (chỉnh ổ nếu không có `/workspace`):

```powershell
$env:DATA_ROOT = "/workspace/data"
$env:HF_HOME = "$env:DATA_ROOT/hf_hub"
$env:HF_DATASETS_CACHE = "$env:HF_HOME/datasets"
$env:TRANSFORMERS_CACHE = "$env:HF_HOME/transformers"
$env:OPENAI_API_KEY = "sk-..."
```

### File `.env` (thư mục gốc repo, cùng cấp với `configs/`)

Đặt file **`.env`** tại `ArtDapter/.env` (không phải trong `inference/`). Khi chạy `streamlit run inference/app.py`, app gọi `load_dotenv()` để nạp biến trước khi model dùng OpenAI.

Ví dụ (đường đầy đủ cho `HF_HOME` tránh lệ thuộc nội suy):

```env
DATA_ROOT=/workspace/data
HF_HOME=/workspace/data/hf_hub
OPENAI_API_KEY=sk-your-key-here
```

Nếu dùng `python-dotenv>=1.0` và muốn tham chiếu `DATA_ROOT` trong các dòng sau, có thể viết `HF_HOME=${DATA_ROOT}/hf_hub` tùy phiên bản; cách an toàn nhất là ghi đường dẫn tuyệt đối như trên.

**Cài gói:** `pip install "python-dotenv>=1.0.0"` (hoặc `conda env update` từ `environment.yaml` sau khi đã thêm dependency).

---

## 8. Chạy inference (Streamlit)

Luôn chạy từ **thư mục gốc repo**, với `PYTHONPATH` trỏ vào gốc project (để import `ldm`, `models`, `utils`).

### Linux / macOS / WSL

```bash
export DATA_ROOT=/workspace/data
export HF_HOME="$DATA_ROOT/hf_hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

cd /đường/dẫn/ArtDapter
conda activate artgen
export PYTHONPATH=.
streamlit run inference/app.py --server.port 8502 --server.address 0.0.0.0
```

Hoặc dùng script có sẵn (Linux, **cần checkpoint trong `ckpt/trained/`** — thường là symlink tới `$DATA_ROOT/ckpt/trained`; script copy `pod_inference_config` sang `inference_config` và sửa `init_path` — chủ yếu phục vụ RunPod):

```bash
bash control/inference.sh
# hoặc chỉ định file:
bash control/inference.sh --ckpt ckpt/trained/your.ckpt
```

Lưu ý: `control/inference.sh` dùng `sed -i` kiểu GNU; trên macOS có thể cần chỉnh hoặc cập nhật `init_path` thủ công như mục 6. Nếu dùng RunPod, có thể thêm các dòng `export HF_HOME=...` vào đầu phiên shell hoặc vào `~/.bashrc` cho khớp mục 3 và 7.

### Windows (PowerShell)

```powershell
cd E:\đường\dẫn\ArtDapter
conda activate artgen
$env:PYTHONPATH = "."
streamlit run inference/app.py --server.port 8502
```

Mở trình duyệt: **http://localhost:8502** (hoặc cổng bạn chỉ định).

---

## 9. Dùng giao diện

1. **Sidebar — Model Options:** chọn GPU (`Cuda device`), precision (thường `16-mixed`), và **Checkpoint** (danh sách từ `ckpt/trained/*.ckpt`).
2. **Pipeline Mode:**
   - **regular:** một luồng điều kiện như `ArtDaptedModel` — **phù hợp checkpoint public trên HF** nếu checkpoint được train với kiến trúc chuẩn trong `inference_config.yaml`.
   - **ctf:** `ArtDaptedModelCTF` + sampler CTF — cần checkpoint **tương thích** kiến trúc CTF (`CTFUNetModel`, v.v.). Nếu checkpoint chỉ khớp UNet/ArtDapter bản regular, có thể báo missing/unexpected keys hoặc hành vi sai; khi đó dùng **regular** hoặc dùng checkpoint đã train cho CTF.
3. Nhập **Prompt**, **Art Style**, các trường **Principles of Art**, chỉnh bước lấy mẫu / CFG / seed, bấm **GENERATE**.

Lần đầu chạy, **T5** và **CLIP** (và mô hình liên quan) có thể tự tải từ Hugging Face Hub — cần mạng và đủ dung lượng cache.

---

## 10. Xử lý sự cố thường gặp

| Triệu chứng | Hướng xử lý |
|-------------|-------------|
| `CUDA not available` / Trainer không lên GPU | Cài PyTorch build có CUDA đúng driver; kiểm tra `nvidia-smi`. |
| Sidebar không có checkpoint | Đảm bảo ít nhất một file `*.ckpt` nằm trong `ckpt/trained/`. |
| CTF báo lỗi API / phân rã kém | Cài `openai`, đặt `OPENAI_API_KEY`, hoặc tạm dùng **regular**. |
| Missing / Unexpected keys khi load | Checkpoint không khớp `model.target` đang chọn (regular vs **ctf**); đổi Pipeline Mode hoặc dùng đúng checkpoint. |
| Lỗi font (chủ yếu khi train/log) | Một số util dùng `font/DejaVuSans.ttf`; inference UI thường không cần. Có thể tạo `font/DejaVuSans.ttf` hoặc symlink như `control/setup_env.sh`. |

---

## 11. Bản đồ file liên quan (tham chiếu nhanh)

| Thành phần | Đường dẫn |
|-------------|-----------|
| Gốc dữ liệu (mặc định) | `/workspace/data` — con: `hf_hub/`, `hf_downloads/`, `ckpt/trained/` |
| App Streamlit | [`inference/app.py`](../inference/app.py) |
| Config regular | [`configs/inference_config.yaml`](../configs/inference_config.yaml) |
| Config CTF | [`configs/ctf_inference_config.yaml`](../configs/ctf_inference_config.yaml) |
| Checkpoint inference | `ckpt/trained/*.ckpt` trong repo (thường symlink → `/workspace/data/ckpt/trained/`) |
| Môi trường conda | [`environment.yaml`](../environment.yaml) |
| Chạy nhanh (Linux) | [`run_inference_app.sh`](../run_inference_app.sh), [`control/inference.sh`](../control/inference.sh) |

Tài liệu môi trường khác (Pod RTX 5090, Kaggle) vẫn có tại [`pod_guide.md`](pod_guide.md) và [`kaggle_guide.md`](kaggle_guide.md); tài liệu này tập trung **inference + weights HF** cho mọi máy có GPU phù hợp.
