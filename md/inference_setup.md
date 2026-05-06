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

## 3. Môi trường Conda (khuyến nghị)

File gốc: [`environment.yaml`](../environment.yaml) — tạo env tên `artgen`:

```bash
conda env create -f environment.yaml --yes
conda activate artgen
```

Nếu conda báo lỗi ToS kênh mặc định, tham khảo [`control/setup_env.sh`](../control/setup_env.sh) (phần `conda tos accept`).

### Gói pip bổ sung cho inference

- **`openai`**: chế độ **CTF** gọi API phân rã prompt (`PromptDecomposer`). Nếu không cài, code fallback (chất lượng kém hơn).
- **`huggingface_hub`**: tiện tải checkpoint từ Hugging Face (CLI `huggingface-cli`).

```bash
pip install "openai>=1.0.0" "huggingface_hub>=0.20.0"
```

---

## 4. Tải weights từ Hugging Face

Repo model: **[paingoat/artdapter-v1](https://huggingface.co/paingoat/artdapter-v1/tree/main)**.

### Cách A — Hugging Face CLI (khuyến nghị)

Đăng nhập **không bắt buộc** nếu repo public; nếu cần token:

```bash
huggingface-cli login
```

Tải file checkpoint trong `trained/` về thư mục tạm rồi copy vào repo:

```bash
# Ví dụ: tải cả snapshot (có thể ~15 GB — kiểm tra dung lượng trên trang HF)
huggingface-cli download paingoat/artdapter-v1 \
  --include "trained/*.ckpt" \
  --local-dir ./hf_artdapter_v1

# Đặt checkpoint vào đúng chỗ app đọc
mkdir -p ckpt/trained
cp hf_artdapter_v1/trained/*.ckpt ckpt/trained/
```

Nếu có **nhiều** file `.ckpt`, giữ ít nhất một file; sidebar Streamlit liệt kê mọi file trong `ckpt/trained/*.ckpt`.

### Cách B — Tải tay trên trình duyệt

1. Mở [Files · paingoat/artdapter-v1](https://huggingface.co/paingoat/artdapter-v1/tree/main).
2. Vào thư mục `trained/`, tải file `.ckpt`.
3. Đặt file vào `ArtDapter/ckpt/trained/` (cùng cấp với thư mục `configs/`, `inference/`, …).

---

## 5. Cấu hình YAML (tuỳ chọn nhưng nên đồng bộ)

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

## 6. Biến môi trường

| Biến | Khi nào cần |
|------|-------------|
| `OPENAI_API_KEY` | Bật **Pipeline Mode → ctf** và muốn phân rã prompt qua GPT (không dùng fallback). |
| `HF_HOME` / `TRANSFORMERS_CACHE` | Tuỳ chọn: chỉ định cache Hugging Face nếu ổ hệ thống nhỏ. |

Ví dụ Linux/macOS:

```bash
export OPENAI_API_KEY="sk-..."
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

---

## 7. Chạy inference (Streamlit)

Luôn chạy từ **thư mục gốc repo**, với `PYTHONPATH` trỏ vào gốc project (để import `ldm`, `models`, `utils`).

### Linux / macOS / WSL

```bash
cd /đường/dẫn/ArtDapter
conda activate artgen
export PYTHONPATH=.
streamlit run inference/app.py --server.port 8502 --server.address 0.0.0.0
```

Hoặc dùng script có sẵn (Linux, **cần checkpoint trong `ckpt/trained/`**; script copy `pod_inference_config` sang `inference_config` và sửa `init_path` — chủ yếu phục vụ RunPod):

```bash
bash control/inference.sh
# hoặc chỉ định file:
bash control/inference.sh --ckpt ckpt/trained/your.ckpt
```

Lưu ý: `control/inference.sh` dùng `sed -i` kiểu GNU; trên macOS có thể cần chỉnh hoặc cập nhật `init_path` thủ công như mục 5.

### Windows (PowerShell)

```powershell
cd E:\đường\dẫn\ArtDapter
conda activate artgen
$env:PYTHONPATH = "."
streamlit run inference/app.py --server.port 8502
```

Mở trình duyệt: **http://localhost:8502** (hoặc cổng bạn chỉ định).

---

## 8. Dùng giao diện

1. **Sidebar — Model Options:** chọn GPU (`Cuda device`), precision (thường `16-mixed`), và **Checkpoint** (danh sách từ `ckpt/trained/*.ckpt`).
2. **Pipeline Mode:**
   - **regular:** một luồng điều kiện như `ArtDaptedModel` — **phù hợp checkpoint public trên HF** nếu checkpoint được train với kiến trúc chuẩn trong `inference_config.yaml`.
   - **ctf:** `ArtDaptedModelCTF` + sampler CTF — cần checkpoint **tương thích** kiến trúc CTF (`CTFUNetModel`, v.v.). Nếu checkpoint chỉ khớp UNet/ArtDapter bản regular, có thể báo missing/unexpected keys hoặc hành vi sai; khi đó dùng **regular** hoặc dùng checkpoint đã train cho CTF.
3. Nhập **Prompt**, **Art Style**, các trường **Principles of Art**, chỉnh bước lấy mẫu / CFG / seed, bấm **GENERATE**.

Lần đầu chạy, **T5** và **CLIP** (và mô hình liên quan) có thể tự tải từ Hugging Face Hub — cần mạng và đủ dung lượng cache.

---

## 9. Xử lý sự cố thường gặp

| Triệu chứng | Hướng xử lý |
|-------------|-------------|
| `CUDA not available` / Trainer không lên GPU | Cài PyTorch build có CUDA đúng driver; kiểm tra `nvidia-smi`. |
| Sidebar không có checkpoint | Đảm bảo ít nhất một file `*.ckpt` nằm trong `ckpt/trained/`. |
| CTF báo lỗi API / phân rã kém | Cài `openai`, đặt `OPENAI_API_KEY`, hoặc tạm dùng **regular**. |
| Missing / Unexpected keys khi load | Checkpoint không khớp `model.target` đang chọn (regular vs **ctf**); đổi Pipeline Mode hoặc dùng đúng checkpoint. |
| Lỗi font (chủ yếu khi train/log) | Một số util dùng `font/DejaVuSans.ttf`; inference UI thường không cần. Có thể tạo `font/DejaVuSans.ttf` hoặc symlink như `control/setup_env.sh`. |

---

## 10. Bản đồ file liên quan (tham chiếu nhanh)

| Thành phần | Đường dẫn |
|-------------|-----------|
| App Streamlit | [`inference/app.py`](../inference/app.py) |
| Config regular | [`configs/inference_config.yaml`](../configs/inference_config.yaml) |
| Config CTF | [`configs/ctf_inference_config.yaml`](../configs/ctf_inference_config.yaml) |
| Checkpoint inference | `ckpt/trained/*.ckpt` |
| Môi trường conda | [`environment.yaml`](../environment.yaml) |
| Chạy nhanh (Linux) | [`run_inference_app.sh`](../run_inference_app.sh), [`control/inference.sh`](../control/inference.sh) |

Tài liệu môi trường khác (Pod RTX 5090, Kaggle) vẫn có tại [`pod_guide.md`](pod_guide.md) và [`kaggle_guide.md`](kaggle_guide.md); tài liệu này tập trung **inference + weights HF** cho mọi máy có GPU phù hợp.
