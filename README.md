## ArtDapter

ArtDapter là project fine-tune/generate ảnh theo prompt + Art Style + Principles of Art (PoA), xây trên Stable Diffusion + ELLA.

### Điểm rẽ nhanh (quan trọng)

- **RTX 5090 (RunPod hoặc local training Linux/WSL2)**: đi theo luồng **`md/pod`** (tài liệu hiện tại: `md/pod_guide.md`).
- **Notebook environment (Kaggle, Colab-style)**: đi theo luồng **`md/kaggle.md`** (tài liệu hiện tại: `md/kaggle_guide.md`).
- Không trộn config giữa hai luồng:
	- Pod/local RTX 5090: `configs/pod_train_config.yaml`, `configs/pod_inference_config.yaml`
	- Kaggle notebook: `configs/kaggle_train_config.yaml`, `configs/kaggle_inference_config.yaml`

---

## Tổng quan pipeline

1. **Chuẩn bị weights**: `prepare_weights.py` gộp SD v1.5 + ELLA thành `ckpt/init/init.ckpt`
2. **Tải dataset**: `download_dataset.sh` tải tập CompArt thông qua thư viện HF datasets.
3. **Train**: `train.py` (PyTorch Lightning + WandB)
4. **Inference UI**: `inference/app.py` (Streamlit)

Dataset chính: CompArt (qua HuggingFace dataset path trong config).

---

## Cấu trúc chính

- `control/`: script shell để setup/train/inference/sanity check
- `configs/`: config tách theo môi trường (`pod_*`, `kaggle_*`, default)
- `ckpt/init/`: raw pretrained + `init.ckpt`
- `ckpt/trained/`: checkpoint sau training
- `inference/`: Streamlit app + CSS
- `models/`, `ldm/`: kiến trúc model và diffusion modules

---

## Quick Start — RTX 5090 (ưu tiên cho RunPod và local training)

> Đây là luồng mặc định được khuyến nghị cho máy có RTX 5090 32GB VRAM.

### 1) Setup môi trường

```bash
bash control/setup_env.sh
conda activate artgen
```

### 2) Tải và gộp weights

```bash
bash control/download_weights.sh
bash control/prepare_weights.sh
```

### 3) Sanity check trước khi train

```bash
bash control/download_dataset.sh
bash control/sanity_check.sh
```

### 4) Train

```bash
# train từ đầu
bash control/train.sh

# resume checkpoint mới nhất
bash control/train.sh --resume

# resume checkpoint cụ thể
bash control/train.sh --resume_from ckpt/trained/<your_ckpt>.ckpt
```

### 5) Inference (Streamlit)

```bash
# tự chọn checkpoint mới nhất
bash control/inference.sh

# chỉ định checkpoint
bash control/inference.sh --ckpt ckpt/trained/<your_ckpt>.ckpt
```

Tài liệu đầy đủ cho luồng này: `md/pod_guide.md`.

---

## Notebook/Kaggle workflow

Nếu bạn chạy trong notebook environment (Kaggle), **không dùng luồng Pod/local ở trên**.

- Đi theo tài liệu: `md/kaggle_guide.md` (nhánh `md/kaggle.md`)
- Dùng config Kaggle:
	- `configs/kaggle_train_config.yaml`
	- `configs/kaggle_inference_config.yaml`

---

## Bảng map nhanh môi trường

| Môi trường           | Train config                       | Inference config                       | Tài liệu             |
| -------------------- | ---------------------------------- | -------------------------------------- | -------------------- |
| Pod / Local RTX 5090 | `configs/pod_train_config.yaml`    | `configs/pod_inference_config.yaml`    | `md/pod_guide.md`    |
| Notebook (Kaggle)    | `configs/kaggle_train_config.yaml` | `configs/kaggle_inference_config.yaml` | `md/kaggle_guide.md` |
| Default/base         | `configs/train_config.yaml`        | `configs/inference_config.yaml`        | N/A                  |

---

## Lưu ý vận hành

- Các script trong `control/` là bash script, phù hợp Linux/WSL2/RunPod.
- `control/inference.sh` sẽ cập nhật tạm `configs/inference_config.yaml` theo checkpoint bạn chọn, rồi restore khi app dừng.
- `inference/app.py` dùng bộ ví dụ prompt cục bộ, không khởi tạo dataset CompArt khi mở app.
- Không thấy `eval.py` trong repo hiện tại, nên luồng mặc định tập trung vào **train + streamlit inference**.

---

## Tài liệu chi tiết

- Pod/RTX 5090: `md/pod_guide.md`
- Kaggle notebook: `md/kaggle_guide.md`
- Inference từ đầu (weights HF + Streamlit): `md/inference_setup.md`

