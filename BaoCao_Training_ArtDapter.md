# Báo cáo Ngắn: Quá trình Huấn luyện ArtDapter

## Tóm tắt
Báo cáo này tổng kết quá trình huấn luyện mô hình ArtDapter — một diffusion model tuỳ chỉnh được điều khiển bởi style và principles of art. Sau ~140k bước training, loss đã ổn định ở mức 0.130–0.132. Phân tích chỉ ra rằng mô hình chưa có dấu hiệu phân kỳ và có thể cần điều chỉnh learning rate hoặc chu kỳ để tiếp tục cải thiện.

---

## 1. Giới thiệu

**ArtDapter** là một extension đối với Stable Diffusion, cho phép điều khiển quá trình sinh ảnh thông qua:
- **Art Style**: phong cách nghệ thuật (pop art, watercolor, v.v.)
- **Principles of Art (PoA)**: các nguyên tắc thiết kế (balance, contrast, emphasis, v.v.)
- **Caption**: mô tả văn bản thêm về dự tính nội dung

Mô hình sử dụng **latent diffusion**, trong đó:
- Không gian tiềm ẩn (latent space): VAE encoder nén ảnh thành vector 4D
- Network chính (UNet): học dự đoán nhiễu tại các timestep khác nhau
- Điều kiện (conditioning): embedding từ text encoder (T5) được xử lý qua ArtDapter module để tích hợp style/PoA

---

## 2. Phương pháp

### 2.1 Cấu hình Huấn luyện
| Tham số | Giá trị |
|--------|--------|
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Batch size | 22 |
| Accumulate grad batches | 1 |
| Total training steps | 280,000 |
| Precision | 16-mixed (float16) |
| Checkpoint dir | `./ckpt/trained/` |
| Log frequency | 500 steps |

**Model Architecture:**
- **UNet**: 320→640→1280→1280 channels, 8 attention heads, crossattn conditioning
- **VAE Encoder**: 4×downsampling, embed_dim=4
- **T5 Text Encoder**: 768-dim embeddings, max 512 tokens
- **ArtDapter**: custom module xử lý embeddings trước khi đưa vào diffusion UNet

### 2.2 Hàm Loss

Mô hình sử dụng **diffusion-based loss** với công thức:

$$\mathcal{L} = w_{\text{simple}} \cdot \mathcal{L}_{\text{simple}} + w_{\text{elbo}} \cdot \mathcal{L}_{\text{VLB}}$$

Trong đó:

- **$\mathcal{L}_{\text{simple}}$** (loss_simple): MSE dự đoán nhiễu trong tham số hóa **$\epsilon$** (epsilon):
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) \right\|_2^2 \right]$$

- **$\mathcal{L}_{\text{VLB}}$** (loss_vlb): Variational Lower Bound, trọng số theo timestep:
$$\mathcal{L}_{\text{VLB}} = \mathbb{E}_t \left[ w_t^{\text{VLB}} \cdot \mathcal{L}_{\text{simple}} \right]$$

| Tham số | Mặc định | Ý nghĩa |
|--------|---------|--------|
| $w_{\text{simple}}$ | 1.0 | Trọng số loss_simple |
| $w_{\text{elbo}}$ | 0.0 | Trọng số VLB (hiện tắt) |
| Parameterization | $\epsilon$ | Dự đoán noise thay vì $x_0$ |
| Loss type | L2 (MSE) | Metric so sánh |

**Nhận xét:** Hiện tại $w_{\text{elbo}} = 0$, nên $\mathcal{L}_{\text{VLB}}$ không góp vào objective, mô hình chỉ tối ưu $\mathcal{L}_{\text{simple}}$.

---

## 3. Kết quả Huấn luyện

### 3.1 Phân tích Loss Curves

Dựa trên biểu đồ WandB (140k steps đã train):

#### **train/loss_vlb_step** (mỗi step)
- Có spiking nhưng tập trung ≤ 0.04
- VLB_epoch trung bình ≈ 0.00235 (rất nhỏ, do $w_{\text{elbo}} = 0$)
- **Kết luận**: Ổn định, không diverge

#### **train/loss_simple_step** (mỗi step)
- Dao động trong khoảng 0.05–0.20 (bình thường vì batch nhỏ + t ngẫu nhiên)
- loss_simple_epoch: ổn định ≈ 0.130–0.132
- **Kết luận**: Không có xu hướng tăng, weight accumulation gần như không ảnh hưởng

#### **train/loss_step** (mỗi step)
- Trùng khớp với loss_simple vì $w_{\text{elbo}} = 0$
- loss_epoch: đồng bộ với loss_simple_epoch ≈ 0.130–0.132
- **Kết luận**: Hành vi như mong đợi

### 3.2 Quan sát Chính

| Chỉ số | Nhận xét |
|-------|---------|
| **Sự ổn định** | ✓ Không có spike catastrophe hoặc divergence |
| **Xu hướng giảm** | ⚠ Chậm, gần như plateau từ ~60k steps |
| **Variance (step-level)** | ✓ Bình thường, không bất thường |
| **Sự hội tụ** | ⚠ Chưa rõ model đã lên plateau hay còn cơ hội cải thiện |

---

## 4. Kỹ thuật & Kết quả Liên quan

### 4.1 Tuyên bố Kiến trúc
- **Frozen Stable Diffusion backbone**: sd_locked = True (chỉ train ArtDapter + một số layer cuối)
- **EMA không dùng**: use_ema = False (chưa bật, có thể giúp ổn định)
- **Dropout/Regularization**: weight_decay=0.01 (vừa phải)

### 4.2 Dataset
- **Nguồn**: CompArt (Hugging Face)
- **Kích cỡ ảnh**: 512px (nén về 64px latent space, 4 channels)
- **Augmentation**: drop_caption_prob=0.5, drop_art_style_prob=0.0, PoA dropout tuỳ chỉnh
- **Batchsize hiệu quả**: 22

---

## 5. Khuyến nghị

### 5.1 Ngắn hạn (tiếp tục train)

1. **Bật Learning Rate Scheduler**
   - Loss gần plateau → cần giảm LR để fine-tune
   - Đề xuất: cosine annealing từ 1e-4 → 1e-5 ở bước 200k
   ```yaml
   scheduler_config:
     target: torch.optim.lr_scheduler.CosineAnnealingLR
     params:
       T_max: 100000
       eta_min: 1e-5
   ```

2. **Tăng Batch Size hoặc Gradient Accumulation**
   - Batch size 22 khá nhỏ → tăng lên 32–44 (hay accumulate 2 steps) sẽ giảm noise trong gradient

3. **Bật EMA (Exponential Moving Average)**
   - `use_ema: True` giúp ổn định không gian weight, có thể giảm validation loss

### 5.2 Trung hạn (phân tích kỹ hơn)

4. **Kiểm tra Validation**
   - Báo cáo hiện chỉ log training loss, cần validation splits để phát hiện overfit
   - Monitor: `val/loss_simple`, `val/loss_epoch` 

5. **Ablation Study**
   - Bật $w_{\text{elbo}} > 0$ để xem VLB góp vào objective có tốt hơn không
   - Thử parameterization khác nhau ("x0" vs "eps" vs "v")

6. **Kiểm tra Inference Quality**
   - DDIM sampling 50 steps, CFG scale 7.5
   - So sánh chất lượng ảnh sinh giữa checkpoint lúc 60k vs 140k

### 5.3 Dài hạn

7. **Fine-tuning Ban Đầu**
   - Sau khi loss hội tụ, có thể unfreeze cuối UNet (`sd_locked=False`) để tinh chỉnh
   - Cần transfer learning từ pretrained weights cẩn thận

---

## 6. Kết luận

Quá trình huấn luyện ArtDapter đã hoàn thành 140k / 280k bước với các tín hiệu tích cực:
- ✓ **Ổn định**: không diverge, không NaN
- ✓ **Loss hợp lý**: trung bình 0.130 đối với diffusion model
- ⚠ **Plateau sớm**: cần can thiệp LR để tiếp tục cải thiện

Với các tối ưu hóa được đề xuất, kỳ vọng có thể đạt loss < 0.120 và sinh ảnh có chất lượng cao hơn. Cần hoàn thành thêm 140k bước train với scheduler + validation monitoring để có kết luận cuối cùng.

---

## Phụ lục: Công thức Diffusion Chi tiết

**Diffusion process (forward):**
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

**Reverse process (model training):**
$$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) \approx \boldsymbol{\epsilon}$$

**Loss for timestep $t$:**
$$\ell_t = \mathbb{E} \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) \right\|_2^2$$

**Final loss (averaged over timesteps and batch):**
$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \ell_t$$

---

**Ngày lập báo cáo**: 18-03-2026  
**Phiên bản mô hình**: ArtDapterTSC  
**Repository**: ArtDapter (NCKH)
