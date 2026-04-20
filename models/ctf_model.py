"""
ArtDaptedModelCTF — Coarse-to-Fine variant of ArtDaptedModel.

Dual encoding strategy:
  - P1/P2 → CLIP encoder (native CLIP space, compatible with SD v1.5 U-Net)
  - P3 → T5 encoder (cond_stage_model) → ArtDapter (translates T5 → CLIP space)

The conditioning dict format:
  {
      'c_layout':  [Tensor(B, 77, 768)]   ← CLIP P1 (layout)
      'c_content': [Tensor(B, 77, 768)]   ← CLIP P2 (content)
      'c_style':   [Tensor(B, T, 2048)]   ← T5 P3 raw (ArtDapter runs inside apply_model)
      'alpha':     float                  ← step blend weight (injected by CTFDDIMSampler)
  }
"""
import torch
import logging

from ldm.util import instantiate_from_config
from models.artdapted_model import ArtDaptedModel
from models.prompt_decomposer import PromptDecomposer
from ldm.modules.encoders.modules import CLIPEmbedder

logger = logging.getLogger(__name__)


class ArtDaptedModelCTF(ArtDaptedModel):
    """
    Coarse-to-Fine extension of ArtDaptedModel.

    - Adds a CLIP encoder for P1 (layout) and P2 (content).
    - Uses existing T5 encoder + ArtDapter for P3 (style).
    - Overrides apply_model to build a context_dict for CTFUNetModel.
    """

    def __init__(self, *args,
                 openai_api_key: str = None,
                 llm_model: str = "gpt-4o-mini",
                 style_start: float = 0.7,
                 clip_version: str = "openai/clip-vit-large-patch14",
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.style_start = style_start

        # CLIP encoder for P1/P2 — small (~250MB), native CLIP space
        self.clip_encoder = CLIPEmbedder(version=clip_version, freeze=True)

        # LLM-based prompt decomposer
        self.decomposer = PromptDecomposer(
            api_key=openai_api_key, model=llm_model
        )

        logger.info(
            "CTF model initialized: CLIP=%s, T5=%s, ArtDapter=%s",
            clip_version,
            self.cond_stage_model.__class__.__name__,
            self.artdapter.__class__.__name__,
        )

    # ─────────────────────── Encoding ───────────────────────────

    @torch.no_grad()
    def encode_clip(self, prompts: list) -> torch.Tensor:
        """
        Encode P1/P2 via CLIP → (B, 77, 768) in native CLIP space.
        U-Net cross-attn was trained with CLIP space → perfect compatibility.
        """
        # Ensure CLIP is on same device as model
        if self.clip_encoder.device != self.device:
            self.clip_encoder = self.clip_encoder.to(self.device)
        return self.clip_encoder(prompts)  # (B, 77, 768)

    # ─────────────────────── apply_model ────────────────────────

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """
        Temporal Proxy Prompt routing:
        1. If Phase 1/2: cond has 'c_crossattn' (CLIP native) -> fed directly to UNet.
        2. If Phase 3: cond has 'c_style_raw' (T5 raw) -> push through ArtDapter -> UNet.
        """
        if 'c_style_raw' in cond:
            # Phase 3: T5 -> ArtDapter
            t5_raw = torch.cat(cond['c_style_raw'], 1)
            style = self.artdapter(t5_raw, t)
            return self.model.diffusion_model(x=x_noisy, timesteps=t, context=style)
            
        elif 'c_crossattn' in cond:
            # Phase 1 & 2: Native CLIP
            context = torch.cat(cond['c_crossattn'], 1)
            return self.model.diffusion_model(x=x_noisy, timesteps=t, context=context)
        
        else:
            raise ValueError(f"Unknown conditioning keys: {cond.keys()}")

    # ─────────────────────── CTF Conditioning ──────────────────

    @torch.no_grad()
    def get_ctf_conditioning(
        self,
        captions: list,
        art_styles: list,
        PoAs: list,
        sample_quantity: int,
    ) -> dict:
        """
        Full CTF conditioning pipeline:
        1. Decompose prompts via LLM (1 API call per sample, cached)
        2. Encode P1/P2 via CLIP, P3 via T5
        3. Return conditioning dict ready for CTFDDIMSampler

        Note: ArtDapter runs inside apply_model (needs timestep 't').
        """
        decomposed = self.decomposer.decompose_batch(captions, art_styles, PoAs)

        # Save first sample's decomposition for UI display
        self._last_decomposed = decomposed[0] if decomposed else {}

        # Ép P1 phải vẽ nháp xám (grayscale/blockout) để không phá hỏng Phase 3
        p1_prompts = [d['prompt1'] + ", rough uncolored grayscale blockout, empty neutral background" for d in decomposed]
        
        # P2: Nội dung thô cơ bản
        p2_prompts = [d['prompt2'] for d in decomposed]
        
        # P3: BẮT BUỘC nhúng tên Art Style vào cuối để đảm bảo T5 kích hoạt đúng phong cách
        p3_prompts = []
        for i, d in enumerate(decomposed):
            style = art_styles[i]
            p3_text = d['prompt3']
            if style:
                # Nếu GPT lỡ quên từ khoá style, ta nhồi lại vào cuối
                if style.lower() not in p3_text.lower():
                    p3_text += f", {style} style"
            p3_prompts.append(p3_text)

        # Print decomposition for debugging (visible in Kaggle/terminal output)
        for i, d in enumerate(decomposed):
            print(f"\n{'='*60}")
            print(f"🔍 CTF Decomposition (Sample {i}):")
            print(f"  [P1] Layout : {p1_prompts[i]}")
            print(f"  [P2] Content: {p2_prompts[i]}")
            print(f"  [P3] Full   : {p3_prompts[i]}")
            print(f"{'='*60}")

        cond_layout  = self.encode_clip(p1_prompts)              # (B, 77, 768)
        cond_content = self.encode_clip(p2_prompts)              # (B, 77, 768)
        # T5 raw — ArtDapter will run inside apply_model (needs timestep)
        cond_style_raw = self.get_learned_conditioning(p3_prompts)  # (B, T, 2048)

        return dict(
            c_layout  = [cond_layout],
            c_content = [cond_content],
            c_style   = [cond_style_raw],
            alpha     = 0.0,    # will be updated by CTFDDIMSampler at each step
        )

    @torch.no_grad()
    def get_unconditional_conditioning(self, n: int) -> dict:
        """
        Unconditional conditioning for CFG.
        Must have same dict structure as ctf conditioning.
        """
        empty_clip = self.encode_clip([""] * n)                  # (B, 77, 768)
        # Empty T5 — use parent's cond_stage_model
        empty_t5 = self.get_learned_conditioning([""] * n)       # (B, T, 2048)
        return dict(
            c_layout  = [empty_clip],
            c_content = [empty_clip],
            c_style   = [empty_t5],
            alpha     = 0.0,
        )

    # ─────────────────────── VRAM management ────────────────────

    def low_vram_shift(self, mode, device):
        """Override to also handle CLIP encoder device placement."""
        super().low_vram_shift(mode, device)
        if mode == 'cond_stage':
            self.clip_encoder = self.clip_encoder.to(device)
        elif mode in ('first_stage', 'diffuse_stage'):
            self.clip_encoder = self.clip_encoder.cpu()
