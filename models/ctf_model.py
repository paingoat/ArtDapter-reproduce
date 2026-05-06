"""
ArtDaptedModelCTF — Coarse-to-Fine variant of ArtDaptedModel.

Dual encoding strategy:
  - P1/P2 → CLIP encoder (native CLIP space, compatible with SD v1.5 U-Net)
  - P3 → T5 encoder (cond_stage_model) → ArtDapter (translates T5 → CLIP space)

P3 uses apply_prompt_template() to generate prompts in the EXACT format that
ArtDapter was trained on: "Prompt: ... Style: ... Balance: ... Harmony: ..."
This prevents Data Shift (OOD) that occurs when using free-form ChatGPT prompts.

The conditioning dict format:
  {
      'c_layout':  [Tensor(B, 77, 768)]   ← CLIP P1 (layout)
      'c_content': [Tensor(B, 77, 768)]   ← CLIP P2 (content)
      'c_style':   [Tensor(B, T, 2048)]   ← T5 P3 raw (ArtDapter runs inside apply_model)
      'no_style':  bool                   ← Phase Analysis flag (structure-only mode)
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
    - P3 prompt is generated via apply_prompt_template() (original training format).
    - Overrides apply_model to build a context_dict for CTFUNetModel.
    """

    def __init__(self, *args,
                 openai_api_key: str = None,
                 llm_model: str = "gpt-4o-mini",
                 clip_version: str = "openai/clip-vit-large-patch14",
                 **kwargs):
        super().__init__(*args, **kwargs)

        # CLIP encoder for P1/P2 — small (~250MB), native CLIP space
        self.clip_encoder = CLIPEmbedder(version=clip_version, freeze=True)

        # LLM-based prompt decomposer (for P1 layout & P2 content only)
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
        Build context_dict and forward through CTFUNetModel.

        Supports two cond formats:
          1. CTF dict: {'c_layout': [...], 'c_content': [...], 'c_style': [...]}
          2. Legacy dict: {'c_crossattn': [...]} → falls back to original ArtDaptedModel behavior
        """
        # Legacy fallback: original ArtDaptedModel behavior
        if 'c_crossattn' in cond:
            return super().apply_model(x_noisy, t, cond, *args, **kwargs)

        # ── CTF mode ──────────────────────────────────────────────
        layout  = torch.cat(cond['c_layout'], 1)    # (B, 77, 768) CLIP
        content = torch.cat(cond['c_content'], 1)    # (B, 77, 768) CLIP
        t5_raw  = torch.cat(cond['c_style'], 1)      # (B, T, 2048) T5 raw

        # ArtDapter: translate T5 → CLIP-compatible space, time-conditioned
        # t is the diffusion timestep — ArtDapter's TSC uses it to adapt
        # style features based on the current noise level
        style = self.artdapter(t5_raw, t)             # (B, 64, 768)

        no_style = bool(cond.get('no_style', False))

        context_dict = {
            'layout':   layout,
            'content':  content,
            'style':    style,
            'no_style': no_style,
        }

        # CTFUNetModel reads the dict and routes per block
        return self.model.diffusion_model(
            x=x_noisy, timesteps=t, context=context_dict
        )

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
        1. Decompose prompts via LLM → P1 (layout), P2 (content)
        2. Build P3 via apply_prompt_template (original training format)
        3. Encode P1/P2 via CLIP, P3 via T5
        4. Return conditioning dict ready for CTFDDIMSampler

        Note: ArtDapter runs inside apply_model (needs timestep 't').
        """
        decomposed = self.decomposer.decompose_batch(captions, art_styles, PoAs)

        # Save first sample's decomposition for UI display
        self._last_decomposed = decomposed[0] if decomposed else {}

        p1_prompts = [d['prompt1'] for d in decomposed]
        p2_prompts = [d['prompt2'] for d in decomposed]

        # P3: use ORIGINAL template format (matches ArtDapter training distribution)
        # Format: "Prompt: ... Style: ... Balance: ... Harmony: ... Variety: ..."
        p3_prompts = self.apply_prompt_template(captions, art_styles, PoAs)

        # Print decomposition for debugging (visible in Kaggle/terminal output)
        for i, d in enumerate(decomposed):
            print(f"\n{'='*60}")
            print(f"🔍 CTF Decomposition (Sample {i}):")
            print(f"  [P1] Layout  : {d['prompt1']}")
            print(f"  [P2] Content : {d['prompt2']}")
            print(f"  [P3] Style   : {p3_prompts[i][:120]}...")
            print(f"{'='*60}")

        cond_layout  = self.encode_clip(p1_prompts)              # (B, 77, 768)
        cond_content = self.encode_clip(p2_prompts)              # (B, 77, 768)
        # T5 raw — ArtDapter will run inside apply_model (needs timestep)
        cond_style_raw = self.get_learned_conditioning(p3_prompts)  # (B, T, 2048)

        return dict(
            c_layout  = [cond_layout],
            c_content = [cond_content],
            c_style   = [cond_style_raw],
            no_style  = False,
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
            no_style  = False,
        )

    # ─────────────────────── VRAM management ────────────────────

    def low_vram_shift(self, mode, device):
        """Override to also handle CLIP encoder device placement."""
        super().low_vram_shift(mode, device)
        if mode == 'cond_stage':
            self.clip_encoder = self.clip_encoder.to(device)
        elif mode in ('first_stage', 'diffuse_stage'):
            self.clip_encoder = self.clip_encoder.cpu()
