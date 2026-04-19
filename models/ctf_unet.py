"""
CTFUNetModel — Coarse-to-Fine U-Net with layer-aware context routing.

Drop-in replacement for UNetModel (SD v1.5).
- If context is a dict → CTF routing (different blocks get different contexts).
- If context is a tensor/None → fallback to original UNetModel (backward compatible).

SD v1.5 block layout (channel_mult=[1,2,4,4], attention_resolutions=[4,2,1], num_res_blocks=2):

  input_blocks with SpatialTransformer (cross-attn consumes context):
    [1, 2]  → 64×64, 320ch  (ds=1)  — LAYOUT
    [4, 5]  → 32×32, 640ch  (ds=2)  — BLEND layout→content
    [7, 8]  → 16×16, 1280ch (ds=4)  — CONTENT

  middle_block → 8×8, 1280ch — CONTENT

  output_blocks with SpatialTransformer:
    [3, 4, 5]   → 16×16, 1280ch (ds=4)  — STYLE
    [6, 7, 8]   → 32×32, 640ch  (ds=2)  — BLEND content→style
    [9, 10, 11] → 64×64, 320ch  (ds=1)  — STYLE

  Blocks WITHOUT SpatialTransformer (context is ignored by TimestepEmbedSequential):
    input [0, 3, 6, 9, 10, 11] and output [0, 1, 2]
"""
import torch
import torch.nn as th

from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.diffusionmodules.util import timestep_embedding


# ── SD v1.5 exact block index sets ────────────────────────────────
# Encoder blocks WITH SpatialTransformer
LAYOUT_IN  = {1, 2}        # 64×64, 320ch — coarse layout structure (Prompt 1 / CLIP)
BLEND_IN   = {4, 5}        # 32×32, 640ch — transition layout → content
CONTENT_IN = {7, 8}        # 16×16, 1280ch — object semantics (Prompt 2 / CLIP)

# Decoder blocks WITH SpatialTransformer
STYLE_OUT  = {3, 4, 5,     # 16×16, 1280ch — fine semantic details (Prompt 3 / ArtDapter)
              9, 10, 11}   # 64×64, 320ch — final texture/style
BLEND_OUT  = {6, 7, 8}     # 32×32, 640ch — transition content → style
# ──────────────────────────────────────────────────────────────────


def lerp(a: torch.Tensor, b: torch.Tensor, w: float) -> torch.Tensor:
    """
    Linear interpolation: w=0 → a, w=1 → b.
    Crops to min(T_a, T_b) along the sequence dimension to avoid shape errors
    when CLIP (77 tokens) is blended with ArtDapter (64 tokens).
    """
    T = min(a.shape[1], b.shape[1])
    return (1.0 - w) * a[:, :T] + w * b[:, :T]


class CTFUNetModel(UNetModel):
    """
    Coarse-to-Fine UNetModel: routes different context signals to different
    block groups based on their spatial resolution and role in the U-Net.

    Context dict keys:
        'layout'  : (B, 77, 768)  — CLIP embedding of Prompt 1 (layout)
        'content' : (B, 77, 768)  — CLIP embedding of Prompt 2 (content)
        'style'   : (B, 64, 768)  — ArtDapter output of Prompt 3 (style)
        'alpha'   : float         — denoising progress blend weight (0→1)
    """

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        # ── Fallback: legacy mode (backward compatibility) ────────
        if not isinstance(context, dict):
            return super().forward(x, timesteps, context, y, **kwargs)

        # ── CTF mode ──────────────────────────────────────────────
        layout  = context['layout']     # (B, 77, 768) — CLIP P1
        content = context['content']    # (B, 77, 768) — CLIP P2
        style   = context['style']      # (B, 64, 768) — ArtDapter P3
        alpha   = float(context.get('alpha', 0.0))  # 0.0 → 1.0

        # Standard U-Net time embedding
        hs    = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb   = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y is not None and y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        # ── Alpha Blended Context ──────────────────────────────────
        # content_ctx dynamically interpolates between Content (P2) and Style (P3)
        # alpha=0: early steps, content only. Delineates geometry and structure.
        # alpha=1: late steps, style only. Applies heavy artistic traits.
        content_ctx = lerp(content, style, alpha)

        # ── Encoder: 3-tier routing ──────────────────────────────
        for i, module in enumerate(self.input_blocks):
            if i in LAYOUT_IN:       # {1, 2} — 64×64 — Coarse: subject + position
                ctx = layout
            elif i in BLEND_IN:      # {4, 5} — 32×32 — Spatial transition
                # Let alpha determine if it's content-focused or style-focused
                ctx = lerp(layout, content_ctx, 0.5) 
            elif i in CONTENT_IN:    # {7, 8} — 16×16 — Full content/style detail
                ctx = content_ctx
            else:
                # Blocks without SpatialTransformer (context is ignored)
                ctx = content_ctx
            h = module(h, emb, context=ctx)
            hs.append(h)

        # ── Bottleneck ────────────────────────────────────────────
        h = self.middle_block(h, emb, content_ctx)  # Needs style to alter deep semantics

        # ── Decoder ──────────────────────────────────────────────
        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            # Output blocks all receive the blended content/style
            # if they have SpatialTransformer.
            # Even if they don't, passing context is harmless
            ctx = content_ctx
            h = module(h, emb, ctx)

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
