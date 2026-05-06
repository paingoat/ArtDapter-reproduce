"""
CTFUNetModel — Coarse-to-Fine U-Net with Decoupled Context Routing.

Drop-in replacement for UNetModel (SD v1.5).
- If context is a dict → CTF routing (different blocks get different contexts).
- If context is a tensor/None → fallback to original UNetModel (backward compatible).

SD v1.5 block layout (channel_mult=[1,2,4,4], attention_resolutions=[4,2,1], num_res_blocks=2):

  input_blocks with SpatialTransformer (cross-attn consumes context):
    [1, 2]  → 64×64, 320ch  (ds=1)  — Upper (In-Block 0)
    [4, 5]  → 32×32, 640ch  (ds=2)  — Middle (In-Block 1)
    [7, 8]  → 16×16, 1280ch (ds=4)  — Bottom (In-Block 2)

  middle_block → 8×8, 1280ch — Bottleneck

  output_blocks with SpatialTransformer:
    [3, 4, 5]   → 16×16, 1280ch (ds=4)  — Bottom (Out-Block 1)
    [6, 7, 8]   → 32×32, 640ch  (ds=2)  — Middle (Out-Block 2)
    [9, 10, 11] → 64×64, 320ch  (ds=1)  — Upper (Out-Block 3)

  Blocks WITHOUT SpatialTransformer (context is ignored by TimestepEmbedSequential):
    input [0, 3, 6, 9, 10, 11] and output [0, 1, 2]

Experimental findings on block roles:
  Upper Blocks (In-Block 0 + Out-Block 3):
    → Most effective for STYLIZATION. Fine-tuning these yields optimal
      balance: preserving character details while accurately rendering style.
  Middle Blocks (In-Block 1 + Out-Block 2):
    → Focus on entity detail and character IDENTITY/CONTENT.
      Activating only these for style results in clear objects but lost style.
  Bottom Blocks (In-Block 2 + Out-Block 1):
    → Least effective at absorbing new concepts.
      Using only these loses both character and style information.

Decoupled context routing (each block receives exactly ONE context type):
  Upper  → Style   (ArtDapter output of template prompt P3)
  Middle → Content (CLIP P2)
  Bottom → Layout  (CLIP P1)
"""
import torch
import torch.nn as th

from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.diffusionmodules.util import timestep_embedding


# ── SD v1.5 exact block index sets (based on experimental findings) ───
# Encoder blocks WITH SpatialTransformer
STYLE_IN   = {1, 2}        # In-Block 0 (Upper) — 64×64, 320ch — STYLE (ArtDapter P3)
CONTENT_IN = {4, 5}        # In-Block 1 (Middle) — 32×32, 640ch — CONTENT (CLIP P2)
LAYOUT_IN  = {7, 8}        # In-Block 2 (Bottom) — 16×16, 1280ch — LAYOUT (CLIP P1)

# Decoder blocks WITH SpatialTransformer
LAYOUT_OUT  = {3, 4, 5}    # Out-Block 1 (Bottom) — 16×16, 1280ch — LAYOUT (CLIP P1)
CONTENT_OUT = {6, 7, 8}    # Out-Block 2 (Middle) — 32×32, 640ch — CONTENT (CLIP P2)
STYLE_OUT   = {9, 10, 11}  # Out-Block 3 (Upper) — 64×64, 320ch — STYLE (ArtDapter P3)
# ──────────────────────────────────────────────────────────────────────


class CTFUNetModel(UNetModel):
    """
    Coarse-to-Fine UNetModel with Decoupled Context Routing.

    Each U-Net block group receives exactly ONE context signal, with no
    blending or interpolation between different embeddings. This prevents
    variance mismatch and LayerNorm shock that causes broken/plastic images.

    Context dict keys:
        'layout'   : (B, 77, 768)  — CLIP embedding of Prompt 1 (spatial layout)
        'content'  : (B, 77, 768)  — CLIP embedding of Prompt 2 (object identity)
        'style'    : (B, 64, 768)  — ArtDapter output of template Prompt 3 (artistic style)
        'no_style' : bool          — If True, Upper blocks use content instead of style
                                     (used for Phase 1 structure-only sampling)
    """

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        # ── Fallback: legacy mode (backward compatibility) ────────
        if not isinstance(context, dict):
            return super().forward(x, timesteps, context, y, **kwargs)

        # ── CTF mode ──────────────────────────────────────────────
        layout  = context['layout']     # (B, 77, 768) — CLIP P1
        content = context['content']    # (B, 77, 768) — CLIP P2
        style   = context['style']      # (B, 64, 768) — ArtDapter P3

        # Phase Analysis: when no_style=True, Upper blocks fall back to content
        # so the image shows pure structure without any artistic style applied.
        no_style = bool(context.get('no_style', False))
        upper_ctx = content if no_style else style

        # Standard U-Net time embedding
        hs    = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb   = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y is not None and y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)

        # ── Encoder ──────────────────────────────────────────────
        for i, module in enumerate(self.input_blocks):
            if i in STYLE_IN:            # {1, 2} — Upper — Style
                ctx = upper_ctx
            elif i in CONTENT_IN:        # {4, 5} — Middle — Content (identity)
                ctx = content
            elif i in LAYOUT_IN:         # {7, 8} — Bottom — Layout (spatial)
                ctx = layout
            else:
                # Blocks without SpatialTransformer (context is ignored anyway)
                ctx = content
            h = module(h, emb, context=ctx)
            hs.append(h)

        # ── Bottleneck ────────────────────────────────────────────
        h = self.middle_block(h, emb, content)

        # ── Decoder ──────────────────────────────────────────────
        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            if i in LAYOUT_OUT:          # {3, 4, 5} — Bottom — Layout
                ctx = layout
            elif i in CONTENT_OUT:       # {6, 7, 8} — Middle — Content (identity)
                ctx = content
            elif i in STYLE_OUT:         # {9, 10, 11} — Upper — Style
                ctx = upper_ctx
            else:
                # Blocks without SpatialTransformer
                ctx = content
            h = module(h, emb, ctx)

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
