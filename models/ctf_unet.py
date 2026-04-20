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
    """Legacy lerp function - kept for compatibility if needed elsewhere."""
    T = min(a.shape[1], b.shape[1])
    return (1.0 - w) * a[:, :T] + w * b[:, :T]


class CTFUNetModel(UNetModel):
    """
    Temporal Proxy Prompt U-Net.
    Since we now swap the prompts temporally (along steps) rather than spatially (along layers),
    this network simply acts as a standard UNetModel.
    Any tensor context passed will be routed naturally by the underlying SD architecture.
    """

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        # If context is a dict, it's deprecated. We expect tensors now.
        if isinstance(context, dict):
            raise ValueError(
                "CTFUNetModel now expects a Tensor context (Proxy Prompt) "
                "instead of a dict. Please update the sampling pipeline."
            )

        # Standard U-Net behavior
        return super().forward(x, timesteps, context, y, **kwargs)

