"""
Prompt Decomposer — Phân rã prompt thành 3 cấp bậc qua LLM API.

Workflow:
  1 lần gọi API duy nhất → JSON {"prompt1", "prompt2", "prompt3"}
  - prompt1: Keyword-style spatial layout cues (CLIP-friendly, ≤ ~40 CLIP tokens).
  - prompt2: Keyword-style layout + full content (CLIP-friendly, ≤ ~60 CLIP tokens).
  - prompt3: Natural-language description for T5 (style + PoA aware, ≤ ~100 words).

P1/P2 được cắt cứng theo CLIP tokenizer thật (77 token limit) để không bị silently truncated
khi đi vào SD v1.5 cross-attention.
"""
import os
import re
import json
import logging

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

# CLIP tokenizer hard limit (v1.5 text encoder)
CLIP_MAX_TOKENS = 77
# Safety budgets (< 77 to leave headroom for BOS/EOS/padding variations)
P1_TOKEN_BUDGET = 50
P2_TOKEN_BUDGET = 65

SYSTEM_PROMPT = """\
You are an art prompt decomposer. From the given content + style + principles,
return a JSON object with exactly 3 keys in ONE response.

prompt1 and prompt2 target a CLIP text encoder, which reads best as SHORT,
COMMA-SEPARATED KEYWORD PHRASES (not full sentences). prompt3 targets a T5
encoder and may be a natural-language description.

"prompt1" (Spatial Layout, CLIP-friendly, ≤ 40 tokens):
  - Output a COMMA-SEPARATED list of short phrases (2-5 words each).
  - KEEP core subject nouns. STRIP all adjectives about material, color, lighting, mood, style.
  - Focus on: subject nouns, positions, shapes, silhouettes, composition cues.
  - Composition cues allowed: centered, symmetric, asymmetric, rule of thirds,
    radial balance, diagonal, foreground, background, left, right, top, bottom.
  - No colors, no textures, no lighting, no art-movement words, no adjectives of mood.
  - Good: "circular machine, centered, symmetric, radial rays, frontal view"
  - Bad : "A glowing blue turbine hovering dramatically in the center..."

"prompt2" (Layout + Content, CLIP-friendly, ≤ 60 tokens):
  - Output a COMMA-SEPARATED list of short phrases.
  - MUST RESTATE the same layout cues from prompt1 (so bố cục được giữ nguyên).
  - Then ADD content nouns and descriptive adjectives (size, quantity, actions, secondary objects).
  - Still NO style words, NO art-movement words, NO explicit lighting/mood adjectives.
  - Good: "spaceship, hovering, stone courtyard, centered, symmetric, engine exhaust downward,
           small figures, tall columns, foreground debris"
  - Bad : "A cinematic dramatic scene of a glowing matte spaceship..."

"prompt3" (Full description for T5, ≤ 100 words):
  - Natural-language paragraph.
  - Combine content (from prompt2) + art style + artistic principles vividly.
  - Style words, textures, lighting, mood ARE encouraged here.

Return ONLY: {"prompt1": "...", "prompt2": "...", "prompt3": "..."}
No markdown, no extra keys.\
"""

# Style / aesthetic words that should NOT appear in prompt1 or prompt2.
# Broadened from original list — cover common traps LLM still falls into.
_STYLE_PATTERN = re.compile(
    r'\b('
    # art movements
    r'baroque|renaissance|impressionism|impressionist|expressionism|expressionist|'
    r'cubism|cubist|surrealism|surrealist|pop art|art nouveau|ukiyo-e|'
    r'romanticism|realism|minimalism|minimalist|abstract|abstract expressionism|'
    r'art deco|fauvism|pointillism|symbolism|mannerism|rococo|'
    # media / materials
    r'watercolor|oil painting|oil paint|acrylic|gouache|tempera|ink wash|charcoal|'
    r'pastel|sketch|lineart|line art|woodcut|etching|lithograph|'
    # texture / brushwork
    r'brushstroke|brushwork|impasto|painterly|stylized|textured|grainy|'
    # lighting / mood / aesthetics
    r'chiaroscuro|vibrant|muted|dramatic lighting|ethereal|dreamlike|atmospheric|'
    r'moody|cinematic|matte|photoreal|photorealistic|hyperreal|hyperrealistic|'
    r'neon|cyberpunk|steampunk|vintage|retro|noir'
    r')\b',
    re.IGNORECASE,
)


class PromptDecomposer:
    """
    Decomposes a full art prompt into 3 hierarchical sub-prompts via LLM API.

    Args:
        api_key:  OpenAI API key (falls back to OPENAI_API_KEY env var).
        model:    LLM model name for decomposition.
        clip_version: HF id for the CLIP tokenizer used to count tokens.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o-mini",
        clip_version: str = "openai/clip-vit-large-patch14",
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=api_key) if OpenAI is not None else None
        self.model = model
        self._cache: dict = {}
        self._last_decomposed: dict = {}  # last result for UI display

        # Real CLIP tokenizer for accurate budget enforcement.
        # Lazy-load lightweight tokenizer (no full model weights).
        self._clip_tokenizer = None
        self._clip_version = clip_version

        if OpenAI is None:
            logger.warning(
                "openai package is not installed; PromptDecomposer will use fallback decomposition."
            )

    # ── Tokenizer helpers ─────────────────────────────────────────

    def _get_clip_tokenizer(self):
        if self._clip_tokenizer is None:
            try:
                from transformers import CLIPTokenizer
                self._clip_tokenizer = CLIPTokenizer.from_pretrained(
                    self._clip_version, clean_up_tokenization_spaces=True
                )
            except Exception as e:
                logger.warning("Failed to load CLIP tokenizer (%s); fallback to word count.", e)
                self._clip_tokenizer = False  # sentinel: disabled
        return self._clip_tokenizer

    def _count_clip_tokens(self, text: str) -> int:
        """Return real CLIP token count (including special tokens) or word proxy."""
        tok = self._get_clip_tokenizer()
        if not tok:
            # Rough proxy: ~1.3 tokens/word for English keyword lists
            return int(len(text.split()) * 1.3)
        return len(tok(text, add_special_tokens=True, truncation=False).input_ids)

    def _truncate_to_clip(self, text: str, budget: int) -> str:
        """
        Hard-truncate text to fit within `budget` CLIP tokens.
        Prefers cutting on comma boundaries to preserve phrase integrity.
        """
        tok = self._get_clip_tokenizer()
        if not tok:
            # fallback: word-based approximate cut
            words = text.split()
            approx = int(budget / 1.3)
            return " ".join(words[:approx])

        ids = tok(text, add_special_tokens=True, truncation=False).input_ids
        if len(ids) <= budget:
            return text

        # Try to cut at last comma that keeps us under budget
        parts = [p.strip() for p in text.split(",") if p.strip()]
        kept = []
        for p in parts:
            candidate = ", ".join(kept + [p])
            if len(tok(candidate, add_special_tokens=True, truncation=False).input_ids) > budget:
                break
            kept.append(p)
        if kept:
            return ", ".join(kept)
        # No comma boundaries fit — fall back to hard token slice + decode
        truncated_ids = ids[:budget]
        try:
            return tok.decode(truncated_ids, skip_special_tokens=True)
        except Exception:
            return text  # give up; caller will still be fine since CLIPEmbedder truncates again

    # ── Public API ────────────────────────────────────────────────

    def decompose(
        self,
        caption: str,
        art_style: str,
        PoA: list,
        max_retries: int = 2,
    ) -> dict:
        """
        Decompose a single prompt into 3 hierarchical variants.

        Returns:
            {"prompt1": str, "prompt2": str, "prompt3": str}
        """
        cache_key = (caption, art_style, tuple(PoA))
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.client is None:
            result = self._fallback(caption, art_style, PoA)
            result = self._enforce_budgets(result)
            self._last_decomposed = result
            self._cache[cache_key] = result
            return result

        user_input = self._format_user_input(caption, art_style, PoA)

        result = None
        for attempt in range(max_retries + 1):
            stricter = ""
            if attempt > 0:
                stricter = (
                    f" STRICT: prompt1 <= {P1_TOKEN_BUDGET} CLIP tokens, "
                    f"prompt2 <= {P2_TOKEN_BUDGET} CLIP tokens. "
                    "Use comma-separated keyword phrases. "
                    "Absolutely NO style/art-movement/material/mood words in prompt1 or prompt2."
                )
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT + stricter},
                        {"role": "user", "content": user_input},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=500,
                )
                result = json.loads(response.choices[0].message.content)
            except Exception as e:
                logger.warning("LLM API call failed (attempt %d): %s", attempt, e)
                continue

            p1 = result.get("prompt1", "")
            p2 = result.get("prompt2", "")
            p1_tokens = self._count_clip_tokens(p1)
            p2_tokens = self._count_clip_tokens(p2)
            has_style_p1 = bool(_STYLE_PATTERN.search(p1))
            has_style_p2 = bool(_STYLE_PATTERN.search(p2))

            ok = (
                p1_tokens <= P1_TOKEN_BUDGET
                and p2_tokens <= P2_TOKEN_BUDGET
                and not has_style_p1
                and not has_style_p2
            )
            if ok:
                break

            logger.info(
                "Retry %d: p1=%d tok (style=%s) / p2=%d tok (style=%s)",
                attempt, p1_tokens, has_style_p1, p2_tokens, has_style_p2,
            )

        if result is None:
            logger.warning("All LLM attempts failed, using fallback decomposition")
            result = self._fallback(caption, art_style, PoA)

        # Final safety: even if validation still failed, hard-truncate to fit CLIP.
        result = self._enforce_budgets(result)

        self._last_decomposed = result
        self._cache[cache_key] = result
        return result

    def decompose_batch(
        self,
        captions: list,
        art_styles: list,
        PoAs: list,
    ) -> list:
        """Decompose a batch of prompts. Each sample = 1 API call (cached)."""
        return [
            self.decompose(c, s, p)
            for c, s, p in zip(captions, art_styles, PoAs)
        ]

    # ── Internal ──────────────────────────────────────────────────

    def _enforce_budgets(self, result: dict) -> dict:
        """Hard-truncate P1/P2 to their CLIP token budgets. P3 left untouched."""
        p1 = result.get("prompt1", "") or ""
        p2 = result.get("prompt2", "") or ""
        p3 = result.get("prompt3", "") or ""
        p1_cut = self._truncate_to_clip(p1, P1_TOKEN_BUDGET)
        p2_cut = self._truncate_to_clip(p2, P2_TOKEN_BUDGET)
        if p1_cut != p1:
            logger.info("Truncated prompt1 to fit %d CLIP tokens.", P1_TOKEN_BUDGET)
        if p2_cut != p2:
            logger.info("Truncated prompt2 to fit %d CLIP tokens.", P2_TOKEN_BUDGET)
        return {"prompt1": p1_cut, "prompt2": p2_cut, "prompt3": p3}

    @staticmethod
    def _format_user_input(caption: str, art_style: str, PoA: list) -> str:
        principles = "; ".join(p for p in PoA if p)
        return (
            f"Content: {caption}\n"
            f"Style: {art_style}\n"
            f"Principles: {principles}"
        )

    @staticmethod
    def _fallback(caption: str, art_style: str, PoA: list) -> dict:
        """Minimal fallback when LLM is unavailable."""
        return {
            "prompt1": caption,
            "prompt2": caption,
            "prompt3": f"{caption}. {art_style}. " + " ".join(p for p in PoA if p),
        }
