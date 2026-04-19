"""
Prompt Decomposer — Phân rã prompt thành 3 cấp bậc qua LLM API.

Workflow:
  1 lần gọi API duy nhất → JSON {"prompt1", "prompt2", "prompt3"}
  - prompt1 (≤30 words): Spatial layout WITH key nouns, NO style
  - prompt2 (≤50 words): Content — full objects/details, NO style
  - prompt3 (≤100 words): Full — content + style + artistic principles

Word limits đảm bảo P1/P2 nằm trong giới hạn 77 tokens của CLIP.
"""
import os
import re
import json
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an art prompt decomposer. From the given content + style + principles,
return a JSON object with exactly 3 keys in ONE response:

"prompt1" (Spatial Layout & Minimal Structure, ≤ 30 words):
  - Treat this as a rough geometric sketch or 3D gray-box blockout.
  - KEEP the core subject nouns but STRIP AWAY all descriptive adjectives, materials (metal, glass), and colors.
  - Describe ONLY the basic shapes, silhouettes, and WHERE they are positioned spatially.
  - Include any compositional principles (e.g. radial balance, rule of thirds).
  - Absolutely NO colors (e.g. blue, red), textures, glowing effects, lighting, or style words.
  - Good: "A large circular machine centered in the frame, with straight lines radiating outward symmetrically from the middle."
  - Bad: "A glowing blue metal jet turbine...", "vibrant colors...", "impressionist..."

"prompt2" (Content, ≤ 50 words):
  - Expand prompt1 with MORE detail about the subjects and scene.
  - Add descriptive adjectives, actions, and secondary objects.
  - Still NO style words, art movements, textures, or aesthetic references.
  - Good: "A glowing high-tech spaceship hovering just above an old stone courtyard, engine exhaust blasting downward."

"prompt3" (Full, ≤ 100 words):
  - Complete version: prompt2 content + art style + artistic principles, vivid and clean.

Return ONLY: {"prompt1": "...", "prompt2": "...", "prompt3": "..."}
No markdown, no extra keys.\
"""

# Style words that should NOT appear in prompt1 or prompt2
_STYLE_PATTERN = re.compile(
    r'\b(baroque|renaissance|impressionism|cubism|surrealism|pop art|'
    r'art nouveau|ukiyo-e|watercolor|oil painting|brushstroke|impasto|'
    r'chiaroscuro|vibrant|muted|pastel|dramatic lighting|ethereal|'
    r'dreamlike|atmospheric|moody|stylized)\b',
    re.IGNORECASE,
)


class PromptDecomposer:
    """
    Decomposes a full art prompt into 3 hierarchical sub-prompts via LLM API.

    Args:
        api_key:  OpenAI API key (falls back to OPENAI_API_KEY env var).
        model:    LLM model name for decomposition.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
        self.model = model
        self._cache: dict = {}
        self._last_decomposed: dict = {}  # last result for UI display

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

        user_input = self._format_user_input(caption, art_style, PoA)

        result = None
        for attempt in range(max_retries + 1):
            stricter = (
                " BE EXTREMELY STRICT: prompt1 must have ≤30 words. "
                "Keep subject nouns but absolutely NO style/art-movement words."
                if attempt > 0
                else ""
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
                    max_tokens=400,
                )
                result = json.loads(response.choices[0].message.content)
            except Exception as e:
                logger.warning("LLM API call failed (attempt %d): %s", attempt, e)
                continue

            # Validate: prompt1 must be short + no style words
            p1 = result.get("prompt1", "")
            p1_word_count = len(p1.split())
            has_style = bool(_STYLE_PATTERN.search(p1))

            if p1_word_count <= 35 and not has_style:
                break  # valid result

            logger.info(
                "Retry %d: prompt1 has %d words / style_words=%s",
                attempt, p1_word_count, has_style,
            )

        if result is None:
            # Fallback: use raw inputs directly
            logger.warning("All LLM attempts failed, using fallback decomposition")
            result = self._fallback(caption, art_style, PoA)

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
        # P1 fallback: use caption directly (has nouns, no style)
        return {
            "prompt1": caption,
            "prompt2": caption,
            "prompt3": f"{caption}. {art_style}. " + " ".join(p for p in PoA if p),
        }
