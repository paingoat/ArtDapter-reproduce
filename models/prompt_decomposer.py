"""
Prompt Decomposer — Phân rã prompt thành 3 cấp bậc qua LLM API.

Workflow:
  1 lần gọi API duy nhất → JSON {"prompt1", "prompt2", "prompt3"}
  - prompt1 (≤30 words): Layout only — spatial composition, no objects/style
  - prompt2 (≤50 words): Content — layout + objects, no style
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

"prompt1" (Layout ONLY, STRICT ≤ 30 words):
  - Describe ONLY the spatial composition and scene structure.
  - Absolutely NO specific objects, named subjects, colors, or style words.
  - Good: "Central vertical form on left, open horizontal space on right, low horizon."
  - Bad: "A man stands..." or "Pop Art style..."

"prompt2" (Content, ≤ 50 words):
  - Start from prompt1 structure, then ADD specific subjects/objects.
  - Still NO style, art movement, texture, or aesthetic references.
  - Good: "A man with a gun stands left, empty street extends right."

"prompt3" (Full, ≤ 100 words):
  - Complete version: content + style + artistic principles, vivid and clean.

Return ONLY: {"prompt1": "...", "prompt2": "...", "prompt3": "..."}
No markdown, no extra keys.\
"""

# Words that should NOT appear in prompt1 (layout-only)
_OBJECT_PATTERN = re.compile(
    r'\b(man|woman|person|people|child|dog|cat|horse|bird|fish|tree|flower|'
    r'building|house|car|boat|ship|figure|portrait|landscape|'
    r'standing|sitting|holding|wearing|walking|running|flying)\b',
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
                " BE EXTREMELY STRICT: prompt1 must have ≤30 words and "
                "ZERO object/subject nouns."
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

            # Validate: prompt1 must be short and layout-only
            p1 = result.get("prompt1", "")
            p1_word_count = len(p1.split())
            has_objects = bool(_OBJECT_PATTERN.search(p1))

            if p1_word_count <= 35 and not has_objects:
                break  # valid result

            logger.info(
                "Retry %d: prompt1 has %d words / objects=%s",
                attempt, p1_word_count, has_objects,
            )

        if result is None:
            # Fallback: use raw inputs directly
            logger.warning("All LLM attempts failed, using fallback decomposition")
            result = self._fallback(caption, art_style, PoA)

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
        return {
            "prompt1": "A balanced composition with distinct foreground and background areas.",
            "prompt2": caption,
            "prompt3": f"{caption} {art_style}. " + " ".join(p for p in PoA if p),
        }
