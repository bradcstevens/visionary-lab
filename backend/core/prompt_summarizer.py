"""PromptSummarizer — produces a ≤240-char summary of a project prompt.

Issue 013 of the image-pipeline-and-project-ux-overhaul PRD. Used by
``PATCH /api/v1/staging/projects/{id}`` so the project page can render
a collapsed-summary view of the user's full prompt without paying a
round-trip on every read.

Contract
========

``async summarize(prompt: str) -> str``

- Returns ≤ ``MAX_SUMMARY_LEN`` (240) characters.
- Pass-through when the input is already short enough — no LLM call,
  deterministic, and free. The frontend's collapsed-prompt view only
  needs a summary when the prompt is genuinely long.
- Calls the injected LLM client otherwise.
- On any LLM exception OR a whitespace/empty response, falls back to
  deterministic truncation (word-boundary preferred, hard cut otherwise,
  with a single trailing "…"). The PRD's AC pins this:
  "deterministic truncation fallback when LLM client is unavailable".
- LLM-returned summaries longer than 240 are normalized via the same
  truncation path so the contract holds even if the model overshoots.
- Empty / whitespace-only input → "" (no call, no fallback noise).

Deep module: one public method, ~one screen of implementation, hides
the entire LLM-summarize-or-truncate decision tree behind a single
``await summarizer.summarize(prompt)`` call so the endpoint code stays
a one-liner.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


MAX_SUMMARY_LEN = 240
"""Hard upper bound on the returned summary length, in characters.
Mirrors the PRD's "≤240-char summary" contract. Includes any trailing
ellipsis from the truncation fallback."""


SUMMARIZE_SYSTEM_PROMPT = (
    "You are a copy editor. Summarize the user's image-generation prompt "
    "in at most 240 characters. Keep it a single sentence in plain prose, "
    "no markdown, no quotes, no preamble. Preserve the most concrete "
    "design choices (style, colors, materials, key objects)."
)


def truncate_to_summary(text: str, *, max_len: int = MAX_SUMMARY_LEN) -> str:
    """Deterministic truncation fallback. Public for direct use by the
    PATCH endpoint when normalizing client-supplied summaries.

    Algorithm:

    - Strip surrounding whitespace (the LLM occasionally emits leading
      spaces or trailing newlines; we never want those persisted).
    - If ``len(stripped) <= max_len`` → return stripped (no ellipsis;
      this is already a valid summary).
    - Otherwise reserve 1 char for the trailing "…" (U+2026) and look
      for the last whitespace within ``[0, max_len-1)`` so we cut at a
      word boundary. If no whitespace exists in that window (e.g. one
      enormous unbroken token like a hash), hard-cut at ``max_len-1``.
    """
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= max_len:
        return stripped

    # Reserve the last char for the ellipsis; search for a word boundary
    # in the prefix so we don't slice mid-word when avoidable.
    cut_window = max_len - 1
    boundary = stripped.rfind(" ", 0, cut_window)
    cut_at = boundary if boundary > 0 else cut_window
    return stripped[:cut_at].rstrip() + "\u2026"


class PromptSummarizer:
    """LLM-backed prompt summarizer with deterministic truncation
    fallback. See module docstring for the full contract."""

    def __init__(self, async_llm_client, llm_deployment: str):
        self.async_llm_client = async_llm_client
        self.llm_deployment = llm_deployment

    async def summarize(self, prompt: str) -> str:
        if prompt is None:
            return ""
        stripped = prompt.strip()
        if not stripped:
            return ""

        # Pass-through: already a valid summary, no LLM call needed.
        if len(stripped) <= MAX_SUMMARY_LEN:
            return stripped

        try:
            response = await self.async_llm_client.chat.completions.create(
                model=self.llm_deployment,
                messages=[
                    {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                    {"role": "user", "content": stripped},
                ],
                temperature=0.2,
                max_tokens=200,
            )
            content = response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001 — fallback is the contract
            logger.warning(
                "PromptSummarizer LLM call failed; using truncation fallback: %s",
                exc,
            )
            return truncate_to_summary(stripped)

        if content is None or not content.strip():
            logger.warning(
                "PromptSummarizer LLM returned empty content; using truncation fallback"
            )
            return truncate_to_summary(stripped)

        # Normalize even valid-looking LLM output so the ≤240 contract
        # holds even when the model ignores the instruction.
        return truncate_to_summary(content)
