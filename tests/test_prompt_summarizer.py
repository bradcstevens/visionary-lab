"""Tests for ``PromptSummarizer`` (issue 013 of image-pipeline-and-
project-ux-overhaul PRD).

Contract:
    async summarize(prompt: str) -> str

- Returns ≤240 characters.
- Calls the injected ``async_llm_client.chat.completions.create`` only
  when the prompt is longer than 240 characters (short prompts are
  already valid summaries — pass-through is cheaper and deterministic).
- On any LLM exception OR a whitespace/empty LLM response, falls back
  to deterministic truncation (word-boundary if possible, else hard
  cut, with a single trailing "…" character; final length still ≤240).
- Empty / whitespace-only input → "".
- LLM-returned summaries longer than 240 are normalized via the same
  truncation fallback so the contract holds even if the model ignores
  the prompt's length cap.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.core.prompt_summarizer import PromptSummarizer


def _make_client(content: str | None = None, raises: Exception | None = None):
    """Build a minimal AsyncAzureOpenAI-shaped mock that returns
    ``content`` (or raises) when ``chat.completions.create`` is awaited."""
    client = MagicMock()
    if raises is not None:
        client.chat.completions.create = AsyncMock(side_effect=raises)
    else:
        msg = MagicMock()
        msg.message.content = content
        resp = MagicMock()
        resp.choices = [msg]
        client.chat.completions.create = AsyncMock(return_value=resp)
    return client


# ---- happy path --------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_long_prompt_returns_llm_summary():
    long_prompt = "modern minimalist living room with " * 20  # ~700 chars
    summary = "Modern minimalist living room with warm wood and greenery."
    client = _make_client(content=summary)
    summarizer = PromptSummarizer(async_llm_client=client, llm_deployment="gpt-x")

    out = await summarizer.summarize(long_prompt)

    assert out == summary
    assert len(out) <= 240
    client.chat.completions.create.assert_awaited_once()


# ---- short-input pass-through (no LLM call) ----------------------------


@pytest.mark.asyncio
async def test_summarize_short_prompt_passthrough_no_llm_call():
    """Pass-through optimization: a prompt already ≤240 chars IS its
    own summary. The LLM call would be wasted RU and slow down PATCH
    on the common case (most users type a few sentences)."""
    short = "warm wood and lots of greenery"
    client = _make_client(content="should not be used")
    summarizer = PromptSummarizer(async_llm_client=client, llm_deployment="gpt-x")

    out = await summarizer.summarize(short)

    assert out == short
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_summarize_exactly_240_chars_passthrough():
    """Boundary: exactly 240 chars is a valid summary by definition."""
    text = "x" * 240
    client = _make_client(content="ignored")
    summarizer = PromptSummarizer(async_llm_client=client, llm_deployment="gpt-x")
    out = await summarizer.summarize(text)
    assert out == text
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_summarize_strips_whitespace_for_passthrough_decision():
    """A prompt that's >240 only because of trailing whitespace should
    pass through after strip() — no LLM, no truncation."""
    text = "warm wood" + " " * 300
    client = _make_client(content="ignored")
    summarizer = PromptSummarizer(async_llm_client=client, llm_deployment="gpt-x")
    out = await summarizer.summarize(text)
    assert out == "warm wood"
    client.chat.completions.create.assert_not_awaited()


# ---- empty input -------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_empty_string_returns_empty():
    client = _make_client(content="ignored")
    summarizer = PromptSummarizer(async_llm_client=client, llm_deployment="gpt-x")
    assert await summarizer.summarize("") == ""
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_summarize_whitespace_only_returns_empty():
    client = _make_client(content="ignored")
    summarizer = PromptSummarizer(async_llm_client=client, llm_deployment="gpt-x")
    assert await summarizer.summarize("   \n\t  ") == ""
    client.chat.completions.create.assert_not_awaited()


# ---- LLM failure → truncation fallback ---------------------------------


@pytest.mark.asyncio
async def test_summarize_llm_raises_uses_truncation_fallback():
    """The PRD's AC: deterministic truncation fallback when the LLM
    client is unavailable. Pinned by simulating a transport-layer
    failure mid-call."""
    long_prompt = "warm wood and lots of greenery " * 20  # ~620 chars
    client = _make_client(raises=RuntimeError("LLM unavailable"))
    summarizer = PromptSummarizer(async_llm_client=client, llm_deployment="gpt-x")

    out = await summarizer.summarize(long_prompt)

    assert len(out) <= 240
    assert out.endswith("\u2026")  # ellipsis marker
    # Word boundary preserved — last char before ellipsis must not be
    # mid-word for our test phrase.
    assert " " not in out[-3:-1] or out[-2] != " "  # not whitespace-then-ellipsis


@pytest.mark.asyncio
async def test_summarize_llm_returns_empty_string_uses_fallback():
    """LLM occasionally returns an empty string (safety filter, parse
    failure). Same fallback as a raised exception."""
    long_prompt = "modern minimalist " * 30
    client = _make_client(content="")
    summarizer = PromptSummarizer(async_llm_client=client, llm_deployment="gpt-x")
    out = await summarizer.summarize(long_prompt)
    assert len(out) <= 240
    assert out.endswith("\u2026")


@pytest.mark.asyncio
async def test_summarize_llm_returns_whitespace_uses_fallback():
    long_prompt = "modern minimalist " * 30
    client = _make_client(content="   \n  ")
    summarizer = PromptSummarizer(async_llm_client=client, llm_deployment="gpt-x")
    out = await summarizer.summarize(long_prompt)
    assert len(out) <= 240
    assert out.endswith("\u2026")


# ---- LLM overshoot normalized -----------------------------------------


@pytest.mark.asyncio
async def test_summarize_llm_overshoots_240_is_normalized():
    """Even when the LLM ignores its 240-char instruction, the output
    is normalized via the same truncation path so the ≤240 contract
    holds for callers."""
    long_prompt = "modern minimalist " * 30
    overshoot = "A " + ("very " * 100) + "long summary."  # >240
    assert len(overshoot) > 240
    client = _make_client(content=overshoot)
    summarizer = PromptSummarizer(async_llm_client=client, llm_deployment="gpt-x")
    out = await summarizer.summarize(long_prompt)
    assert len(out) <= 240
    assert out.endswith("\u2026")


# ---- truncate_to_summary helper directly -------------------------------


def test_truncate_to_summary_word_boundary():
    """Word-boundary cut: must not slice mid-word when a space exists
    in the truncation window."""
    from backend.core.prompt_summarizer import truncate_to_summary

    text = "the quick brown fox " * 20  # 400 chars
    out = truncate_to_summary(text)
    assert len(out) <= 240
    assert out.endswith("\u2026")
    # The character right before the ellipsis must be a complete word
    # character — i.e. the last word in the output is not chopped.
    body = out[:-1].rstrip()
    assert body.endswith(("fox", "the", "quick", "brown"))


def test_truncate_to_summary_hard_cut_when_no_space_in_window():
    """Defensive: if the input is one giant unbroken token (e.g. a
    base64 blob the user accidentally pasted), hard-cut at 239 + 1
    char ellipsis = 240 total. No infinite-loop, no IndexError."""
    from backend.core.prompt_summarizer import truncate_to_summary

    text = "x" * 1000
    out = truncate_to_summary(text)
    assert len(out) == 240
    assert out.endswith("\u2026")
    assert out[:-1] == "x" * 239


def test_truncate_to_summary_short_input_unchanged():
    from backend.core.prompt_summarizer import truncate_to_summary

    assert truncate_to_summary("hello world") == "hello world"
    assert truncate_to_summary("") == ""
    assert truncate_to_summary("   x   ") == "x"
