"""Pure helper that biases a fresh-regen prompt away from a rejected one.

Issue 003 of the single-variation-regeneration PRD ("Try Something New").
When the user rejects a generated variation, "Try Something New" must
produce a DEMONSTRABLY different aesthetic — not the same prompt at a
different temperature. This module returns a steering wrapper that the
prompt-generation paths (``BriefGeneratorService.brief_to_prompts`` and
``StagingPipeline.adapt_prompt``) prepend to their existing system content
on the fresh-regen path.

The module is intentionally pure: no LLM client, no Azure SDK, no I/O. The
prompt-generation paths remain the only LLM-call sites; this module only
shapes the text they send. Keeping the wrapper inert lets us unit-test the
diversity logic in isolation and reuse it from any future path that needs
"avoid this prior direction" steering.

Wrapper placement: the steering block is PREPENDED above ``base`` so the
existing JSON-output / format-contract instructions at the END of ``base``
remain the LLM's last-read instruction. Appending after ``base`` would let
the steering text become the latest instruction and risk overriding the
"output JSON only" contract the prompt-generation templates rely on.
"""

from __future__ import annotations

from typing import Optional


def build_diversifying_prompt(
    rejected_prompt: Optional[str],
    base: str,
    room_analysis: str,
) -> str:
    """Return ``base`` (unchanged) when there is no rejected prompt; otherwise
    prepend a fenced steering block citing ``rejected_prompt`` as negative
    context and including ``room_analysis`` as a grounding anchor.

    Parameters
    ----------
    rejected_prompt:
        The previously-rejected adapted prompt (from the variation's
        ``generation_metadata.adapted_prompt``). ``None``, ``""``, and
        whitespace-only strings are treated identically: the function
        returns ``base`` unchanged so callers can pass through the prior
        prompt without a guard.
    base:
        The full system prompt the prompt-generation path would otherwise
        send. This is wrapped, not replaced — placement / preserve / palette
        / output-format instructions inside ``base`` survive intact.
    room_analysis:
        Short description of the room being regenerated. Included in the
        steering text so the LLM has a concrete anchor for "meaningfully
        different but plausible for THIS room."
    """
    # Treat None / empty / whitespace-only identically: no steering needed,
    # caller likely on a non-regen first-time path.
    if rejected_prompt is None or not rejected_prompt.strip():
        return base

    # Prepend the steering block. Note the explicit "do not repeat" / "depart
    # from" framing — without it the LLM tends to read the rejected text as a
    # live instruction rather than negative context.
    steering = (
        "=== REGENERATION STEERING (read first, then follow the brief below) ===\n"
        "The user previously rejected the following aesthetic for this room and "
        "asked for something visually different (\"Try Something New\"). "
        "Treat the text inside the fence below as a REJECTED PRIOR DIRECTION — "
        "do NOT repeat its style, palette, materials, mood, or composition.\n"
        "<<<REJECTED_PRIOR_DIRECTION\n"
        f"{rejected_prompt}\n"
        "REJECTED_PRIOR_DIRECTION>>>\n"
        "Take a meaningfully different visual direction the user is likely to "
        "notice on first look — vary at least the styling and color palette, "
        "and ideally also the materials, mood, or composition. Stay plausible "
        f"for this room: {room_analysis}\n"
        "=== END STEERING — the regular brief follows ===\n\n"
    )
    return steering + base
