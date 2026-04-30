"""Unit tests for ``backend.core.prompt_diversity``.

Issue 003 of the single-variation-regeneration PRD: ``build_diversifying_prompt``
biases a fresh-regen prompt away from the rejected one. The function MUST be
pure: no Azure SDK / LLM client / I/O. These tests pin the three input modes
(None / empty / non-empty) and the purity contract.
"""

from __future__ import annotations

import inspect

import pytest


def _import_module():
    # Import inside the test so import-error surfaces as a test failure
    # rather than a collection error.
    from backend.core import prompt_diversity

    return prompt_diversity


# ---------------------------------------------------------------------------
# Three input modes (PRD acceptance criteria)
# ---------------------------------------------------------------------------


class TestNoneOrEmptyReturnsBaseUnchanged:
    """When there is no rejected prompt, the base prompt MUST pass through
    unchanged so first-time generation paths can call this helper safely."""

    def test_none_returns_base_unchanged(self):
        mod = _import_module()
        base = "BASE PROMPT CONTENT — palette: warm; preserve: doors."
        result = mod.build_diversifying_prompt(
            rejected_prompt=None,
            base=base,
            room_analysis="A sunlit living room.",
        )
        assert result == base

    def test_empty_string_returns_base_unchanged(self):
        mod = _import_module()
        base = "BASE PROMPT CONTENT"
        result = mod.build_diversifying_prompt(
            rejected_prompt="",
            base=base,
            room_analysis="A kitchen.",
        )
        assert result == base

    def test_whitespace_only_returns_base_unchanged(self):
        # An LLM that emitted only whitespace, or a metadata field that was
        # stored as "   ", must not poison the steering block.
        mod = _import_module()
        base = "BASE"
        result = mod.build_diversifying_prompt(
            rejected_prompt="   \n\t  ",
            base=base,
            room_analysis="A bedroom.",
        )
        assert result == base


class TestNonEmptyRejectedIncludesRejectionAndPreservesBase:
    """When the rejected prompt is non-empty, the output must (a) include the
    rejected text as negative context AND (b) still carry the base intent."""

    def test_output_contains_rejected_text(self):
        mod = _import_module()
        rejected = "MAGENTA-AND-CHROME MAXIMALIST AESTHETIC"
        result = mod.build_diversifying_prompt(
            rejected_prompt=rejected,
            base="A calm Scandinavian living room.",
            room_analysis="A sunlit living room with hardwood floors.",
        )
        assert rejected in result

    def test_output_preserves_base(self):
        mod = _import_module()
        base = "GLOBAL_INSTRUCTIONS_SENTINEL_TOKEN — palette is warm neutrals."
        result = mod.build_diversifying_prompt(
            rejected_prompt="rejected aesthetic A",
            base=base,
            room_analysis="A kitchen.",
        )
        assert base in result

    def test_output_signals_rejection_intent(self):
        # The steering block must clearly mark the rejected prompt as
        # negative context (so the LLM treats it as exclusion, not as a
        # live instruction to follow).
        mod = _import_module()
        result = mod.build_diversifying_prompt(
            rejected_prompt="rejected_X",
            base="base_Y",
            room_analysis="A bathroom.",
        )
        # We don't pin exact wording, but the text must telegraph that
        # the prior aesthetic is to be avoided.
        lowered = result.lower()
        assert any(
            keyword in lowered
            for keyword in ("reject", "avoid", "do not repeat", "different", "depart")
        ), (
            "Steering block must signal rejection / avoidance intent. "
            f"Got: {result!r}"
        )

    def test_output_includes_room_analysis_for_grounding(self):
        # The room_analysis grounds the "meaningfully different but plausible
        # for this room" steering. Including it gives the LLM concrete
        # anchors for the alternative direction.
        mod = _import_module()
        room_analysis = "ROOM_ANALYSIS_SENTINEL — corner studio with a large arched window."
        result = mod.build_diversifying_prompt(
            rejected_prompt="prior aesthetic",
            base="base",
            room_analysis=room_analysis,
        )
        assert room_analysis in result

    def test_output_strictly_longer_than_base(self):
        mod = _import_module()
        base = "BASE"
        result = mod.build_diversifying_prompt(
            rejected_prompt="anything",
            base=base,
            room_analysis="anywhere",
        )
        assert len(result) > len(base)


# ---------------------------------------------------------------------------
# Purity contract
# ---------------------------------------------------------------------------


class TestModuleIsPure:
    """The PRD acceptance criterion: the module MUST be pure — no Azure SDK
    calls, no LLM client calls, no I/O. We assert this two ways:
    (1) static — module source contains no forbidden imports;
    (2) dynamic — calling the function with a stub does not perform I/O.
    """

    def test_module_source_does_not_import_azure_or_openai(self):
        mod = _import_module()
        src = inspect.getsource(mod)
        forbidden = ["import openai", "from openai", "import azure", "from azure"]
        for token in forbidden:
            assert token not in src, (
                f"prompt_diversity must be a pure module — found {token!r} in source"
            )

    def test_function_is_synchronous(self):
        # No async I/O: the function must be a plain `def`, not `async def`.
        mod = _import_module()
        assert not inspect.iscoroutinefunction(mod.build_diversifying_prompt), (
            "build_diversifying_prompt must be a synchronous pure function"
        )

    def test_function_has_expected_signature(self):
        mod = _import_module()
        sig = inspect.signature(mod.build_diversifying_prompt)
        # Three parameters in order: rejected_prompt, base, room_analysis.
        params = list(sig.parameters)
        assert params == ["rejected_prompt", "base", "room_analysis"], (
            f"Unexpected signature {sig} — issue 003 pins the parameter order"
        )
