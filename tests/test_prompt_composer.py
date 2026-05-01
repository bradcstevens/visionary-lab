"""Tests for ``PromptComposer.compose``.

The composer is the single source of truth for assembling the final
``adapted_prompt`` that goes to the image-generation model. It encapsulates
the precedence rules the staging pipeline (and slice 004's edit-prompt
endpoint) need to honor consistently across multiple call sites:

    1. ``variation_override`` (when non-empty) is the BASE — used by the
       Edit Prompt path (slice 004).
    2. Otherwise, ``design_brief`` (when non-empty) is the BASE — typical
       case where the per-variation prompt came from
       ``brief_to_prompts`` or ``adapt_prompt``.
    3. Otherwise, ``project_prompt`` is the BASE — fallback for legacy
       projects without a brief.

Once a base is chosen, the room ``room_addendum`` (when non-empty) is
ALWAYS appended with a paragraph break separator. This matches the PRD's
Edit Prompt semantic: a user-typed prompt still respects the room's
addendum constraint.

Whitespace and ``None``/empty inputs are handled cleanly at every layer.
"""

from typing import Optional

import pytest

from backend.core.prompt_composer import PromptComposer


# ----------------------------------------------------------------------
# Base selection — which input is chosen as the base before addendum
# ----------------------------------------------------------------------

class TestBaseSelection:
    def test_project_only_returns_project_prompt(self):
        out = PromptComposer.compose(
            project_prompt="modern minimalist",
            design_brief=None,
            room_addendum=None,
        )
        assert out == "modern minimalist"

    def test_project_plus_brief_uses_brief_as_base(self):
        out = PromptComposer.compose(
            project_prompt="modern minimalist",
            design_brief="Add Adirondack chairs and warm lighting",
            room_addendum=None,
        )
        assert out == "Add Adirondack chairs and warm lighting"

    def test_variation_override_wins_over_brief_for_base(self):
        out = PromptComposer.compose(
            project_prompt="modern minimalist",
            design_brief="brief-generated prompt",
            room_addendum=None,
            variation_override="user-typed prompt",
        )
        assert out == "user-typed prompt"

    def test_variation_override_wins_over_project_when_no_brief(self):
        out = PromptComposer.compose(
            project_prompt="modern minimalist",
            design_brief=None,
            room_addendum=None,
            variation_override="user-typed prompt",
        )
        assert out == "user-typed prompt"


# ----------------------------------------------------------------------
# Addendum is layered on top of WHATEVER base was chosen
# ----------------------------------------------------------------------

class TestAddendumLayering:
    def test_brief_plus_addendum_appended_with_paragraph_break(self):
        out = PromptComposer.compose(
            project_prompt="modern",
            design_brief="Add chairs",
            room_addendum="always in front of the fence",
        )
        assert out == "Add chairs\n\nalways in front of the fence"

    def test_project_plus_addendum_when_no_brief(self):
        out = PromptComposer.compose(
            project_prompt="modern minimalist",
            design_brief=None,
            room_addendum="never include lavender",
        )
        assert out == "modern minimalist\n\nnever include lavender"

    def test_variation_override_plus_addendum_layered(self):
        """Edit Prompt semantic per PRD: user-typed prompt still
        respects the room's addendum constraint. Override beats
        brief/project for BASE but addendum still appends."""
        out = PromptComposer.compose(
            project_prompt="modern",
            design_brief="brief",
            room_addendum="always preserve the pergola",
            variation_override="ignore everything else",
        )
        assert out == "ignore everything else\n\nalways preserve the pergola"

    def test_addendum_only_returns_addendum_alone_when_base_empty(self):
        out = PromptComposer.compose(
            project_prompt="",
            design_brief=None,
            room_addendum="this is all I have",
        )
        assert out == "this is all I have"


# ----------------------------------------------------------------------
# Empty / None / whitespace handling — clean at every layer
# ----------------------------------------------------------------------

class TestEmptyAndNoneHandling:
    def test_all_none_returns_empty_string(self):
        out = PromptComposer.compose(
            project_prompt="",
            design_brief=None,
            room_addendum=None,
        )
        assert out == ""

    def test_none_brief_falls_back_to_project_prompt(self):
        out = PromptComposer.compose(
            project_prompt="modern",
            design_brief=None,
            room_addendum=None,
        )
        assert out == "modern"

    def test_empty_string_brief_falls_back_to_project_prompt(self):
        out = PromptComposer.compose(
            project_prompt="modern",
            design_brief="",
            room_addendum=None,
        )
        assert out == "modern"

    def test_whitespace_only_brief_falls_back_to_project_prompt(self):
        out = PromptComposer.compose(
            project_prompt="modern",
            design_brief="   \n\t ",
            room_addendum=None,
        )
        assert out == "modern"

    def test_whitespace_only_addendum_ignored(self):
        out = PromptComposer.compose(
            project_prompt="modern",
            design_brief="brief text",
            room_addendum="   \n\t  ",
        )
        assert out == "brief text"

    def test_whitespace_only_override_falls_back_to_brief(self):
        out = PromptComposer.compose(
            project_prompt="modern",
            design_brief="brief text",
            room_addendum=None,
            variation_override="   \t ",
        )
        assert out == "brief text"

    def test_none_override_default_uses_brief(self):
        out = PromptComposer.compose(
            project_prompt="modern",
            design_brief="brief text",
            room_addendum=None,
        )
        assert out == "brief text"


# ----------------------------------------------------------------------
# Whitespace handling — base + addendum are stripped before joining
# ----------------------------------------------------------------------

class TestWhitespaceStripping:
    def test_brief_leading_trailing_whitespace_stripped_before_compose(self):
        out = PromptComposer.compose(
            project_prompt="",
            design_brief="  brief text  \n",
            room_addendum=None,
        )
        assert out == "brief text"

    def test_addendum_leading_trailing_whitespace_stripped_before_append(self):
        out = PromptComposer.compose(
            project_prompt="",
            design_brief="brief",
            room_addendum="\n  always upright  \n",
        )
        assert out == "brief\n\nalways upright"

    def test_override_stripped_before_use_as_base(self):
        out = PromptComposer.compose(
            project_prompt="",
            design_brief="brief",
            room_addendum="addendum",
            variation_override="\n  user prompt  \n",
        )
        assert out == "user prompt\n\naddendum"


# ----------------------------------------------------------------------
# Multi-line content preserved verbatim once chosen
# ----------------------------------------------------------------------

class TestMultilineHandling:
    def test_multiline_addendum_preserved_in_body(self):
        addendum = "line one\nline two\nline three"
        out = PromptComposer.compose(
            project_prompt="",
            design_brief="brief",
            room_addendum=addendum,
        )
        assert out == "brief\n\nline one\nline two\nline three"

    def test_multiline_brief_preserved(self):
        brief = "line A\nline B"
        out = PromptComposer.compose(
            project_prompt="",
            design_brief=brief,
            room_addendum="constraint",
        )
        assert out == "line A\nline B\n\nconstraint"


# ----------------------------------------------------------------------
# Purity — calling compose twice with the same inputs returns the same
# value and does not mutate the inputs
# ----------------------------------------------------------------------

class TestPurity:
    def test_idempotent_for_same_inputs(self):
        kwargs = dict(
            project_prompt="modern",
            design_brief="brief content",
            room_addendum="constraint",
            variation_override=None,
        )
        first = PromptComposer.compose(**kwargs)
        second = PromptComposer.compose(**kwargs)
        assert first == second
