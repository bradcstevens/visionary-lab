"""Tests for ``BriefSectionRegistry`` and the ``PromptComposer`` brief
extensions shipped in issue 015 of the image-pipeline-and-project-ux-
overhaul PRD.

The registry is the single source of truth for the eight canonical
sections the wizard and the settings panel expose. ``PromptComposer.
compose_brief_markdown`` renders that structured payload to a
deterministic top-level prompt; ``extract_sections`` parses it back so
the lazy-backfill (issue 016) and the settings preview tab can round-trip.
"""

from __future__ import annotations

import pytest

from backend.core.brief_section_registry import (
    SECTIONS,
    BriefSection,
    get_section,
    section_ids,
    title_to_id,
)
from backend.core.prompt_composer import PromptComposer
from backend.models.design_brief import DesignBrief


# --- BriefSectionRegistry -------------------------------------------------


def test_registry_exports_eight_sections_in_pinned_order():
    """The PRD names the eight sections in a specific order. Both the
    rendered markdown and the wizard step sequence follow this order
    verbatim — pin it here so a casual edit can't reorder either."""
    assert section_ids() == (
        "edit_task",
        "edit_zone",
        "do_not_alter",
        "object_palette",
        "arrangement",
        "regional_constraints",
        "aesthetic_goal",
        "scale_fidelity",
    )
    assert len(SECTIONS) == 8


def test_registry_section_metadata_shape():
    """Each section carries an id, a display title, and a one-sentence
    description. All three are non-empty strings — the wizard and the
    settings panel both render the title + description verbatim."""
    for s in SECTIONS:
        assert isinstance(s, BriefSection)
        assert isinstance(s.id, str) and s.id
        assert isinstance(s.title, str) and s.title
        assert isinstance(s.description, str) and s.description


def test_registry_ids_are_unique():
    ids = [s.id for s in SECTIONS]
    assert len(ids) == len(set(ids))


def test_registry_titles_are_unique():
    """Title-to-id lookup would be ambiguous otherwise."""
    titles = [s.title for s in SECTIONS]
    assert len(titles) == len(set(titles))


def test_get_section_returns_section_by_id():
    s = get_section("edit_task")
    assert s.id == "edit_task"
    assert s.title == "Edit Task"


def test_get_section_unknown_id_raises_keyerror():
    """Caller asked for a section that does not exist — that's a
    programming error, not user input."""
    with pytest.raises(KeyError):
        get_section("not_a_real_section")


def test_title_to_id_exact_match():
    assert title_to_id("Edit Task") == "edit_task"
    assert title_to_id("Scale & Fidelity") == "scale_fidelity"


def test_title_to_id_case_and_punctuation_insensitive():
    """Tolerate whitespace, case, and stray punctuation drift in titles
    so persisted briefs survive cosmetic title polish."""
    assert title_to_id("  EDIT TASK  ") == "edit_task"
    assert title_to_id("Do-Not-Alter!") == "do_not_alter"
    assert title_to_id("scale & fidelity") == "scale_fidelity"


def test_title_to_id_unknown_returns_none():
    """Extractors route unknown headings to a free-form bucket without
    raising — pin the contract."""
    assert title_to_id("Random Header") is None
    assert title_to_id("") is None


# --- PromptComposer.compose_brief_markdown --------------------------------


def test_compose_brief_markdown_empty_returns_empty():
    assert PromptComposer.compose_brief_markdown({}) == ""


def test_compose_brief_markdown_renders_sections_in_registry_order():
    """Determinism contract: input dict insertion order is irrelevant;
    output is registry-ordered."""
    out = PromptComposer.compose_brief_markdown(
        {
            # Insert in REVERSE registry order to prove ordering is
            # registry-driven, not insertion-driven.
            "scale_fidelity": "1080p, photoreal",
            "edit_task": "Stage the backyard",
        }
    )
    assert out == (
        "## Edit Task\nStage the backyard\n\n"
        "## Scale & Fidelity\n1080p, photoreal"
    )


def test_compose_brief_markdown_is_deterministic():
    """Same input → byte-identical output across calls."""
    sections = {"edit_task": "A", "arrangement": "B", "aesthetic_goal": "C"}
    assert PromptComposer.compose_brief_markdown(
        sections
    ) == PromptComposer.compose_brief_markdown(sections)


def test_compose_brief_markdown_drops_empty_and_whitespace_sections():
    out = PromptComposer.compose_brief_markdown(
        {
            "edit_task": "Real content",
            "edit_zone": "",
            "do_not_alter": "   \n\t  ",
            "object_palette": None,  # type: ignore[dict-item]
            "arrangement": "More content",
        }
    )
    assert out == (
        "## Edit Task\nReal content\n\n"
        "## Arrangement\nMore content"
    )


def test_compose_brief_markdown_strips_section_body_outer_whitespace():
    """Outer whitespace stripped; internal newlines preserved."""
    out = PromptComposer.compose_brief_markdown(
        {"edit_task": "  line one\nline two  \n"}
    )
    assert out == "## Edit Task\nline one\nline two"


def test_compose_brief_markdown_drops_unknown_section_ids():
    """A future schema where we remove a section from the registry must
    not poison the output with stale content."""
    out = PromptComposer.compose_brief_markdown(
        {"edit_task": "kept", "deprecated_section": "should not render"}
    )
    assert out == "## Edit Task\nkept"
    assert "deprecated_section" not in out
    assert "should not render" not in out


def test_compose_brief_markdown_raw_override_wins_when_set():
    """PRD AC: when raw_override is set, composer returns it unchanged."""
    out = PromptComposer.compose_brief_markdown(
        {"edit_task": "structured content"},
        raw_override="HAND-WRITTEN PROMPT",
    )
    assert out == "HAND-WRITTEN PROMPT"
    assert "Edit Task" not in out
    assert "structured content" not in out


def test_compose_brief_markdown_raw_override_preserves_inner_whitespace():
    """Power-user path: returned VERBATIM, not stripped — preserves
    intentional leading/trailing whitespace the user typed."""
    raw = "  hand-written\n  with leading spaces  \n"
    out = PromptComposer.compose_brief_markdown({}, raw_override=raw)
    assert out == raw


def test_compose_brief_markdown_raw_override_none_falls_through():
    """None means 'no override'; sections take effect."""
    out = PromptComposer.compose_brief_markdown(
        {"edit_task": "structured"}, raw_override=None
    )
    assert out == "## Edit Task\nstructured"


def test_compose_brief_markdown_raw_override_empty_string_falls_through():
    """Empty / whitespace-only override does NOT blank the prompt — the
    PRD's revert path sets raw_override back to None, but a stray empty
    string in the input should not silently erase the composed output."""
    out = PromptComposer.compose_brief_markdown(
        {"edit_task": "structured"}, raw_override=""
    )
    assert out == "## Edit Task\nstructured"
    out2 = PromptComposer.compose_brief_markdown(
        {"edit_task": "structured"}, raw_override="   \n  "
    )
    assert out2 == "## Edit Task\nstructured"


# --- PromptComposer.extract_sections --------------------------------------


def test_extract_sections_round_trip_all_eight():
    """PRD AC: round-trip sections → rendered markdown → re-extracted
    sections. With every section populated, the round trip is lossless
    (modulo body-whitespace stripping, which is part of the contract)."""
    original = {s.id: f"content for {s.id}" for s in SECTIONS}
    md = PromptComposer.compose_brief_markdown(original)
    extracted = PromptComposer.extract_sections(md)
    assert extracted == original


def test_extract_sections_round_trip_partial():
    """Sparse input → empty sections dropped on the way out → only the
    populated keys come back."""
    original = {"edit_task": "only this", "aesthetic_goal": "and this"}
    md = PromptComposer.compose_brief_markdown(original)
    extracted = PromptComposer.extract_sections(md)
    assert extracted == original


def test_extract_sections_round_trip_strips_body_whitespace():
    """The lossy edge of the round trip — body outer whitespace is
    stripped by compose_brief_markdown and stays stripped on extract."""
    original = {"edit_task": "  padded  "}
    md = PromptComposer.compose_brief_markdown(original)
    extracted = PromptComposer.extract_sections(md)
    assert extracted == {"edit_task": "padded"}


def test_extract_sections_drops_preamble_before_first_heading():
    """Legacy briefs may have free-form preamble; the registry-driven
    schema has no slot for it."""
    md = (
        "Some legacy preamble text.\n\n"
        "## Edit Task\nThe real content\n"
    )
    assert PromptComposer.extract_sections(md) == {"edit_task": "The real content"}


def test_extract_sections_drops_unknown_headings():
    """Headings whose title doesn't match any registered section are
    silently dropped — same posture as compose_brief_markdown for
    unknown section ids."""
    md = (
        "## Edit Task\nkept\n\n"
        "## Random Notes\ndropped because not a registered section\n"
    )
    assert PromptComposer.extract_sections(md) == {"edit_task": "kept"}


def test_extract_sections_empty_string_returns_empty_dict():
    assert PromptComposer.extract_sections("") == {}


def test_extract_sections_no_headings_returns_empty_dict():
    """No ## headings → no structured content to extract."""
    assert PromptComposer.extract_sections("Just a free-form paragraph.") == {}


def test_extract_sections_duplicate_heading_last_writer_wins():
    """Hand-edited legacy briefs may repeat a heading; the principle of
    least surprise is that the user's bottom-most edit takes effect."""
    md = (
        "## Edit Task\nold value\n\n"
        "## Edit Task\nnew value\n"
    )
    assert PromptComposer.extract_sections(md) == {"edit_task": "new value"}


def test_extract_sections_tolerates_title_drift():
    """Round-trip survives cosmetic title polish via case/punctuation-
    insensitive matching."""
    md = "## SCALE & FIDELITY\nbody"
    assert PromptComposer.extract_sections(md) == {"scale_fidelity": "body"}


def test_extract_sections_ignores_compose_output_when_raw_override_was_used():
    """raw_override output bypasses the markdown contract entirely; an
    extractor running over a power-user-typed prompt should NOT pretend
    to have parsed it. (Documented behavior: returns whatever ## headings
    happen to be in the raw text — caller decides whether to call
    extract_sections in the raw-override path. This test pins that we
    don't crash and don't manufacture sections from non-heading text.)"""
    raw = "Just a wall of text with no headings at all"
    composed = PromptComposer.compose_brief_markdown({}, raw_override=raw)
    assert composed == raw
    assert PromptComposer.extract_sections(composed) == {}


# --- DesignBrief schema extension -----------------------------------------


def test_design_brief_defaults_include_empty_sections_and_no_override():
    """Backward compat: an existing call site that constructs a brief
    without the new fields gets an empty sections dict and no override."""
    b = DesignBrief(global_instructions="legacy")
    assert b.sections == {}
    assert b.raw_override is None


def test_design_brief_accepts_sections_payload():
    b = DesignBrief(
        global_instructions="x",
        sections={"edit_task": "stage backyard", "aesthetic_goal": "modern"},
    )
    assert b.sections == {"edit_task": "stage backyard", "aesthetic_goal": "modern"}


def test_design_brief_accepts_raw_override():
    b = DesignBrief(global_instructions="x", raw_override="hand-written")
    assert b.raw_override == "hand-written"


def test_design_brief_legacy_dict_still_constructs():
    """The legacy-palette migration validator must not be broken by the
    new fields — a raw dict with NO sections / raw_override entries
    still constructs cleanly."""
    legacy = {
        "global_instructions": "synth",
        "object_palette": [],
    }
    b = DesignBrief.model_validate(legacy)
    assert b.sections == {}
    assert b.raw_override is None


def test_design_brief_sections_drives_compose_brief_markdown():
    """End-to-end happy path: brief.sections + brief.raw_override feed
    straight into the composer."""
    b = DesignBrief(
        global_instructions="x",
        sections={"edit_task": "Stage the backyard", "arrangement": "Tight cluster"},
    )
    out = PromptComposer.compose_brief_markdown(b.sections, b.raw_override)
    assert out == (
        "## Edit Task\nStage the backyard\n\n"
        "## Arrangement\nTight cluster"
    )

    # Setting raw_override flips the output to the verbatim path.
    b2 = b.model_copy(update={"raw_override": "MANUAL"})
    out2 = PromptComposer.compose_brief_markdown(b2.sections, b2.raw_override)
    assert out2 == "MANUAL"
