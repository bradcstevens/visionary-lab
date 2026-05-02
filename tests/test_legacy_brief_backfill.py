"""Tests for the lazy legacy-brief sections backfill (issue 016).

The backfill mutates ``project['design_brief']`` in-place to populate
``sections`` from existing legacy fields (``global_instructions``,
``object_palette``, ``placement_guide``, ``preserve_elements``).
Idempotent: a brief whose ``sections`` already carries any canonical
section is left untouched.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from backend.core.brief_generator import (
    backfill_legacy_brief_sections,
    derive_sections_from_legacy_brief,
)
from backend.core.brief_section_registry import section_ids


def _legacy_brief() -> Dict[str, Any]:
    return {
        "global_instructions": "Add native xeriscape plants and a fire pit.",
        "object_palette": [
            {
                "id": "uuid-1",
                "name": "Vanderwolf Pine",
                "description": "Pinus flexilis 'Vanderwolf'",
                "category": "tree",
                "default_quantity": 3,
                "size": "8-10 ft",
                "placement": "back row",
                "visual_notes": "Silver-blue needles",
            },
            {
                "id": "uuid-2",
                "name": "Adirondack chair",
                "description": None,
                "category": "furniture",
                "default_quantity": 2,
                "size": "standard",
                "placement": "patio",
                "visual_notes": None,
            },
        ],
        "placement_guide": {
            "back_row": "Tall conifers along fence",
            "middle_row": "Perennials and grasses",
            "front_row": "Low groundcover",
            "accent_areas": "Fire pit centerpiece",
        },
        "per_image_notes": {},
        "per_image_objects": {},
        "preserve_elements": ["existing patio", "fence", "lawn"],
    }


def _legacy_project() -> Dict[str, Any]:
    return {
        "id": "proj-1",
        "name": "Backyard",
        "design_brief": _legacy_brief(),
    }


# -----------------------------------------------------------------------------
# derive_sections_from_legacy_brief — pure helper
# -----------------------------------------------------------------------------


def test_derive_populates_edit_task_from_global_instructions():
    brief = _legacy_brief()
    sections = derive_sections_from_legacy_brief(brief)
    assert "edit_task" in sections
    assert "xeriscape" in sections["edit_task"]


def test_derive_populates_do_not_alter_from_preserve_elements():
    brief = _legacy_brief()
    sections = derive_sections_from_legacy_brief(brief)
    assert "do_not_alter" in sections
    assert "existing patio" in sections["do_not_alter"]
    assert "fence" in sections["do_not_alter"]
    assert "lawn" in sections["do_not_alter"]


def test_derive_populates_object_palette_from_palette_entries():
    brief = _legacy_brief()
    sections = derive_sections_from_legacy_brief(brief)
    assert "object_palette" in sections
    assert "Vanderwolf Pine" in sections["object_palette"]
    assert "Adirondack chair" in sections["object_palette"]
    # Quantity should appear
    assert "3" in sections["object_palette"]


def test_derive_populates_arrangement_from_placement_guide():
    brief = _legacy_brief()
    sections = derive_sections_from_legacy_brief(brief)
    assert "arrangement" in sections
    assert "Tall conifers" in sections["arrangement"]
    assert "Perennials" in sections["arrangement"]
    assert "Fire pit" in sections["arrangement"]


def test_derive_omits_sections_with_no_source_data():
    brief = _legacy_brief()
    sections = derive_sections_from_legacy_brief(brief)
    # No source for these in a vanilla legacy brief — must not appear
    assert "edit_zone" not in sections
    assert "regional_constraints" not in sections
    assert "aesthetic_goal" not in sections
    assert "scale_fidelity" not in sections


def test_derive_is_deterministic():
    brief = _legacy_brief()
    a = derive_sections_from_legacy_brief(brief)
    b = derive_sections_from_legacy_brief(brief)
    assert a == b


def test_derive_extracts_embedded_markdown_headings():
    brief = _legacy_brief()
    brief["global_instructions"] = (
        "## Edit Task\nReplace the lawn with native plants.\n\n"
        "## Aesthetic Goal\nXeriscape, modern semi-arid look.\n\n"
        "## Regional Constraints\nUSDA zone 5b, water-wise."
    )
    sections = derive_sections_from_legacy_brief(brief)
    assert sections["edit_task"] == "Replace the lawn with native plants."
    assert sections["aesthetic_goal"] == "Xeriscape, modern semi-arid look."
    assert sections["regional_constraints"] == "USDA zone 5b, water-wise."


def test_derive_falls_back_to_full_text_when_no_headings():
    brief = {"global_instructions": "Plain free-form prompt with no headings."}
    sections = derive_sections_from_legacy_brief(brief)
    assert sections["edit_task"] == "Plain free-form prompt with no headings."


def test_derive_handles_empty_brief():
    sections = derive_sections_from_legacy_brief({})
    assert sections == {}


def test_derive_handles_missing_optional_fields():
    sections = derive_sections_from_legacy_brief(
        {"global_instructions": "Just some text."}
    )
    assert sections == {"edit_task": "Just some text."}


def test_derive_skips_palette_entries_without_name():
    brief = {
        "global_instructions": "x",
        "object_palette": [
            {"name": "", "default_quantity": 1},
            {"name": "Pine", "default_quantity": 2},
        ],
    }
    sections = derive_sections_from_legacy_brief(brief)
    assert "Pine" in sections["object_palette"]


def test_derive_handles_non_dict_palette_entry():
    brief = {
        "global_instructions": "x",
        "object_palette": ["not a dict", {"name": "Oak", "default_quantity": 1}],
    }
    sections = derive_sections_from_legacy_brief(brief)
    assert "Oak" in sections["object_palette"]


def test_derive_only_returns_canonical_section_ids():
    brief = _legacy_brief()
    sections = derive_sections_from_legacy_brief(brief)
    canonical = set(section_ids())
    assert set(sections.keys()).issubset(canonical)


# -----------------------------------------------------------------------------
# backfill_legacy_brief_sections — project mutation contract
# -----------------------------------------------------------------------------


def test_backfill_mutates_project_in_place_and_returns_true():
    project = _legacy_project()
    mutated = backfill_legacy_brief_sections(project)
    assert mutated is True
    assert project["design_brief"]["sections"]
    assert "edit_task" in project["design_brief"]["sections"]


def test_backfill_is_idempotent_on_repeat_call():
    project = _legacy_project()
    first = backfill_legacy_brief_sections(project)
    snapshot = deepcopy(project)
    second = backfill_legacy_brief_sections(project)
    assert first is True
    assert second is False
    assert project == snapshot


def test_backfill_skips_when_sections_already_populated():
    project = _legacy_project()
    project["design_brief"]["sections"] = {"edit_task": "manually set"}
    mutated = backfill_legacy_brief_sections(project)
    assert mutated is False
    assert project["design_brief"]["sections"] == {"edit_task": "manually set"}


def test_backfill_treats_empty_sections_dict_as_legacy():
    project = _legacy_project()
    project["design_brief"]["sections"] = {}
    mutated = backfill_legacy_brief_sections(project)
    assert mutated is True
    assert "edit_task" in project["design_brief"]["sections"]


def test_backfill_returns_false_for_project_without_brief():
    mutated = backfill_legacy_brief_sections({"id": "p"})
    assert mutated is False


def test_backfill_returns_false_for_non_dict_brief():
    mutated = backfill_legacy_brief_sections({"id": "p", "design_brief": None})
    assert mutated is False


def test_backfill_returns_false_for_non_dict_project():
    assert backfill_legacy_brief_sections(None) is False  # type: ignore[arg-type]
    assert backfill_legacy_brief_sections("not a dict") is False  # type: ignore[arg-type]


def test_backfill_returns_false_when_no_derivable_content():
    # Brief exists but every legacy field is empty -> nothing to derive
    project = {"id": "p", "design_brief": {}}
    mutated = backfill_legacy_brief_sections(project)
    assert mutated is False


def test_backfill_does_not_invoke_extractor_on_second_read():
    """AC bullet: second read returns same sections without invoking extractor.

    Pinned by patching ``derive_sections_from_legacy_brief`` after the
    first call and asserting the second call's mutation path is not
    entered (returns False, no derivation work performed).
    """
    project = _legacy_project()
    backfill_legacy_brief_sections(project)
    sections_before = deepcopy(project["design_brief"]["sections"])

    # If the second call were to run the extractor, sections would be
    # rewritten. Instead, the idempotent guard short-circuits.
    derive_called = MagicMock()
    import backend.core.brief_generator as bg
    original = bg.derive_sections_from_legacy_brief
    bg.derive_sections_from_legacy_brief = derive_called
    try:
        result = backfill_legacy_brief_sections(project)
    finally:
        bg.derive_sections_from_legacy_brief = original

    assert result is False
    derive_called.assert_not_called()
    assert project["design_brief"]["sections"] == sections_before
