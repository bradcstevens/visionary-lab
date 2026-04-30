"""Tests for the pure brief_resolver module.

The module owns three pure operations the rest of the codebase depends on:

    * ``migrate_legacy_plant_palette`` — pure, idempotent legacy → generic dict
      shape converter. Issue 001 of the per-image-object-quantities PRD.
    * ``resolve_objects_for_image`` — basic-mode in this slice (palette-only
      projection). Override-aware behaviour lands in issue 003.
    * ``ResolvedObject`` — frozen-ish projection used by ``brief_to_prompts``.

Tests assert externally observable behaviour: input / output of each function,
not internal helper signatures or log strings.
"""
from __future__ import annotations

import re
import uuid

import pytest


# ---------------------------------------------------------------------------
# migrate_legacy_plant_palette — pure transformation
# ---------------------------------------------------------------------------


class TestMigrateLegacyIdempotency:
    """Migration must be safe to apply zero-or-more times."""

    def test_returns_same_object_when_object_palette_already_present(self):
        from backend.core.brief_resolver import migrate_legacy_plant_palette

        already_migrated = {
            "global_instructions": "x",
            "object_palette": [{"id": "id-1", "name": "Pine", "category": "tree"}],
            "per_image_objects": {},
        }
        result = migrate_legacy_plant_palette(already_migrated)
        # Same object — no copy, no allocation, fast no-op.
        assert result is already_migrated

    def test_double_migration_yields_equal_output(self):
        from backend.core.brief_resolver import migrate_legacy_plant_palette

        legacy = {
            "global_instructions": "x",
            "plant_palette": [
                {"species": "Lavender", "quantity": 2, "size": "2 ft"},
            ],
        }
        once = migrate_legacy_plant_palette(legacy)
        twice = migrate_legacy_plant_palette(once)
        assert once == twice

    def test_input_is_not_mutated(self):
        from backend.core.brief_resolver import migrate_legacy_plant_palette

        legacy = {
            "global_instructions": "x",
            "plant_palette": [{"species": "Lavender"}],
        }
        # Snapshot key set before migration.
        original_keys = set(legacy.keys())
        migrate_legacy_plant_palette(legacy)
        # Original dict's keys must be untouched.
        assert set(legacy.keys()) == original_keys
        assert "plant_palette" in legacy
        assert "object_palette" not in legacy


class TestMigrateLegacyShapeConversion:
    """Field-level conversion: PlantEntry shape → ObjectEntry shape."""

    def test_species_to_name(self):
        from backend.core.brief_resolver import migrate_legacy_plant_palette

        result = migrate_legacy_plant_palette(
            {"global_instructions": "x", "plant_palette": [{"species": "Lavender"}]}
        )
        assert result["object_palette"][0]["name"] == "Lavender"

    def test_botanical_name_to_description(self):
        from backend.core.brief_resolver import migrate_legacy_plant_palette

        result = migrate_legacy_plant_palette(
            {
                "global_instructions": "x",
                "plant_palette": [
                    {"species": "Pine", "botanical_name": "Pinus flexilis"}
                ],
            }
        )
        assert result["object_palette"][0]["description"] == "Pinus flexilis"

    def test_quantity_to_default_quantity(self):
        from backend.core.brief_resolver import migrate_legacy_plant_palette

        result = migrate_legacy_plant_palette(
            {
                "global_instructions": "x",
                "plant_palette": [{"species": "Lavender", "quantity": 7}],
            }
        )
        assert result["object_palette"][0]["default_quantity"] == 7

    def test_size_placement_visual_notes_copied_through(self):
        from backend.core.brief_resolver import migrate_legacy_plant_palette

        result = migrate_legacy_plant_palette(
            {
                "global_instructions": "x",
                "plant_palette": [
                    {
                        "species": "Lavender",
                        "size": "2 ft",
                        "placement": "front row",
                        "visual_notes": "purple flowers",
                    }
                ],
            }
        )
        out = result["object_palette"][0]
        assert out["size"] == "2 ft"
        assert out["placement"] == "front row"
        assert out["visual_notes"] == "purple flowers"

    def test_each_entry_gets_fresh_uuid_id(self):
        from backend.core.brief_resolver import migrate_legacy_plant_palette

        result = migrate_legacy_plant_palette(
            {
                "global_instructions": "x",
                "plant_palette": [
                    {"species": "Lavender"},
                    {"species": "Rosemary"},
                ],
            }
        )
        ids = [e["id"] for e in result["object_palette"]]
        assert len(ids) == 2
        assert len(set(ids)) == 2  # unique
        for entry_id in ids:
            uuid.UUID(entry_id)  # raises if not a valid UUID

    def test_legacy_plant_palette_key_dropped(self):
        from backend.core.brief_resolver import migrate_legacy_plant_palette

        result = migrate_legacy_plant_palette(
            {
                "global_instructions": "x",
                "plant_palette": [{"species": "Lavender"}],
            }
        )
        assert "plant_palette" not in result
        assert "object_palette" in result

    def test_per_image_objects_initialised_to_empty_dict_when_missing(self):
        from backend.core.brief_resolver import migrate_legacy_plant_palette

        result = migrate_legacy_plant_palette(
            {"global_instructions": "x", "plant_palette": []}
        )
        assert result["per_image_objects"] == {}

    def test_other_fields_preserved(self):
        from backend.core.brief_resolver import migrate_legacy_plant_palette

        result = migrate_legacy_plant_palette(
            {
                "global_instructions": "Add evergreens",
                "plant_palette": [],
                "preserve_elements": ["fire pit"],
                "per_image_notes": {"r1": "extra dense"},
                "placement_guide": {"back_row": "Tall conifers"},
            }
        )
        assert result["global_instructions"] == "Add evergreens"
        assert result["preserve_elements"] == ["fire pit"]
        assert result["per_image_notes"] == {"r1": "extra dense"}
        assert result["placement_guide"] == {"back_row": "Tall conifers"}


class TestMigrateLegacyTreeHeuristic:
    """Categorisation rule: numeric ≥ 6 with ft/feet/tall in size, OR a
    tree-name token in species (whole word, case-insensitive) → TREE; else
    PLANT.
    """

    @pytest.mark.parametrize(
        "size,expected_category",
        [
            ("8-10 ft", "tree"),
            ("20-25 ft tall", "tree"),
            ("6 feet", "tree"),
            ("4-6 ft", "tree"),  # max ≥ 6 → tree
            ("5 ft", "plant"),  # < 6
            ("2 ft", "plant"),
            ("", "plant"),  # no info → default plant
        ],
    )
    def test_size_heuristic(self, size, expected_category):
        from backend.core.brief_resolver import migrate_legacy_plant_palette

        result = migrate_legacy_plant_palette(
            {
                "global_instructions": "x",
                "plant_palette": [{"species": "Some Generic Shrub", "size": size}],
            }
        )
        assert result["object_palette"][0]["category"] == expected_category

    @pytest.mark.parametrize(
        "species,expected",
        [
            ("Vanderwolf's Pyramid Limber Pine", "tree"),
            ("Baby Blue Eyes Spruce", "tree"),
            ("Bur Oak", "tree"),
            ("Sugar Maple", "tree"),
            ("Birch Tree", "tree"),
            ("Western Red Cedar", "tree"),
            ("Douglas Fir", "tree"),
            ("Weeping Willow", "tree"),
            ("Saucer Magnolia", "tree"),
            ("Coast Redwood", "tree"),
            ("Sequoia Giant", "tree"),
            ("Date Palm", "tree"),
            ("English Elm", "tree"),
            ("Green Ash", "tree"),
            ("Eastern Juniper", "tree"),
            ("Italian Cypress", "tree"),
            # Substring should NOT match — word boundary required
            ("Oakleaf Hydrangea", "plant"),  # 'oak' is part of 'oakleaf'
            ("Spruceland Boxwood", "plant"),  # 'spruce' is a substring
            ("Lavender", "plant"),
            ("Rosemary", "plant"),
            ("Boxwood", "plant"),
            ("Hydrangea", "plant"),
        ],
    )
    def test_species_token_heuristic(self, species, expected):
        from backend.core.brief_resolver import migrate_legacy_plant_palette

        result = migrate_legacy_plant_palette(
            {
                "global_instructions": "x",
                "plant_palette": [{"species": species, "size": "3 ft"}],
            }
        )
        assert result["object_palette"][0]["category"] == expected


# ---------------------------------------------------------------------------
# ResolvedObject + resolve_objects_for_image — basic projection
# ---------------------------------------------------------------------------


class TestResolveObjectsForImageBasic:
    """In this slice the resolver simply projects the palette into
    ResolvedObject rows; override-aware logic lands in issue 003.
    """

    def test_empty_palette_returns_empty_list(self):
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import DesignBrief

        brief = DesignBrief(global_instructions="x")
        assert resolve_objects_for_image(brief, room_id="r1") == []

    def test_each_palette_entry_becomes_a_resolved_object(self):
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import DesignBrief, ObjectEntry

        brief = DesignBrief(
            global_instructions="x",
            object_palette=[
                ObjectEntry(name="Lavender", default_quantity=3, size="2 ft"),
                ObjectEntry(name="Pine", default_quantity=2, size="8 ft"),
            ],
        )
        resolved = resolve_objects_for_image(brief, room_id="r1")
        assert len(resolved) == 2
        names = [r.name for r in resolved]
        assert names == ["Lavender", "Pine"]

    def test_resolved_object_carries_id_and_category(self):
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import DesignBrief, ObjectCategory, ObjectEntry

        entry = ObjectEntry(name="Pine", category=ObjectCategory.TREE)
        brief = DesignBrief(global_instructions="x", object_palette=[entry])
        [resolved] = resolve_objects_for_image(brief, room_id="r1")
        assert resolved.id == entry.id
        assert resolved.category == ObjectCategory.TREE

    def test_default_quantity_projects_to_quantity(self):
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import DesignBrief, ObjectEntry

        brief = DesignBrief(
            global_instructions="x",
            object_palette=[ObjectEntry(name="Lavender", default_quantity=4)],
        )
        [resolved] = resolve_objects_for_image(brief, room_id="r1")
        assert resolved.quantity == 4

    def test_room_id_does_not_affect_result_in_basic_mode(self):
        """In this slice resolver ignores room_id; same input → same output."""
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import DesignBrief, ObjectEntry

        brief = DesignBrief(
            global_instructions="x",
            object_palette=[ObjectEntry(name="Lavender", default_quantity=3)],
        )
        a = resolve_objects_for_image(brief, room_id="room-A")
        b = resolve_objects_for_image(brief, room_id="room-B")
        assert [r.name for r in a] == [r.name for r in b]
        assert [r.quantity for r in a] == [r.quantity for r in b]
