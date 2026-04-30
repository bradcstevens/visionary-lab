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

    def test_room_id_with_no_overrides_yields_palette_defaults(self):
        """No overrides for either room → both rooms see palette defaults."""
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


# ---------------------------------------------------------------------------
# ImageObjectOverride field validators
# ---------------------------------------------------------------------------


class TestImageObjectOverrideValidators:
    """Field-level invariants on the override model itself."""

    def test_negative_quantity_rejected(self):
        import pydantic
        from backend.models.design_brief import ImageObjectOverride

        with pytest.raises(pydantic.ValidationError):
            ImageObjectOverride(object_id="x", quantity=-1)

    def test_missing_quantity_rejected(self):
        """Critique catch: defaulting quantity to 0 would silently turn
        ``{object_id}`` into a skip signal. Quantity is REQUIRED so missing
        quantity is a programming error, not a legitimate skip."""
        import pydantic
        from backend.models.design_brief import ImageObjectOverride

        with pytest.raises(pydantic.ValidationError):
            ImageObjectOverride(object_id="x")  # type: ignore[call-arg]

    def test_zero_quantity_allowed(self):
        from backend.models.design_brief import ImageObjectOverride

        ovr = ImageObjectOverride(object_id="x", quantity=0)
        assert ovr.quantity == 0

    @pytest.mark.parametrize("raw", ["", "   ", "\t\n"])
    def test_empty_or_whitespace_placement_coerced_to_none(self, raw):
        from backend.models.design_brief import ImageObjectOverride

        ovr = ImageObjectOverride(object_id="x", quantity=1, placement=raw)
        assert ovr.placement is None

    def test_explicit_none_placement_kept(self):
        from backend.models.design_brief import ImageObjectOverride

        ovr = ImageObjectOverride(object_id="x", quantity=1, placement=None)
        assert ovr.placement is None

    def test_real_placement_string_kept(self):
        from backend.models.design_brief import ImageObjectOverride

        ovr = ImageObjectOverride(object_id="x", quantity=1, placement="back row")
        assert ovr.placement == "back row"

    def test_placement_string_is_stripped(self):
        """Whitespace at the edges trimmed; interior whitespace preserved."""
        from backend.models.design_brief import ImageObjectOverride

        ovr = ImageObjectOverride(object_id="x", quantity=1, placement="  back  row  ")
        assert ovr.placement == "back  row"

    def test_default_enabled_is_true(self):
        from backend.models.design_brief import ImageObjectOverride

        ovr = ImageObjectOverride(object_id="x", quantity=1)
        assert ovr.enabled is True


# ---------------------------------------------------------------------------
# resolve_objects_for_image with per-image overrides — issue 003 ruleset
# ---------------------------------------------------------------------------


class TestResolveObjectsForImageWithOverrides:
    """Full ruleset from the per-image-object-quantities PRD's
    "Resolution logic" section.
    """

    def _make_brief(self, palette, per_image_objects=None):
        from backend.models.design_brief import DesignBrief

        return DesignBrief(
            global_instructions="x",
            object_palette=palette,
            per_image_objects=per_image_objects or {},
        )

    def test_no_overrides_for_room_yields_palette_defaults(self):
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import ObjectEntry

        entry = ObjectEntry(name="Lavender", default_quantity=3, placement="front row")
        brief = self._make_brief([entry])
        [r] = resolve_objects_for_image(brief, room_id="r1")
        assert r.quantity == 3
        assert r.placement == "front row"

    def test_quantity_override_applied(self):
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        entry = ObjectEntry(name="Lavender", default_quantity=3)
        brief = self._make_brief(
            [entry],
            per_image_objects={"r1": [ImageObjectOverride(object_id=entry.id, quantity=8)]},
        )
        [r] = resolve_objects_for_image(brief, room_id="r1")
        assert r.quantity == 8

    def test_placement_none_inherits_palette_placement(self):
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        entry = ObjectEntry(name="Lavender", default_quantity=3, placement="middle row")
        brief = self._make_brief(
            [entry],
            per_image_objects={
                "r1": [ImageObjectOverride(object_id=entry.id, quantity=5, placement=None)]
            },
        )
        [r] = resolve_objects_for_image(brief, room_id="r1")
        assert r.placement == "middle row"

    def test_placement_non_none_replaces(self):
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        entry = ObjectEntry(name="Lavender", default_quantity=3, placement="middle row")
        brief = self._make_brief(
            [entry],
            per_image_objects={
                "r1": [ImageObjectOverride(object_id=entry.id, quantity=5, placement="back row")]
            },
        )
        [r] = resolve_objects_for_image(brief, room_id="r1")
        assert r.placement == "back row"

    def test_enabled_false_excludes_object(self):
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        kept = ObjectEntry(name="Pine", default_quantity=2)
        skipped = ObjectEntry(name="Lavender", default_quantity=3)
        brief = self._make_brief(
            [kept, skipped],
            per_image_objects={
                "r1": [ImageObjectOverride(object_id=skipped.id, quantity=3, enabled=False)]
            },
        )
        names = [r.name for r in resolve_objects_for_image(brief, room_id="r1")]
        assert names == ["Pine"]

    def test_zero_quantity_excludes_object(self):
        """quantity=0 is the second equivalent skip signal alongside enabled=False."""
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        kept = ObjectEntry(name="Pine", default_quantity=2)
        skipped = ObjectEntry(name="Lavender", default_quantity=3)
        brief = self._make_brief(
            [kept, skipped],
            per_image_objects={
                "r1": [ImageObjectOverride(object_id=skipped.id, quantity=0, enabled=True)]
            },
        )
        names = [r.name for r in resolve_objects_for_image(brief, room_id="r1")]
        assert names == ["Pine"]

    def test_unknown_object_id_silently_ignored(self):
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        entry = ObjectEntry(name="Pine", default_quantity=2)
        brief = self._make_brief(
            [entry],
            per_image_objects={
                "r1": [
                    ImageObjectOverride(object_id="ghost-id-not-in-palette", quantity=99),
                    ImageObjectOverride(object_id=entry.id, quantity=5),
                ]
            },
        )
        [r] = resolve_objects_for_image(brief, room_id="r1")
        # Only Pine returned; the ghost override is silently dropped.
        assert r.quantity == 5
        assert r.name == "Pine"

    def test_other_rooms_unaffected_by_overrides(self):
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        entry = ObjectEntry(name="Lavender", default_quantity=3)
        brief = self._make_brief(
            [entry],
            per_image_objects={"r1": [ImageObjectOverride(object_id=entry.id, quantity=99)]},
        )
        [a] = resolve_objects_for_image(brief, room_id="r1")
        [b] = resolve_objects_for_image(brief, room_id="r2")
        assert a.quantity == 99
        assert b.quantity == 3

    def test_duplicate_overrides_last_write_wins(self):
        """Defensive contract for legacy / hand-edited data: if the same
        object_id appears twice in the override list, the LAST entry wins.
        Critique catch: keep this deterministic so dict-by-id ordering is
        not load-bearing.
        """
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        entry = ObjectEntry(name="Lavender", default_quantity=3)
        brief = self._make_brief(
            [entry],
            per_image_objects={
                "r1": [
                    ImageObjectOverride(object_id=entry.id, quantity=5),
                    ImageObjectOverride(object_id=entry.id, quantity=10),
                    ImageObjectOverride(object_id=entry.id, quantity=7),
                ]
            },
        )
        [r] = resolve_objects_for_image(brief, room_id="r1")
        assert r.quantity == 7

    def test_palette_order_preserved(self):
        """Resolution preserves palette declaration order, not override order."""
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        a = ObjectEntry(name="A", default_quantity=1)
        b = ObjectEntry(name="B", default_quantity=2)
        c = ObjectEntry(name="C", default_quantity=3)
        brief = self._make_brief(
            [a, b, c],
            per_image_objects={
                "r1": [
                    ImageObjectOverride(object_id=c.id, quantity=30),
                    ImageObjectOverride(object_id=a.id, quantity=10),
                ]
            },
        )
        names = [r.name for r in resolve_objects_for_image(brief, room_id="r1")]
        assert names == ["A", "B", "C"]

    def test_empty_palette_with_overrides_returns_empty(self):
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import ImageObjectOverride

        brief = self._make_brief(
            [],
            per_image_objects={
                "r1": [ImageObjectOverride(object_id="ghost", quantity=5)]
            },
        )
        assert resolve_objects_for_image(brief, room_id="r1") == []

    def test_empty_override_list_for_room_yields_palette_defaults(self):
        """A room with `per_image_objects[r1] = []` (empty list, not absent)
        should behave the same as no key at all."""
        from backend.core.brief_resolver import resolve_objects_for_image
        from backend.models.design_brief import ObjectEntry

        entry = ObjectEntry(name="Lavender", default_quantity=3)
        brief = self._make_brief([entry], per_image_objects={"r1": []})
        [r] = resolve_objects_for_image(brief, room_id="r1")
        assert r.quantity == 3
