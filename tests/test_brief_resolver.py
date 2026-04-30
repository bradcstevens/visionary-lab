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


# ---------------------------------------------------------------------------
# reconcile_overrides_by_name — issue 004 of the per-image-object-quantities
# PRD. Carries per-image overrides from a prior brief into a freshly-
# regenerated brief by matching palette entries on case-insensitive,
# whitespace-trimmed name. ``ReconcileSummary`` reports how many were
# carried forward vs. dropped.
# ---------------------------------------------------------------------------


class TestReconcileOverridesByName:
    """End-to-end behaviour of ``reconcile_overrides_by_name``.

    Tests construct prev/new ``DesignBrief`` instances directly (no LLM
    mock involved). ``ImageObjectOverride.object_id`` strings are taken
    from the palette entries so the prev → new mapping is realistic.
    """

    def _make_brief(self, palette, per_image_objects=None):
        from backend.models.design_brief import DesignBrief

        return DesignBrief(
            global_instructions="x",
            object_palette=palette,
            per_image_objects=per_image_objects or {},
        )

    def test_identical_name_carry_forward(self):
        """Same name in prev + new palette → override carried forward
        with object_id rewritten to the new palette's UUID.
        """
        from backend.core.brief_resolver import reconcile_overrides_by_name
        from backend.models.design_brief import (
            ImageObjectOverride,
            ObjectEntry,
            ReconcileSummary,
        )

        prev_pine = ObjectEntry(name="Pine", default_quantity=2)
        new_pine = ObjectEntry(name="Pine", default_quantity=2)
        # Sanity: prev id != new id (different UUIDs).
        assert prev_pine.id != new_pine.id

        prev = self._make_brief(
            [prev_pine],
            per_image_objects={
                "r1": [ImageObjectOverride(object_id=prev_pine.id, quantity=8)]
            },
        )
        new = self._make_brief([new_pine])

        reconciled, summary = reconcile_overrides_by_name(prev, new)

        assert isinstance(summary, ReconcileSummary)
        assert summary.carried_forward == 1
        assert summary.dropped == 0
        # Override now points at the NEW palette's UUID.
        assert reconciled.per_image_objects["r1"][0].object_id == new_pine.id
        assert reconciled.per_image_objects["r1"][0].quantity == 8

    def test_renamed_object_dropped(self):
        """Prior override referencing a name that no longer exists in the
        new palette is dropped and counted in ``summary.dropped``.
        """
        from backend.core.brief_resolver import reconcile_overrides_by_name
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        prev_pine = ObjectEntry(name="Pine", default_quantity=2)
        new_oak = ObjectEntry(name="Oak", default_quantity=1)

        prev = self._make_brief(
            [prev_pine],
            per_image_objects={
                "r1": [ImageObjectOverride(object_id=prev_pine.id, quantity=8)]
            },
        )
        new = self._make_brief([new_oak])

        reconciled, summary = reconcile_overrides_by_name(prev, new)

        assert summary.carried_forward == 0
        assert summary.dropped == 1
        # No room key for r1 because nothing was carried forward, OR
        # the room key has an empty list. Either is acceptable contract.
        assert reconciled.per_image_objects.get("r1", []) == []

    def test_case_insensitive_matching(self):
        """``"Lavender"`` in prev should match ``"lavender"`` in new."""
        from backend.core.brief_resolver import reconcile_overrides_by_name
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        prev_lav = ObjectEntry(name="Lavender", default_quantity=3)
        new_lav = ObjectEntry(name="lavender", default_quantity=3)

        prev = self._make_brief(
            [prev_lav],
            per_image_objects={
                "r1": [ImageObjectOverride(object_id=prev_lav.id, quantity=10)]
            },
        )
        new = self._make_brief([new_lav])

        reconciled, summary = reconcile_overrides_by_name(prev, new)

        assert summary.carried_forward == 1
        assert summary.dropped == 0
        assert reconciled.per_image_objects["r1"][0].quantity == 10
        assert reconciled.per_image_objects["r1"][0].object_id == new_lav.id

    def test_whitespace_trimmed_matching(self):
        """``"  Lavender  "`` in prev matches ``"Lavender"`` in new."""
        from backend.core.brief_resolver import reconcile_overrides_by_name
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        prev_lav = ObjectEntry(name="  Lavender  ", default_quantity=3)
        new_lav = ObjectEntry(name="Lavender", default_quantity=3)

        prev = self._make_brief(
            [prev_lav],
            per_image_objects={
                "r1": [ImageObjectOverride(object_id=prev_lav.id, quantity=10)]
            },
        )
        new = self._make_brief([new_lav])

        reconciled, summary = reconcile_overrides_by_name(prev, new)

        assert summary.carried_forward == 1
        assert reconciled.per_image_objects["r1"][0].object_id == new_lav.id

    def test_orphan_in_prev_palette_dropped(self):
        """Prior override referencing an object_id that doesn't exist in
        the prev palette either (legacy / hand-edited data) is dropped.
        """
        from backend.core.brief_resolver import reconcile_overrides_by_name
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        prev_pine = ObjectEntry(name="Pine", default_quantity=2)
        new_pine = ObjectEntry(name="Pine", default_quantity=2)

        prev = self._make_brief(
            [prev_pine],
            per_image_objects={
                "r1": [
                    # Valid override.
                    ImageObjectOverride(object_id=prev_pine.id, quantity=8),
                    # Orphan: object_id not in prev_pine palette.
                    ImageObjectOverride(object_id="ghost-uuid", quantity=99),
                ]
            },
        )
        new = self._make_brief([new_pine])

        reconciled, summary = reconcile_overrides_by_name(prev, new)

        # Only Pine carried forward; orphan dropped.
        assert summary.carried_forward == 1
        assert summary.dropped == 1
        [r] = reconciled.per_image_objects["r1"]
        assert r.object_id == new_pine.id
        assert r.quantity == 8

    def test_duplicate_normalized_name_in_new_palette_drops_override(self):
        """If the new palette contains two entries that normalize to the
        same name, the match is ambiguous and we drop the override rather
        than guess. Caught by rubber-duck review of the issue-004 plan.
        """
        from backend.core.brief_resolver import reconcile_overrides_by_name
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        prev_pine = ObjectEntry(name="Pine", default_quantity=2)
        new_pine_1 = ObjectEntry(name="Pine", default_quantity=2)
        new_pine_2 = ObjectEntry(name="pine", default_quantity=4)  # normalizes to same.

        prev = self._make_brief(
            [prev_pine],
            per_image_objects={
                "r1": [ImageObjectOverride(object_id=prev_pine.id, quantity=8)]
            },
        )
        new = self._make_brief([new_pine_1, new_pine_2])

        reconciled, summary = reconcile_overrides_by_name(prev, new)

        assert summary.carried_forward == 0
        assert summary.dropped == 1
        assert reconciled.per_image_objects.get("r1", []) == []

    def test_duplicate_normalized_name_in_prev_palette_drops_override(self):
        """Same rule applies symmetrically: if the prev palette has
        ambiguous names, we can't reliably identify which one the
        override referred to even if the new palette is unambiguous.
        """
        from backend.core.brief_resolver import reconcile_overrides_by_name
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        prev_pine_1 = ObjectEntry(name="Pine", default_quantity=2)
        prev_pine_2 = ObjectEntry(name=" pine ", default_quantity=2)
        new_pine = ObjectEntry(name="Pine", default_quantity=2)

        prev = self._make_brief(
            [prev_pine_1, prev_pine_2],
            per_image_objects={
                "r1": [
                    # Targets prev_pine_1 specifically — but reconcile
                    # only looks up by NAME, and "pine" is duplicated.
                    ImageObjectOverride(object_id=prev_pine_1.id, quantity=8)
                ]
            },
        )
        new = self._make_brief([new_pine])

        reconciled, summary = reconcile_overrides_by_name(prev, new)

        assert summary.carried_forward == 0
        assert summary.dropped == 1

    def test_skip_override_enabled_false_carried_forward(self):
        """``enabled=False`` is a meaningful user edit ("skip this object
        in this image"). It must carry forward when the name matches.
        """
        from backend.core.brief_resolver import reconcile_overrides_by_name
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        prev_pine = ObjectEntry(name="Pine", default_quantity=2)
        new_pine = ObjectEntry(name="Pine", default_quantity=2)

        prev = self._make_brief(
            [prev_pine],
            per_image_objects={
                "r1": [
                    ImageObjectOverride(
                        object_id=prev_pine.id, quantity=0, enabled=False
                    )
                ]
            },
        )
        new = self._make_brief([new_pine])

        reconciled, summary = reconcile_overrides_by_name(prev, new)

        assert summary.carried_forward == 1
        assert summary.dropped == 0
        [r] = reconciled.per_image_objects["r1"]
        assert r.object_id == new_pine.id
        assert r.enabled is False
        assert r.quantity == 0

    def test_skip_override_quantity_zero_carried_forward(self):
        """``quantity=0`` (the other equivalent skip signal) must also
        survive reconciliation when the name matches.
        """
        from backend.core.brief_resolver import reconcile_overrides_by_name
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        prev_pine = ObjectEntry(name="Pine", default_quantity=2)
        new_pine = ObjectEntry(name="Pine", default_quantity=2)

        prev = self._make_brief(
            [prev_pine],
            per_image_objects={
                "r1": [
                    ImageObjectOverride(object_id=prev_pine.id, quantity=0)
                ]
            },
        )
        new = self._make_brief([new_pine])

        reconciled, summary = reconcile_overrides_by_name(prev, new)

        assert summary.carried_forward == 1
        [r] = reconciled.per_image_objects["r1"]
        assert r.quantity == 0

    def test_prev_wins_over_new_pre_populated_override_on_conflict(self):
        """If both prev and new have an override for the same (room_id,
        normalized name), the prev override (user edit) wins over the
        LLM-emitted suggestion in new. Conflict resolution decision
        documented in the issue-004 plan; rationale: never silently
        overwrite a user edit with a regenerated auto-suggestion.
        """
        from backend.core.brief_resolver import reconcile_overrides_by_name
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        prev_lav = ObjectEntry(name="Lavender", default_quantity=3)
        new_lav = ObjectEntry(name="Lavender", default_quantity=3)

        prev = self._make_brief(
            [prev_lav],
            per_image_objects={
                "r1": [ImageObjectOverride(object_id=prev_lav.id, quantity=8)]
            },
        )
        new = self._make_brief(
            [new_lav],
            per_image_objects={
                # LLM-suggested override pre-populated in new brief.
                "r1": [ImageObjectOverride(object_id=new_lav.id, quantity=4)]
            },
        )

        reconciled, summary = reconcile_overrides_by_name(prev, new)

        # Prev wins. quantity=8, NOT 4.
        [r] = reconciled.per_image_objects["r1"]
        assert r.quantity == 8
        assert r.object_id == new_lav.id
        assert summary.carried_forward == 1
        assert summary.dropped == 0

    def test_prev_override_appended_when_room_not_in_new(self):
        """Prev override on a (room_id, name) pair that the new brief
        didn't pre-populate is appended to new — preserving overrides
        even when the LLM doesn't re-emit them.
        """
        from backend.core.brief_resolver import reconcile_overrides_by_name
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        prev_lav = ObjectEntry(name="Lavender", default_quantity=3)
        new_lav = ObjectEntry(name="Lavender", default_quantity=3)

        prev = self._make_brief(
            [prev_lav],
            per_image_objects={
                "r2": [ImageObjectOverride(object_id=prev_lav.id, quantity=8)]
            },
        )
        new = self._make_brief([new_lav])  # no pre-populated overrides at all.

        reconciled, summary = reconcile_overrides_by_name(prev, new)

        assert summary.carried_forward == 1
        assert summary.dropped == 0
        [r] = reconciled.per_image_objects["r2"]
        assert r.quantity == 8
        assert r.object_id == new_lav.id

    def test_prev_brief_with_empty_per_image_objects_yields_zero_counts(self):
        """No-op base case: prev had a palette but no overrides → both
        counts 0, new brief returned unchanged.
        """
        from backend.core.brief_resolver import reconcile_overrides_by_name
        from backend.models.design_brief import ObjectEntry

        prev_pine = ObjectEntry(name="Pine", default_quantity=2)
        new_pine = ObjectEntry(name="Pine", default_quantity=2)

        prev = self._make_brief([prev_pine])
        new = self._make_brief([new_pine])

        reconciled, summary = reconcile_overrides_by_name(prev, new)

        assert summary.carried_forward == 0
        assert summary.dropped == 0
        assert reconciled.per_image_objects == {}

    def test_input_briefs_are_not_mutated(self):
        """Reconciliation must produce a NEW DesignBrief; the prev and
        new inputs stay untouched so callers can keep using them.
        """
        from backend.core.brief_resolver import reconcile_overrides_by_name
        from backend.models.design_brief import ImageObjectOverride, ObjectEntry

        prev_pine = ObjectEntry(name="Pine", default_quantity=2)
        new_pine = ObjectEntry(name="Pine", default_quantity=2)

        prev = self._make_brief(
            [prev_pine],
            per_image_objects={
                "r1": [ImageObjectOverride(object_id=prev_pine.id, quantity=8)]
            },
        )
        new = self._make_brief([new_pine])

        # Snapshot relevant state.
        prev_overrides_before = list(prev.per_image_objects["r1"])
        new_overrides_before = dict(new.per_image_objects)

        reconcile_overrides_by_name(prev, new)

        assert prev.per_image_objects["r1"] == prev_overrides_before
        assert new.per_image_objects == new_overrides_before
