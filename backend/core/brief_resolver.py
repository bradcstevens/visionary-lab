"""Pure operations on a DesignBrief.

This module is intentionally LLM-free: every function takes typed Python data
in and returns typed Python data out, so callers can unit-test brief logic
without mocking an OpenAI client.

Exports:

* ``ResolvedObject`` — the projection emitted by the resolver. ``brief_to_prompts``
  reads this; nothing reaches into ``ObjectEntry`` directly when building
  prompts.
* ``resolve_objects_for_image(brief, room_id)`` — palette + per-image-override
  projection.
* ``reconcile_overrides_by_name(prev_brief, new_brief)`` — issue 004 of the
  per-image-object-quantities PRD: lifts per-image overrides from ``prev_brief``
  onto ``new_brief`` by matching palette entries on case-insensitive,
  whitespace-trimmed name. Returns ``(updated_brief, ReconcileSummary)``.
* ``migrate_legacy_plant_palette(raw)`` — pure, idempotent legacy → generic
  dict converter. Drives the ``DesignBrief`` ``model_validator(mode='before')``
  AND the GET-project endpoint's opportunistic write-back.
"""
from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.models.design_brief import (
        DesignBrief,
        ObjectCategory,
        ReconcileSummary,
    )


# ---------------------------------------------------------------------------
# Tree-heuristic helpers — module-private, pure.
# ---------------------------------------------------------------------------

# Common tree-name tokens for the legacy migration heuristic. Word-boundary
# matched, case-insensitive — so "oak" matches "Bur Oak" but NOT "oakleaf".
_TREE_TOKENS = (
    "pine",
    "spruce",
    "oak",
    "maple",
    "birch",
    "cedar",
    "fir",
    "willow",
    "magnolia",
    "redwood",
    "sequoia",
    "palm",
    "elm",
    "ash",
    "juniper",
    "cypress",
)
_TREE_TOKEN_RE = re.compile(
    r"\b(?:" + "|".join(_TREE_TOKENS) + r")\b", re.IGNORECASE
)
# All numeric values in a size string. ``max(...) >= 6`` paired with an
# ft / feet / tall hint counts as "tall enough to be a tree".
_NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?")
_TALL_HINT_RE = re.compile(r"\b(?:ft|feet|tall)\b", re.IGNORECASE)


def _looks_like_tree(species: str, size: str) -> bool:
    """Return True if a legacy plant entry should migrate into the TREE
    category, False otherwise (caller defaults to PLANT)."""
    if species and _TREE_TOKEN_RE.search(species):
        return True
    if size and _TALL_HINT_RE.search(size):
        nums = [float(m) for m in _NUMERIC_RE.findall(size)]
        if nums and max(nums) >= 6:
            return True
    return False


# ---------------------------------------------------------------------------
# ResolvedObject — projection consumed by brief_to_prompts.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedObject:
    """Frozen projection of an effective object after applying overrides.

    Carries everything ``brief_to_prompts`` needs so prompt-building code
    never reaches back into the brief model. ``quantity`` is the *effective*
    count (palette default in this slice; per-image override in issue 003).
    """

    id: str
    name: str
    description: Optional[str]
    category: "ObjectCategory"
    quantity: int
    size: str
    placement: str
    visual_notes: Optional[str]


# ---------------------------------------------------------------------------
# resolve_objects_for_image — basic projection (issue 001 scope).
# ---------------------------------------------------------------------------


def resolve_objects_for_image(brief: "DesignBrief", room_id: str) -> List[ResolvedObject]:
    """Return the effective object list for a single image.

    Applies per-image overrides on top of the palette, per the
    per-image-object-quantities PRD's "Resolution logic" section:

    * Start from ``brief.object_palette`` in declaration order.
    * Build an override-by-object_id dict from ``brief.per_image_objects[room_id]``.
      If multiple overrides target the same ``object_id`` (which shouldn't
      happen via the normal frontend flow), the LAST wins — a defensive
      last-write-wins contract that keeps resolution deterministic when
      duplicates somehow leak through.
    * For each ``ObjectEntry``: if the matching override has ``enabled=False``
      OR ``quantity=0``, omit the object from the result entirely (the two
      flags are equivalent skip signals).
    * Effective quantity: override quantity (when non-zero / enabled), else
      ``default_quantity`` from the palette.
    * Effective placement: override.placement when non-None, else palette
      placement. (The model's field validator coerces empty / whitespace
      strings to None upstream, so this branch reliably means "inherit".)
    * Overrides whose ``object_id`` is not in the palette are silently
      ignored — palette is the source of truth.
    """
    overrides_by_id: Dict[str, Any] = {}
    for ovr in brief.per_image_objects.get(room_id, []):
        # Last-write-wins on duplicate object_ids. The frontend never emits
        # duplicates, but legacy / hand-edited data could.
        overrides_by_id[ovr.object_id] = ovr

    resolved: List[ResolvedObject] = []
    for obj in brief.object_palette:
        ovr = overrides_by_id.get(obj.id)
        if ovr is not None:
            if not ovr.enabled or ovr.quantity == 0:
                continue
            quantity = ovr.quantity
            placement = obj.placement if ovr.placement is None else ovr.placement
        else:
            quantity = obj.default_quantity
            placement = obj.placement

        resolved.append(
            ResolvedObject(
                id=obj.id,
                name=obj.name,
                description=obj.description,
                category=obj.category,
                quantity=quantity,
                size=obj.size,
                placement=placement,
                visual_notes=obj.visual_notes,
            )
        )

    return resolved


# ---------------------------------------------------------------------------
# migrate_legacy_plant_palette — pure, idempotent legacy → generic dict.
# ---------------------------------------------------------------------------


def migrate_legacy_plant_palette(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a legacy DesignBrief raw dict (with ``plant_palette``) to the
    new generic shape (with ``object_palette``).

    Pure (does not mutate ``raw``) and idempotent: applying the function to
    an already-migrated dict returns the same object, so callers can use
    ``result is raw`` to short-circuit no-op writebacks.

    Behaviour:

    * If ``object_palette`` is already in ``raw``, the input is returned
      unchanged (``result is raw``).
    * Otherwise each entry under ``plant_palette`` is converted to the new
      ``ObjectEntry`` shape: fresh UUID ``id``; ``species`` → ``name``;
      ``botanical_name`` → ``description``; ``quantity`` → ``default_quantity``;
      ``size``, ``placement``, ``visual_notes`` copied through; ``category``
      chosen heuristically (TREE if tall-or-tree-named, else PLANT).
    * The legacy ``plant_palette`` key is dropped.
    * ``per_image_objects`` is initialised to ``{}`` if missing.
    """
    if "object_palette" in raw:
        return raw  # idempotent — same object, no-op.

    result: Dict[str, Any] = dict(raw)
    legacy_palette = result.pop("plant_palette", []) or []

    migrated_palette: List[Dict[str, Any]] = []
    for legacy_entry in legacy_palette:
        if not isinstance(legacy_entry, dict):
            continue
        species = legacy_entry.get("species") or ""
        size = legacy_entry.get("size") or ""
        category = "tree" if _looks_like_tree(species, size) else "plant"

        new_entry: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "name": species,
            "description": legacy_entry.get("botanical_name"),
            "category": category,
            "default_quantity": legacy_entry.get("quantity", 1),
            "size": size,
            "placement": legacy_entry.get("placement", "") or "",
            "visual_notes": legacy_entry.get("visual_notes"),
        }
        migrated_palette.append(new_entry)

    result["object_palette"] = migrated_palette
    if "per_image_objects" not in result:
        result["per_image_objects"] = {}

    return result


__all__ = [
    "ResolvedObject",
    "resolve_objects_for_image",
    "reconcile_overrides_by_name",
    "migrate_legacy_plant_palette",
]


# ---------------------------------------------------------------------------
# reconcile_overrides_by_name — issue 004 of the per-image-object-quantities
# PRD. Carries surviving per-image overrides from a prior brief into a freshly
# regenerated brief by matching palette entries on case-insensitive,
# whitespace-trimmed name.
# ---------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Canonical name for matching: trim + lowercase. Keep the rule simple
    and explicit so the test suite can assert exact behaviour. Two names
    that produce the same canonical string are considered "the same"
    object across regeneration.
    """
    return (name or "").strip().lower()


def reconcile_overrides_by_name(
    prev_brief: "DesignBrief",
    new_brief: "DesignBrief",
) -> Tuple["DesignBrief", "ReconcileSummary"]:
    """Lift per-image overrides from ``prev_brief`` onto ``new_brief``.

    Matching rule: case-insensitive, whitespace-trimmed name match between
    ``prev_brief.object_palette`` and ``new_brief.object_palette``. For each
    prior override we look up the prior ``object_id``'s name in the prev
    palette, then look that name up in the new palette to get the new
    ``object_id``.

    Drop conditions (each counted in ``summary.dropped``):

    * Override's ``object_id`` is not in the prev palette (orphan / legacy
      data).
    * Object name is duplicated in the prev palette (ambiguous prev → can't
      identify which one the override referred to).
    * Object name is not present in the new palette (renamed / removed).
    * Object name is duplicated in the new palette (ambiguous new → can't
      decide which UUID to map onto).

    Conflict resolution: if both ``prev_brief`` and ``new_brief`` have an
    override for the same ``(room_id, normalized name)`` pair, the prev
    override wins. Rationale: prev overrides are user edits; new overrides
    pre-populated by the LLM are auto-suggestions that should never silently
    overwrite a user edit.

    The returned ``DesignBrief`` is a NEW instance; ``prev_brief`` and
    ``new_brief`` are not mutated.
    """
    # Local imports avoid a top-level cycle: brief_resolver is imported by
    # design_brief's model_validator.
    from backend.models.design_brief import DesignBrief, ReconcileSummary

    # Step 1: build name maps for prev and new palettes. Track normalized-
    # name duplicates so we can drop ambiguous matches deterministically.
    prev_name_counts = Counter(_normalize_name(e.name) for e in prev_brief.object_palette)
    prev_id_to_name: Dict[str, str] = {}
    for e in prev_brief.object_palette:
        normalized = _normalize_name(e.name)
        if prev_name_counts[normalized] > 1:
            # Ambiguous: leave the id out of the lookup table so any
            # override referring to it falls into the "dropped" bucket.
            continue
        prev_id_to_name[e.id] = normalized

    new_name_counts = Counter(_normalize_name(e.name) for e in new_brief.object_palette)
    new_name_to_id: Dict[str, str] = {}
    for e in new_brief.object_palette:
        normalized = _normalize_name(e.name)
        if new_name_counts[normalized] > 1:
            continue
        new_name_to_id[normalized] = e.id

    # Step 2: build the reconciled per_image_objects on top of new_brief's
    # existing dict (deep-copied so we don't mutate the input).
    reconciled_per_image: Dict[str, List[Any]] = {
        room_id: [ovr.copy() for ovr in overrides]
        for room_id, overrides in new_brief.per_image_objects.items()
    }

    carried_forward = 0
    dropped = 0

    # Step 3: walk prev overrides and lift each into the reconciled dict.
    for room_id, overrides in prev_brief.per_image_objects.items():
        for prev_override in overrides:
            prev_name = prev_id_to_name.get(prev_override.object_id)
            if prev_name is None:
                # Orphan in prev OR ambiguous prev name.
                dropped += 1
                continue
            new_id = new_name_to_id.get(prev_name)
            if new_id is None:
                # Renamed / removed in new OR ambiguous new name.
                dropped += 1
                continue

            # Build the carried-forward override with the rewritten id.
            carried_override = prev_override.copy(update={"object_id": new_id})

            # Insert into reconciled_per_image: prev wins on (room_id,
            # object_id) conflict — replace any existing entry, else
            # append.
            room_list = reconciled_per_image.setdefault(room_id, [])
            replaced = False
            for i, existing in enumerate(room_list):
                if existing.object_id == new_id:
                    room_list[i] = carried_override
                    replaced = True
                    break
            if not replaced:
                room_list.append(carried_override)

            carried_forward += 1

    # Step 4: assemble the new brief by copying new_brief and replacing
    # its per_image_objects. Use model_copy / dict round-trip to keep
    # this pydantic-version-agnostic.
    updated = new_brief.copy(update={"per_image_objects": reconciled_per_image})

    return updated, ReconcileSummary(
        carried_forward=carried_forward, dropped=dropped
    )
