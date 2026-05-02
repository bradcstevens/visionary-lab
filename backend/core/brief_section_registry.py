"""Single source of truth for the eight canonical Design Brief sections.

The wizard step config and the settings-panel tab config both consume this
registry so the two surfaces always expose the same sections in the same
order. Adding a section in the future is a one-line change here.

Per the image-pipeline-and-project-ux-overhaul PRD § BriefSectionRegistry
and user stories 32, 35.

The registry is intentionally tiny (one dataclass + a frozen tuple + a
handful of pure-lookup helpers). Section content lives on
``DesignBrief.sections`` keyed by ``BriefSection.id``; rendering to the
top-level prompt markdown lives in ``PromptComposer.compose_brief_markdown``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class BriefSection:
    """One canonical section of the structured Design Brief.

    Attributes:
        id: Stable machine identifier — used as the key in
            ``DesignBrief.sections`` and as the section's URL slug. Lower
            snake_case. NEVER change once shipped.
        title: Display title used as the rendered markdown ``## <title>``
            heading and as the wizard step / settings-panel tab label.
            Round-trip extraction matches headings to ids via a
            normalised lookup so ``title`` text MAY be polished without
            breaking persisted briefs.
        description: One-sentence helper text for the wizard step / tab
            tooltip. Not rendered into the prompt markdown.
    """

    id: str
    title: str
    description: str


# The eight canonical sections, in the rendered + wizard order pinned by
# the PRD. Order is part of the contract — both the markdown output of
# ``PromptComposer.compose_brief_markdown`` and the wizard step sequence
# follow this list verbatim.
SECTIONS: Tuple[BriefSection, ...] = (
    BriefSection(
        id="edit_task",
        title="Edit Task",
        description="What overall change should the model make to the scene?",
    ),
    BriefSection(
        id="edit_zone",
        title="Edit Zone",
        description="Which area of the image is in scope for editing?",
    ),
    BriefSection(
        id="do_not_alter",
        title="Do Not Alter",
        description="Elements that must remain unchanged across renders.",
    ),
    BriefSection(
        id="object_palette",
        title="Object Palette",
        description="The set of objects available for placement in the scene.",
    ),
    BriefSection(
        id="arrangement",
        title="Arrangement",
        description="How objects should be composed and positioned.",
    ),
    BriefSection(
        id="regional_constraints",
        title="Regional Constraints",
        description="Climate, plant-hardiness, code, or other regional rules.",
    ),
    BriefSection(
        id="aesthetic_goal",
        title="Aesthetic Goal",
        description="The overall visual style, mood, or design intent.",
    ),
    BriefSection(
        id="scale_fidelity",
        title="Scale & Fidelity",
        description="Sizing, level of detail, and rendering fidelity expectations.",
    ),
)


# Lookup tables built once at import time. Both lookups are O(1).
_BY_ID: Dict[str, BriefSection] = {s.id: s for s in SECTIONS}


def _normalise_title(text: str) -> str:
    """Return a lookup key that survives whitespace, case, and punctuation
    drift in section titles. Used by ``title_to_id`` so the round-trip
    ``sections → markdown → sections`` works even if a future title is
    polished from "Scale & Fidelity" to "Scale and fidelity"."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


_BY_NORM_TITLE: Dict[str, BriefSection] = {_normalise_title(s.title): s for s in SECTIONS}


def section_ids() -> Tuple[str, ...]:
    """Return the eight canonical section ids in registry order."""
    return tuple(s.id for s in SECTIONS)


def get_section(section_id: str) -> BriefSection:
    """Return the ``BriefSection`` for ``section_id``.

    Raises ``KeyError`` if the id is not one of the eight canonical
    sections — the caller is asking for a section that does not exist,
    which is a programming error, not user input.
    """
    return _BY_ID[section_id]


def title_to_id(title: str) -> Optional[str]:
    """Return the section id whose title matches ``title``, or ``None``.

    Match is case-insensitive and ignores punctuation / whitespace so
    ``## Scale & Fidelity`` and ``## scale and fidelity`` both resolve
    to ``scale_fidelity``. ``None`` is returned for headings that don't
    correspond to any registered section so extractors can route unknown
    headings to a free-form bucket without raising.
    """
    section = _BY_NORM_TITLE.get(_normalise_title(title))
    return section.id if section is not None else None
