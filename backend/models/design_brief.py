"""Pydantic models for the AI Design Session and Design Brief."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.models.staging import StagingSettings


class ObjectCategory(str, Enum):
    """Generic object category — UI-facing metadata only.

    Resolution / prompt-building code MUST NOT branch on category — see the
    per-image-object-quantities PRD's ``Further Notes`` for the rationale.
    """

    PLANT = "plant"
    TREE = "tree"
    ROCK = "rock"
    FURNITURE = "furniture"
    LIGHTING = "lighting"
    HARDSCAPE = "hardscape"
    DECOR = "decor"
    OTHER = "other"


# Direct synonyms / common LLM mis-spellings → enum value. Applied AFTER the
# lowercase-and-strip-trailing-s normalisation pass.
_CATEGORY_SYNONYMS: Dict[str, str] = {
    "shrub": "plant",
    "bush": "plant",
    "light": "lighting",
}


def _coerce_category(raw: Any) -> ObjectCategory:
    """Tolerant coercion: silently maps stray inputs (None, mis-cased,
    pluralised, hallucinated) to the closest ObjectCategory; falls back to
    OTHER rather than raising. The PRD explicitly forbids resolution-time
    branching on category, so a misclassification is cosmetic, not harmful.
    """
    if isinstance(raw, ObjectCategory):
        return raw
    if not isinstance(raw, str):
        return ObjectCategory.OTHER

    normalised = raw.strip().lower()
    if not normalised:
        return ObjectCategory.OTHER
    # Strip a single trailing 's' to handle naive pluralisation
    # ("plants" → "plant", "rocks" → "rock"). Don't strip "ss"-suffixed
    # words; we have none in the enum today, but be defensive anyway.
    if normalised.endswith("s") and not normalised.endswith("ss"):
        normalised = normalised[:-1]
    normalised = _CATEGORY_SYNONYMS.get(normalised, normalised)
    try:
        return ObjectCategory(normalised)
    except ValueError:
        return ObjectCategory.OTHER


class ObjectEntry(BaseModel):
    """One object in the project palette.

    Replaces the old plant-specific ``PlantEntry``. Carries a stable UUID
    ``id`` so per-image overrides (issue 003) can reference it without
    coupling to a name string.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Display name, e.g. 'Vanderwolf Pine' or 'Adirondack chair'")
    description: Optional[str] = Field(
        None, description="Optional detail (botanical name, model number, etc.)"
    )
    category: ObjectCategory = Field(ObjectCategory.OTHER)
    default_quantity: int = Field(1, ge=0, description="Default placement count, project-wide")
    size: str = Field("", description="Free-form size description, e.g. '8-10 ft tall'")
    placement: str = Field("", description="Free-form placement guidance")
    visual_notes: Optional[str] = Field(None, description="Visual characteristics for image generation")

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, v: Any) -> ObjectCategory:
        return _coerce_category(v)


class ImageObjectOverride(BaseModel):
    """One per-image override targeting a single ObjectEntry by id.

    Stored sparsely under ``DesignBrief.per_image_objects[room_id]``. Only
    rows that actually differ from the palette defaults need to be present;
    the resolver applies overrides on top of the palette at render time.

    Skip semantics: ``enabled=False`` AND ``quantity=0`` are equivalent
    skip signals. The resolver omits the object from the rendered scene if
    either is true. The frontend canonical form for "skip" is
    ``{enabled=False, quantity=0, placement=None}`` — but the resolver does
    not enforce that contract; either flag suffices.
    """

    object_id: str = Field(..., description="The ObjectEntry.id this override targets")
    # ``quantity`` is REQUIRED — there is no "inherit palette default" sentinel.
    # The frontend pre-fills the input with palette ``default_quantity`` before
    # any user edit, so any persisted override carries an explicit count. A
    # missing quantity in deserialized data is a programming error, not a
    # legitimate "skip" signal. Issue 003 critique: defaulting to 0 would
    # silently turn ``{object_id: "x"}`` into a skip; we reject that here.
    quantity: int = Field(..., ge=0, description="Effective count for this image (0 = skip)")
    # ``placement is None`` means "inherit palette placement"; any other
    # string replaces it. The mode='before' validator coerces empty /
    # whitespace strings to None so a frontend bug can't persist '' as a
    # blanking override.
    placement: Optional[str] = Field(None, description="None inherits, any other string replaces")
    enabled: bool = Field(True, description="False = skip this object in this image")

    @field_validator("placement", mode="before")
    @classmethod
    def _normalise_placement(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            return v  # let the type checker raise its own error
        stripped = v.strip()
        return stripped if stripped else None


class PlacementGuide(BaseModel):
    back_row: str = Field("", description="Tall plants / trees description")
    middle_row: Optional[str] = Field(None, description="Mid-height shrubs description")
    front_row: Optional[str] = Field(None, description="Low groundcover description")
    accent_areas: Optional[str] = Field(None, description="Special areas like pergola posts, patio edges")


class DesignBrief(BaseModel):
    global_instructions: str = Field(..., description="Overall styling direction synthesized from conversation")
    object_palette: List[ObjectEntry] = Field(default_factory=list)
    placement_guide: PlacementGuide = Field(default_factory=PlacementGuide)
    per_image_notes: Dict[str, str] = Field(default_factory=dict, description="room_id → specific note")
    # ``per_image_objects`` carries sparse per-image overrides keyed by
    # ``room_id``. Each list entry is an ``ImageObjectOverride`` that targets
    # one ``ObjectEntry`` by id. The resolver in ``brief_resolver`` applies
    # these on top of the palette per the per-image-object-quantities PRD.
    per_image_objects: Dict[str, List[ImageObjectOverride]] = Field(
        default_factory=dict, description="room_id → list of per-image overrides"
    )
    preserve_elements: List[str] = Field(default_factory=list, description="Elements to keep unchanged")
    settings: StagingSettings = Field(default_factory=StagingSettings)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_palette(cls, data: Any) -> Any:
        """Auto-migrate any legacy raw dict (``plant_palette`` shape) to the
        generic ``object_palette`` shape on construction. Combined with the
        GET-project endpoint's opportunistic write-back, this guarantees no
        code path can surface legacy keys to the UI.
        """
        if isinstance(data, dict):
            # Local import avoids a circular dependency: brief_resolver
            # references ObjectCategory for typing only via TYPE_CHECKING.
            from backend.core.brief_resolver import migrate_legacy_plant_palette

            return migrate_legacy_plant_palette(data)
        return data


class ImageAnalysis(BaseModel):
    room_id: str
    description: str = Field(..., description="What the AI sees in this image")
    features: List[str] = Field(default_factory=list, description="Detected features: fence, turf, patio, etc.")
    zones: List[str] = Field(default_factory=list, description="Identifiable areas for object placement")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    focused_image_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's latest message")
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    focused_image_id: Optional[str] = Field(None, description="Room ID the user is focused on")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="AI's response text")
    ready_for_brief: bool = Field(False, description="True when AI has enough info to generate a brief")
    suggested_actions: List[str] = Field(default_factory=list, description="Suggested quick-reply action keys")


class GenerateBriefRequest(BaseModel):
    conversation_history: List[ChatMessage] = Field(default_factory=list, description="Full chat history for brief synthesis")
