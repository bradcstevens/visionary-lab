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
    # ``per_image_objects`` is reserved for issue 003 of the
    # per-image-object-quantities PRD. The migration helper initialises it
    # whenever a legacy brief is migrated, so the round-trip shape is stable.
    # Until 003 lands the inner items round-trip as raw dicts.
    per_image_objects: Dict[str, List[Dict[str, Any]]] = Field(
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
