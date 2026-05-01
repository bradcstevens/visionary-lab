"""Pydantic models for the virtual staging feature."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ProjectStatus(str, Enum):
    UPLOADING = "uploading"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ItemStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class StagingSettings(BaseModel):
    variations_per_room: int = Field(5, description="Number of variations to generate per room (1-10)")
    model: str = Field("gpt-image-2", description="Image generation model")
    quality: str = Field("high", description="Image quality setting")
    size: str = Field("auto", description="Image size")

    @validator("variations_per_room")
    def validate_variations(cls, v):
        if v < 1 or v > 10:
            raise ValueError("variations_per_room must be between 1 and 10")
        return v


class GenerationMetadata(BaseModel):
    model: Optional[str] = None
    adapted_prompt: Optional[str] = None
    tokens_used: Optional[int] = None
    generation_time_ms: Optional[int] = None


class Variation(BaseModel):
    id: str
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    status: str = Field(ItemStatus.PENDING, description="Variation generation status")
    error: Optional[str] = None
    generation_metadata: Optional[GenerationMetadata] = None


class Room(BaseModel):
    id: str
    label: str = Field(..., description="Room label, e.g. 'Living Room', 'Backyard'")
    original_image_url: str = Field(..., description="Blob storage URL of uploaded original")
    original_thumbnail_url: Optional[str] = None
    status: str = Field(ItemStatus.PENDING, description="Room processing status")
    error: Optional[str] = None
    variations: List[Variation] = Field(default_factory=list)
    prompt_addendum: Optional[str] = Field(
        None,
        description=(
            "Optional per-room clarification appended to the project prompt for "
            "every future generation of this room. Existing variations are not "
            "regenerated when this changes."
        ),
    )


class StagingProject(BaseModel):
    id: str
    name: str
    prompt: str
    status: str = Field(ProjectStatus.UPLOADING, description="Overall project status")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    rooms: List[Room] = Field(default_factory=list)
    settings: StagingSettings = Field(default_factory=StagingSettings)
    folder_path: Optional[str] = None
    design_brief: Optional[Dict[str, Any]] = Field(None, description="Structured design brief from AI conversation")
    analyses: Optional[List[Dict[str, Any]]] = Field(None, description="Image analysis results")


class CreateProjectRequest(BaseModel):
    name: str = Field(..., description="Project name", examples=["Modern Minimalist Refresh"])
    prompt: str = Field(..., description="Overall styling direction", examples=["Clean lines, warm wood tones, lots of greenery"])
    settings: StagingSettings = Field(default_factory=StagingSettings)


class UpdateRoomRequest(BaseModel):
    """Partial-update body for ``PATCH /projects/{id}/rooms/{rid}``.

    Only ``prompt_addendum`` is editable today (issue 003 of the
    projects-page-improvements PRD). Other Room fields (``status``,
    ``variations``, ``original_image_url``) are derived from generation
    activity and must not be edited through this endpoint.

    ``prompt_addendum=None`` is meaningful: it explicitly clears the
    addendum. The endpoint normalizes empty / whitespace-only strings to
    ``None`` before persisting so the model stays clean.
    """
    prompt_addendum: Optional[str] = Field(
        None,
        description=(
            "Per-room clarification appended to the project prompt for future "
            "generations of this room. Pass null or omit to clear."
        ),
    )


class UploadRoomsResponse(BaseModel):
    project_id: str
    rooms_added: int
    rooms: List[Room]


class ProjectResponse(BaseModel):
    project: StagingProject


class ProjectListResponse(BaseModel):
    projects: List[StagingProject]
    total: int
