"""Pydantic models for the AI Design Session and Design Brief."""
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from backend.models.staging import StagingSettings


class PlantEntry(BaseModel):
    species: str = Field(..., description="Common name, e.g. 'Vanderwolf's Pyramid Limber Pine'")
    botanical_name: Optional[str] = Field(None, description="e.g. 'Pinus flexilis Vanderwolf's Pyramid'")
    quantity: int = Field(1, description="Number of this species to place")
    size: str = Field("", description="e.g. '8-10 ft tall'")
    placement: str = Field("", description="e.g. 'back row along east fence'")
    visual_notes: Optional[str] = Field(None, description="Key visual characteristics for image generation")


class PlacementGuide(BaseModel):
    back_row: str = Field("", description="Tall plants / trees description")
    middle_row: Optional[str] = Field(None, description="Mid-height shrubs description")
    front_row: Optional[str] = Field(None, description="Low groundcover description")
    accent_areas: Optional[str] = Field(None, description="Special areas like pergola posts, patio edges")


class DesignBrief(BaseModel):
    global_instructions: str = Field(..., description="Overall styling direction synthesized from conversation")
    plant_palette: List[PlantEntry] = Field(default_factory=list)
    placement_guide: PlacementGuide = Field(default_factory=PlacementGuide)
    per_image_notes: Dict[str, str] = Field(default_factory=dict, description="room_id → specific note")
    preserve_elements: List[str] = Field(default_factory=list, description="Elements to keep unchanged")
    settings: StagingSettings = Field(default_factory=StagingSettings)


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
