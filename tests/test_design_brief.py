"""Tests for DesignBrief and related Pydantic models."""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import ValidationError


def test_plant_entry_defaults():
    from backend.models.design_brief import PlantEntry
    p = PlantEntry(species="Vanderwolf's Pyramid Limber Pine")
    assert p.species == "Vanderwolf's Pyramid Limber Pine"
    assert p.quantity == 1
    assert p.botanical_name is None
    assert p.visual_notes is None


def test_plant_entry_full():
    from backend.models.design_brief import PlantEntry
    p = PlantEntry(
        species="Baby Blue Eyes Spruce",
        botanical_name="Picea pungens 'Baby Blue Eyes'",
        quantity=3,
        size="15-30 ft",
        placement="back row along fence",
        visual_notes="Intense powder-blue to steel-blue needles",
    )
    assert p.quantity == 3
    assert "powder-blue" in p.visual_notes


def test_placement_guide_defaults():
    from backend.models.design_brief import PlacementGuide
    pg = PlacementGuide()
    assert pg.back_row == ""
    assert pg.middle_row is None
    assert pg.front_row is None
    assert pg.accent_areas is None


def test_design_brief_valid():
    from backend.models.design_brief import DesignBrief, PlantEntry, PlacementGuide
    brief = DesignBrief(
        global_instructions="Add layered evergreen privacy screen along fence",
        plant_palette=[
            PlantEntry(species="Columnar Norway Spruce", quantity=5, size="20 ft", placement="east fence"),
        ],
        placement_guide=PlacementGuide(back_row="Tall conifers along fence"),
        preserve_elements=["patio", "fire pit", "pergola"],
    )
    assert len(brief.plant_palette) == 1
    assert brief.plant_palette[0].species == "Columnar Norway Spruce"
    assert "patio" in brief.preserve_elements
    assert brief.per_image_notes == {}
    assert brief.settings.model == "gpt-image-2"


def test_design_brief_requires_global_instructions():
    from backend.models.design_brief import DesignBrief
    with pytest.raises(ValidationError):
        DesignBrief()


def test_image_analysis_model():
    from backend.models.design_brief import ImageAnalysis
    a = ImageAnalysis(
        room_id="room-1",
        description="Backyard with wooden fence and turf",
        features=["fence", "turf", "shrubs"],
        zones=["fence_line", "open_turf"],
    )
    assert len(a.features) == 3
    assert "fence_line" in a.zones


def test_chat_request_model():
    from backend.models.design_brief import ChatRequest, ChatMessage
    req = ChatRequest(
        message="I want trees along the fence",
        conversation_history=[
            ChatMessage(role="assistant", content="I've analyzed your photos."),
        ],
        focused_image_id="room-123",
    )
    assert req.message == "I want trees along the fence"
    assert len(req.conversation_history) == 1
    assert req.focused_image_id == "room-123"


def test_chat_response_model():
    from backend.models.design_brief import ChatResponse
    resp = ChatResponse(
        reply="Great choice! What species?",
        ready_for_brief=False,
        suggested_actions=["specify_species", "choose_density"],
    )
    assert not resp.ready_for_brief
    assert len(resp.suggested_actions) == 2


def test_generate_brief_request_model():
    from backend.models.design_brief import GenerateBriefRequest, ChatMessage
    req = GenerateBriefRequest(
        conversation_history=[
            ChatMessage(role="user", content="Add trees"),
            ChatMessage(role="assistant", content="What kind?"),
        ],
    )
    assert len(req.conversation_history) == 2


@pytest.mark.asyncio
async def test_brief_generation_from_conversation():
    from backend.core.brief_generator import BriefGeneratorService
    from backend.models.design_brief import ChatMessage, ImageAnalysis

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({
            "global_instructions": "Add layered evergreen privacy screen along fence line",
            "plant_palette": [
                {
                    "species": "Vanderwolf's Pyramid Limber Pine",
                    "botanical_name": "Pinus flexilis 'Vanderwolf's Pyramid'",
                    "quantity": 3,
                    "size": "8-10 ft",
                    "placement": "back row along east fence",
                    "visual_notes": "Silvery-blue twisted needles, narrow pyramid form",
                }
            ],
            "placement_guide": {"back_row": "Tall conifers", "middle_row": None, "front_row": None},
            "preserve_elements": ["patio", "fire pit"],
            "per_image_notes": {},
        })))]
    ))

    analyses = [
        ImageAnalysis(room_id="r1", description="Fence line", features=["fence"], zones=["fence_line"]),
    ]
    history = [
        ChatMessage(role="user", content="Add Vanderwolf Pine along the fence, 3 trees, back row"),
    ]

    service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
    brief = await service.generate_brief(conversation_history=history, image_analyses=analyses)

    assert brief.global_instructions == "Add layered evergreen privacy screen along fence line"
    assert len(brief.plant_palette) == 1
    assert brief.plant_palette[0].species == "Vanderwolf's Pyramid Limber Pine"
    assert "patio" in brief.preserve_elements


@pytest.mark.asyncio
async def test_brief_to_prompts_produces_specific_prompts():
    from backend.core.brief_generator import BriefGeneratorService
    from backend.models.design_brief import (
        DesignBrief, PlantEntry, PlacementGuide, ImageAnalysis,
    )

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps([
            "Along the fence in this backyard, add 3 Vanderwolf's Pyramid Limber Pines with silvery-blue needles in back row",
            "Place 3 narrow pyramid conifers with twisted blue-green needles along the wooden fence line",
        ])))]
    ))

    brief = DesignBrief(
        global_instructions="Add trees along fence",
        plant_palette=[
            PlantEntry(
                species="Vanderwolf's Pyramid Limber Pine",
                quantity=3,
                size="8-10 ft",
                placement="back row along fence",
                visual_notes="Silvery-blue twisted needles, narrow pyramid form",
            ),
        ],
        placement_guide=PlacementGuide(back_row="Tall conifers"),
        preserve_elements=["patio"],
    )
    analyses = [
        ImageAnalysis(room_id="r1", description="Fence line view", features=["fence"], zones=["fence_line"]),
    ]

    service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
    prompts = await service.brief_to_prompts(brief=brief, image_analyses=analyses, n_variations=2)

    assert "r1" in prompts
    assert len(prompts["r1"]) == 2
    assert any("Vanderwolf" in p or "silvery" in p.lower() for p in prompts["r1"])
