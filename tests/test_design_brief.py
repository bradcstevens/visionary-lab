"""Tests for DesignBrief and related Pydantic models."""
import pytest
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
