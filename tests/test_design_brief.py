"""Tests for DesignBrief and related Pydantic models."""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import ValidationError


def test_object_entry_defaults():
    from backend.models.design_brief import ObjectCategory, ObjectEntry
    e = ObjectEntry(name="Vanderwolf's Pyramid Limber Pine")
    assert e.name == "Vanderwolf's Pyramid Limber Pine"
    assert e.default_quantity == 1
    assert e.description is None
    assert e.visual_notes is None
    # default category is OTHER (no auto-detection on bare construction)
    assert e.category == ObjectCategory.OTHER
    # default_factory id should be a non-empty UUID-shaped string
    assert isinstance(e.id, str) and len(e.id) > 0


def test_object_entry_full():
    from backend.models.design_brief import ObjectCategory, ObjectEntry
    e = ObjectEntry(
        name="Baby Blue Eyes Spruce",
        description="Picea pungens 'Baby Blue Eyes'",
        category="tree",
        default_quantity=3,
        size="15-30 ft",
        placement="back row along fence",
        visual_notes="Intense powder-blue to steel-blue needles",
    )
    assert e.default_quantity == 3
    assert e.category == ObjectCategory.TREE
    assert "powder-blue" in e.visual_notes


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("plant", "plant"),
        ("Plant", "plant"),
        (" PLANT ", "plant"),
        ("plants", "plant"),       # naive plural
        ("Trees", "tree"),
        ("shrub", "plant"),         # synonym
        ("bush", "plant"),          # synonym
        ("light", "lighting"),      # synonym
        ("LIGHTING", "lighting"),
        ("hardscape", "hardscape"),
        ("decor", "decor"),
        ("rocks", "rock"),
        ("furniture", "furniture"),
        ("plant_tree_hybrid", "other"),  # unknown → OTHER, no raise
        ("", "other"),
        (None, "other"),
        (42, "other"),                    # non-string → OTHER, no raise
    ],
)
def test_object_entry_category_coercion(raw, expected):
    from backend.models.design_brief import ObjectCategory, ObjectEntry
    e = ObjectEntry(name="x", category=raw)
    assert e.category == ObjectCategory(expected)


def test_design_brief_auto_migrates_legacy_dict():
    """DesignBrief(**raw) MUST migrate legacy plant_palette transparently —
    this is what guarantees old persisted briefs deserialise on read."""
    from backend.models.design_brief import DesignBrief
    legacy_raw = {
        "global_instructions": "Add evergreens",
        "plant_palette": [
            {
                "species": "Sequoia",
                "botanical_name": "Sequoiadendron giganteum",
                "quantity": 2,
                "size": "20 ft tall",
                "placement": "north fence",
                "visual_notes": "tall, conical",
            }
        ],
        "placement_guide": {"back_row": "Tall conifers"},
        "preserve_elements": ["patio"],
    }
    brief = DesignBrief(**legacy_raw)
    assert len(brief.object_palette) == 1
    obj = brief.object_palette[0]
    assert obj.name == "Sequoia"
    assert obj.description == "Sequoiadendron giganteum"
    assert obj.default_quantity == 2
    # Sequoia in species → TREE
    from backend.models.design_brief import ObjectCategory
    assert obj.category == ObjectCategory.TREE
    assert isinstance(obj.id, str) and len(obj.id) > 0
    # per_image_objects always initialised by migration
    assert brief.per_image_objects == {}


def test_placement_guide_defaults():
    from backend.models.design_brief import PlacementGuide
    pg = PlacementGuide()
    assert pg.back_row == ""
    assert pg.middle_row is None
    assert pg.front_row is None
    assert pg.accent_areas is None


def test_design_brief_valid():
    from backend.models.design_brief import DesignBrief, ObjectEntry, PlacementGuide
    brief = DesignBrief(
        global_instructions="Add layered evergreen privacy screen along fence",
        object_palette=[
            ObjectEntry(name="Columnar Norway Spruce", category="tree", default_quantity=5, size="20 ft", placement="east fence"),
        ],
        placement_guide=PlacementGuide(back_row="Tall conifers along fence"),
        preserve_elements=["patio", "fire pit", "pergola"],
    )
    assert len(brief.object_palette) == 1
    assert brief.object_palette[0].name == "Columnar Norway Spruce"
    assert "patio" in brief.preserve_elements
    assert brief.per_image_notes == {}
    assert brief.per_image_objects == {}
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
            "object_palette": [
                {
                    "name": "Vanderwolf's Pyramid Limber Pine",
                    "description": "Pinus flexilis 'Vanderwolf's Pyramid'",
                    "category": "tree",
                    "default_quantity": 3,
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
    assert len(brief.object_palette) == 1
    assert brief.object_palette[0].name == "Vanderwolf's Pyramid Limber Pine"
    assert "patio" in brief.preserve_elements
    # generate_brief must assign a UUID id
    assert isinstance(brief.object_palette[0].id, str) and len(brief.object_palette[0].id) > 0


@pytest.mark.asyncio
async def test_brief_to_prompts_produces_specific_prompts():
    from backend.core.brief_generator import BriefGeneratorService
    from backend.models.design_brief import (
        DesignBrief, ObjectEntry, PlacementGuide, ImageAnalysis,
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
        object_palette=[
            ObjectEntry(
                name="Vanderwolf's Pyramid Limber Pine",
                category="tree",
                default_quantity=3,
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


def test_outdoor_prompt_template_detects_landscape_context():
    from backend.core.staging_pipeline import build_adaptation_template

    template = build_adaptation_template(
        room_analysis="A backyard with wooden fence, turf, and patio",
        is_outdoor=True,
    )
    assert "landscape" in template.lower() or "outdoor" in template.lower()
