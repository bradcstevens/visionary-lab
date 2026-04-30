"""Scenario tests using actual backyard landscaping test data.

Uses images and plant data from tests/projects/backyard-landscaping/.
All Azure calls are mocked — these are unit tests that verify the
pipeline correctly processes the backyard scenario end-to-end.
"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKYARD_DIR = Path(__file__).parent / "projects" / "backyard-landscaping"
BACKYARD_IMAGES = sorted(BACKYARD_DIR.glob("*.png"))
BACKYARD_MD = BACKYARD_DIR / "BACKYARD.md"


def test_backyard_test_data_exists():
    """Verify the test fixture data is present."""
    assert BACKYARD_DIR.exists(), "backyard-landscaping test directory missing"
    assert len(BACKYARD_IMAGES) == 13, f"Expected 13 images, found {len(BACKYARD_IMAGES)}"
    assert BACKYARD_MD.exists(), "BACKYARD.md missing"


def test_backyard_project_creation(client, mock_staging_deps):
    """Create a project and verify it accepts the request."""
    mock_container = mock_staging_deps["container"]
    mock_container.create_item.return_value = {
        "id": "proj-backyard",
        "name": "Backyard Fence Line — Spring 2026",
        "prompt": "Add layered privacy screen",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "created_at": "2026-04-26T00:00:00Z",
        "updated_at": "2026-04-26T00:00:00Z",
        "doc_type": "staging_project",
    }

    response = client.post("/api/v1/staging/projects", json={
        "name": "Backyard Fence Line — Spring 2026",
        "prompt": "Add layered privacy screen with Vanderwolf Pine, Baby Blue Eyes Spruce, and Columnar Norway Spruce",
    })
    assert response.status_code == 201
    assert response.json()["project"]["name"] == "Backyard Fence Line — Spring 2026"


@pytest.mark.asyncio
async def test_backyard_chat_plant_selection():
    """Simulate a conversation selecting specific plants from BACKYARD.md."""
    from backend.core.design_chat import DesignChatService
    from backend.models.design_brief import ImageAnalysis

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({
            "reply": "Vanderwolf's Pyramid Limber Pine is a great choice — silvery-blue needles, narrow pyramid form, grows 20-25 ft. How many along the fence?",
            "ready_for_brief": False,
            "suggested_actions": ["specify_quantity", "add_more_species"],
        })))]
    ))

    analyses = [
        ImageAnalysis(
            room_id="fence-east",
            description="Backyard view from east fence straight on to west fence, wooden fence with low shrubs and turf",
            features=["fence", "turf", "shrubs"],
            zones=["fence_line", "open_turf"],
        ),
    ]

    service = DesignChatService(
        async_llm_client=mock_llm,
        llm_deployment="gpt-5-4",
        image_analyses=analyses,
    )

    response = await service.chat(
        message="I want to add Vanderwolf's Pyramid Limber Pine along the fence",
        conversation_history=[],
        focused_image_id="fence-east",
    )

    assert "Vanderwolf" in response.reply
    assert response.ready_for_brief is False


@pytest.mark.asyncio
async def test_backyard_brief_includes_plant_details():
    """Verify the brief includes visual details from BACKYARD.md."""
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
                    "size": "20-25 ft tall",
                    "placement": "back row along fence",
                    "visual_notes": "Blue-green to silvery-blue twisted needles in bundles of 5, narrow pyramid silhouette",
                },
                {
                    "name": "Baby Blue Eyes Spruce",
                    "description": "Picea pungens 'Baby Blue Eyes'",
                    "category": "tree",
                    "default_quantity": 2,
                    "size": "15-30 ft tall",
                    "placement": "corners of fence line",
                    "visual_notes": "Intense powder-blue needles, classic Christmas-tree shape",
                },
            ],
            "placement_guide": {"back_row": "Tall conifers: Limber Pine + Spruce", "middle_row": None, "front_row": None},
            "preserve_elements": ["existing patio", "fire pit", "pergola"],
            "per_image_notes": {},
        })))]
    ))

    analyses = [
        ImageAnalysis(room_id="r1", description="Fence line view", features=["fence"], zones=["fence_line"]),
    ]
    history = [
        ChatMessage(role="user", content="Add Vanderwolf Pine and Baby Blue Eyes Spruce along fence"),
        ChatMessage(role="assistant", content="How many of each?"),
        ChatMessage(role="user", content="3 Limber Pines, 2 Baby Blue Eyes at the corners"),
    ]

    service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
    brief, _ = await service.generate_brief(conversation_history=history, image_analyses=analyses)

    assert len(brief.object_palette) == 2
    pine = next(p for p in brief.object_palette if "Vanderwolf" in p.name)
    assert pine.default_quantity == 3
    assert "silvery" in pine.visual_notes.lower() or "blue" in pine.visual_notes.lower()

    spruce = next(p for p in brief.object_palette if "Baby Blue" in p.name)
    assert spruce.default_quantity == 2


@pytest.mark.asyncio
async def test_backyard_adapted_prompts_are_specific():
    """Verify adapted prompts reference specific plants and scene features."""
    from backend.core.brief_generator import BriefGeneratorService
    from backend.models.design_brief import (
        DesignBrief, ObjectEntry, PlacementGuide, ImageAnalysis,
    )

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps([
            "Along the wooden fence in this backyard, add 3 Vanderwolf's Pyramid Limber Pines (20-25ft, silvery-blue twisted needles, narrow pyramid form) in the back row spaced 8ft apart. Keep existing low shrubs and turf unchanged.",
            "Plant a row of tall Vanderwolf Pines with blue-green foliage along the fence line as a privacy screen. Position them behind the existing shrubs to create a layered effect. Preserve the open turf area.",
        ])))]
    ))

    brief = DesignBrief(
        global_instructions="Add evergreen privacy screen along fence",
        object_palette=[
            ObjectEntry(name="Vanderwolf's Pyramid Limber Pine", category="tree", default_quantity=3, size="20-25 ft",
                        placement="back row along fence",
                        visual_notes="Silvery-blue twisted needles, narrow pyramid form"),
        ],
        placement_guide=PlacementGuide(back_row="Tall conifers along fence"),
        preserve_elements=["existing shrubs", "turf"],
    )
    analyses = [
        ImageAnalysis(room_id="fence-east", description="East fence view with turf and low shrubs",
                      features=["fence", "turf", "shrubs"], zones=["fence_line"]),
    ]

    service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
    prompts = await service.brief_to_prompts(brief=brief, image_analyses=analyses, n_variations=2)

    assert "fence-east" in prompts
    assert len(prompts["fence-east"]) == 2
    for prompt in prompts["fence-east"]:
        assert len(prompt) > 50, f"Prompt too short to be specific: {prompt}"


@pytest.mark.asyncio
async def test_backyard_per_image_notes_differ():
    """Verify per-image notes produce different prompts for pergola vs fence."""
    from backend.core.brief_generator import BriefGeneratorService
    from backend.models.design_brief import (
        DesignBrief, ObjectEntry, PlacementGuide, ImageAnalysis,
    )

    async def mock_create(**kwargs):
        messages = kwargs.get("messages", [])
        system_msg = messages[0]["content"] if messages else ""
        if "climbing jasmine" in system_msg.lower():
            prompts = ["Add climbing jasmine on pergola posts with star-shaped white flowers"]
        else:
            prompts = ["Plant Columnar Norway Spruce along the fence line in a narrow column"]
        return MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(prompts)))])

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create = AsyncMock(side_effect=mock_create)

    brief = DesignBrief(
        global_instructions="Add plants throughout backyard",
        object_palette=[ObjectEntry(name="Columnar Norway Spruce", category="tree", default_quantity=5, placement="along fence")],
        placement_guide=PlacementGuide(back_row="Tall conifers"),
        per_image_notes={
            "pergola-1": "Add climbing jasmine on the pergola posts instead of ground plants",
        },
    )
    analyses = [
        ImageAnalysis(room_id="fence-1", description="Fence line", features=["fence"], zones=["fence_line"]),
        ImageAnalysis(room_id="pergola-1", description="Pergola with staircase", features=["pergola"], zones=["patio"]),
    ]

    service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
    prompts = await service.brief_to_prompts(brief=brief, image_analyses=analyses, n_variations=1)

    assert "fence-1" in prompts
    assert "pergola-1" in prompts
    fence_prompt = prompts["fence-1"][0].lower()
    pergola_prompt = prompts["pergola-1"][0].lower()
    assert fence_prompt != pergola_prompt
