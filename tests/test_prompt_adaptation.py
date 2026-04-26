"""Tests for staging pipeline prompt adaptation logic."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_adapt_prompt_for_room_includes_user_prompt():
    from backend.core.staging_pipeline import StagingPipeline

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps([
            "Add a wooden bookshelf with plants",
            "Place a woven rug and floor lamp",
            "Add floating shelves with ceramics",
        ])))]
    ))

    pipeline = StagingPipeline.__new__(StagingPipeline)
    pipeline.async_llm_client = mock_llm
    pipeline.llm_deployment = "gpt-5-4"

    prompts = await pipeline.adapt_prompt(
        user_prompt="Modern minimalist with warm tones",
        room_analysis="A living room with a grey couch, bare white walls, hardwood floor",
        n_variations=3,
    )

    assert len(prompts) == 3
    assert all(isinstance(p, str) for p in prompts)
    call_args = mock_llm.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages", [])
    system_msg = messages[0]["content"]
    assert "Modern minimalist with warm tones" in system_msg
    assert "grey couch" in system_msg


@pytest.mark.asyncio
async def test_adapt_prompt_handles_llm_non_json():
    from backend.core.staging_pipeline import StagingPipeline

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create = AsyncMock(side_effect=[
        MagicMock(choices=[MagicMock(message=MagicMock(content="not json"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(["prompt 1", "prompt 2"])))]),
    ])

    pipeline = StagingPipeline.__new__(StagingPipeline)
    pipeline.async_llm_client = mock_llm
    pipeline.llm_deployment = "gpt-5-4"

    prompts = await pipeline.adapt_prompt(
        user_prompt="Rustic farmhouse",
        room_analysis="A kitchen with white cabinets",
        n_variations=2,
    )
    assert len(prompts) == 2


@pytest.mark.asyncio
async def test_analyze_room_returns_description():
    from backend.core.staging_pipeline import StagingPipeline

    mock_analyzer = AsyncMock()
    mock_analyzer.async_image_chat = AsyncMock(return_value={
        "description": "A bright living room with hardwood floors and large windows",
        "features": ["couch", "windows", "hardwood floor"],
    })

    pipeline = StagingPipeline.__new__(StagingPipeline)
    pipeline.image_analyzer = mock_analyzer

    result = await pipeline.analyze_room(image_base64="fake_base64_data")
    assert "description" in result
    assert "living room" in result["description"]