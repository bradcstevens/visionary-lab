"""Tests for DesignChatService and analyze endpoint."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

MOCK_ANALYSIS = {
    "description": "Backyard with wooden fence, open turf area, and low shrubs",
    "features": ["fence", "turf", "shrubs"],
    "zones": ["fence_line", "open_turf"],
}


@pytest.mark.asyncio
async def test_chat_returns_reply_and_suggested_actions():
    from backend.core.design_chat import DesignChatService
    from backend.models.design_brief import ImageAnalysis

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({
            "reply": "What species of trees are you considering?",
            "ready_for_brief": False,
            "suggested_actions": ["specify_species", "choose_density"],
        })))]
    ))

    analyses = [ImageAnalysis(room_id="r1", **MOCK_ANALYSIS)]
    service = DesignChatService(
        async_llm_client=mock_llm,
        llm_deployment="gpt-5-4",
        image_analyses=analyses,
    )

    response = await service.chat(
        message="I want trees along the fence",
        conversation_history=[],
        focused_image_id=None,
    )

    assert response.reply == "What species of trees are you considering?"
    assert response.ready_for_brief is False
    assert "specify_species" in response.suggested_actions

    call_args = mock_llm.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages", [])
    system_msg = messages[0]["content"]
    assert "fence" in system_msg
    assert "turf" in system_msg


@pytest.mark.asyncio
async def test_chat_with_focused_image_highlights_that_image():
    from backend.core.design_chat import DesignChatService
    from backend.models.design_brief import ImageAnalysis

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({
            "reply": "For the pergola area, I'd suggest...",
            "ready_for_brief": False,
            "suggested_actions": [],
        })))]
    ))

    analyses = [
        ImageAnalysis(room_id="r1", description="Fence line view", features=["fence"], zones=["fence_line"]),
        ImageAnalysis(room_id="r2", description="Pergola with staircase", features=["pergola", "staircase"], zones=["patio"]),
    ]

    service = DesignChatService(
        async_llm_client=mock_llm,
        llm_deployment="gpt-5-4",
        image_analyses=analyses,
    )

    response = await service.chat(
        message="What should I add here?",
        conversation_history=[],
        focused_image_id="r2",
    )

    call_args = mock_llm.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages", [])
    system_msg = messages[0]["content"]
    assert "FOCUSED IMAGE" in system_msg or "Pergola with staircase" in system_msg


@pytest.mark.asyncio
async def test_chat_signals_ready_for_brief():
    from backend.core.design_chat import DesignChatService
    from backend.models.design_brief import ImageAnalysis, ChatMessage

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({
            "reply": "I have enough details. Ready to generate your Design Brief?",
            "ready_for_brief": True,
            "suggested_actions": ["generate_brief"],
        })))]
    ))

    analyses = [ImageAnalysis(room_id="r1", **MOCK_ANALYSIS)]
    service = DesignChatService(
        async_llm_client=mock_llm,
        llm_deployment="gpt-5-4",
        image_analyses=analyses,
    )

    history = [
        ChatMessage(role="assistant", content="What would you like to add?"),
        ChatMessage(role="user", content="Vanderwolf Pine along the fence"),
        ChatMessage(role="assistant", content="How many?"),
        ChatMessage(role="user", content="3 in the back row, spaced 8ft apart"),
    ]

    response = await service.chat(
        message="That covers everything",
        conversation_history=history,
        focused_image_id=None,
    )

    assert response.ready_for_brief is True
