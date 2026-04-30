"""Tests for BriefGeneratorService prompt parsing robustness."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.core.brief_generator import BriefGeneratorService
from backend.models.design_brief import (
    DesignBrief, ImageAnalysis, ObjectEntry, PlacementGuide,
)


def _make_brief():
    return DesignBrief(
        global_instructions="Add drought-tolerant plants along the fence line",
        object_palette=[
            ObjectEntry(name="Lavender", category="plant", default_quantity=3, size="2ft", placement="front row"),
        ],
        placement_guide=PlacementGuide(back_row="Tall grasses"),
        per_image_notes={},
        preserve_elements=["existing fence", "fire pit"],
    )


def _make_analysis(room_id="room-1"):
    return ImageAnalysis(
        room_id=room_id,
        description="A backyard with a wooden fence and gravel path",
        features=["fence", "gravel", "patio"],
        zones=["along fence", "front border"],
    )


def _mock_llm_response(content: str):
    """Build a mock LLM completion response."""
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))]
    )


class TestBriefToPromptsJsonParsing:
    """Verify that brief_to_prompts handles all JSON shapes the LLM might return."""

    @pytest.mark.asyncio
    async def test_parses_prompts_key(self):
        """LLM wraps array under "prompts" key."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({"prompts": ["Add lavender along fence", "Add grasses behind fence"]})
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        result = await service.brief_to_prompts(_make_brief(), [_make_analysis()], n_variations=2)
        assert "room-1" in result
        assert len(result["room-1"]) == 2
        assert "lavender" in result["room-1"][0].lower()

    @pytest.mark.asyncio
    async def test_parses_variations_key(self):
        """LLM wraps array under "variations" key — previously this caused all 3 retries to fail."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({"variations": ["Prompt A", "Prompt B"]})
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        result = await service.brief_to_prompts(_make_brief(), [_make_analysis()], n_variations=2)
        assert "room-1" in result
        assert result["room-1"] == ["Prompt A", "Prompt B"]

    @pytest.mark.asyncio
    async def test_parses_results_key(self):
        """LLM wraps array under "results" key."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({"results": ["R1", "R2", "R3"]})
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        result = await service.brief_to_prompts(_make_brief(), [_make_analysis()], n_variations=3)
        assert result["room-1"] == ["R1", "R2", "R3"]

    @pytest.mark.asyncio
    async def test_parses_arbitrary_key_with_list(self):
        """LLM uses an unexpected key name — fallback picks first list value."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({"image_prompts": ["X", "Y"]})
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        result = await service.brief_to_prompts(_make_brief(), [_make_analysis()], n_variations=2)
        assert result["room-1"] == ["X", "Y"]

    @pytest.mark.asyncio
    async def test_falls_back_to_global_instructions_on_failure(self):
        """If all 3 attempts return non-parseable JSON, fall back to global instructions."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({"status": "ok"})  # Dict with no list values
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        brief = _make_brief()
        result = await service.brief_to_prompts(brief, [_make_analysis()], n_variations=2)
        assert result["room-1"] == [brief.global_instructions] * 2

    @pytest.mark.asyncio
    async def test_respects_n_variations_limit(self):
        """Only keep up to n_variations prompts even if LLM returns more."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({"prompts": ["A", "B", "C", "D", "E"]})
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        result = await service.brief_to_prompts(_make_brief(), [_make_analysis()], n_variations=3)
        assert len(result["room-1"]) == 3

    @pytest.mark.asyncio
    async def test_parses_string_valued_dict(self):
        """LLM returns dict with numbered keys and string values (no list wrapping)."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({
                "1": "Add three Vanderwolf Pines along the eastern fence line spaced 8 feet apart",
                "2": "Plant a row of Baby Blue Eyes Spruce at the corners of the fence for privacy",
            })
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        result = await service.brief_to_prompts(_make_brief(), [_make_analysis()], n_variations=2)
        assert "room-1" in result
        assert len(result["room-1"]) == 2
        assert "Vanderwolf" in result["room-1"][0]

    @pytest.mark.asyncio
    async def test_rejects_short_string_dict_as_non_prompts(self):
        """Dict with short string values (like status messages) should NOT be treated as prompts."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({"status": "ok", "message": "done"})
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        brief = _make_brief()
        result = await service.brief_to_prompts(brief, [_make_analysis()], n_variations=2)
        # Should fall back to global instructions, not use "ok" and "done" as prompts
        assert result["room-1"] == [brief.global_instructions] * 2
