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


class TestBriefToPromptsPerImageObjectSummary:
    """Issue 003 of the per-image-object-quantities PRD: brief_to_prompts
    must construct a separate object_summary per image using the resolver,
    so two rooms with different overrides produce different prompts.
    """

    @pytest.mark.asyncio
    async def test_two_rooms_different_overrides_produce_different_object_summaries(self):
        """Quantity override on room A vs skip on room B for the same object
        produces materially different ``object_summary`` substrings in the
        captured system prompts.
        """
        from backend.models.design_brief import ImageObjectOverride

        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({"prompts": ["p1", "p2"]})
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")

        lavender = ObjectEntry(name="Lavender", category="plant", default_quantity=3, size="2ft", placement="front row")
        brief = DesignBrief(
            global_instructions="x",
            object_palette=[lavender],
            placement_guide=PlacementGuide(back_row="grasses"),
            per_image_notes={},
            preserve_elements=[],
            per_image_objects={
                "room-A": [ImageObjectOverride(object_id=lavender.id, quantity=10)],
                "room-B": [ImageObjectOverride(object_id=lavender.id, quantity=0)],
            },
        )
        analyses = [
            ImageAnalysis(room_id="room-A", description="backyard A"),
            ImageAnalysis(room_id="room-B", description="backyard B"),
        ]

        result = await service.brief_to_prompts(brief, analyses, n_variations=2)

        # Two LLM calls — one per room — and the system_content arg
        # captures the per-image object_summary substring.
        assert mock_llm.chat.completions.create.call_count == 2
        call_args_list = mock_llm.chat.completions.create.call_args_list
        # Each call: kwargs['messages'][0]['content'] holds system_content.
        system_contents = [
            call.kwargs["messages"][0]["content"] for call in call_args_list
        ]
        # Room A: quantity override 10 → "10x Lavender" appears.
        # Room B: quantity 0 → Lavender skipped → "Lavender" absent.
        room_a_content = next(c for c in system_contents if "backyard A" in c)
        room_b_content = next(c for c in system_contents if "backyard B" in c)
        assert "10x Lavender" in room_a_content
        assert "Lavender" not in room_b_content
        assert result["room-A"] == ["p1", "p2"]
        assert result["room-B"] == ["p1", "p2"]

    @pytest.mark.asyncio
    async def test_palette_only_brief_yields_identical_object_summaries_per_room(self):
        """No overrides → both rooms see palette defaults → identical object_summary."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({"prompts": ["p1", "p2"]})
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        analyses = [
            ImageAnalysis(room_id="room-A", description="A"),
            ImageAnalysis(room_id="room-B", description="B"),
        ]

        await service.brief_to_prompts(_make_brief(), analyses, n_variations=2)

        contents = [
            call.kwargs["messages"][0]["content"]
            for call in mock_llm.chat.completions.create.call_args_list
        ]
        # Both contain "3x Lavender" — palette default flows through both.
        assert all("3x Lavender" in c for c in contents)
