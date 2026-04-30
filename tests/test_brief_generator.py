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


class TestGenerateBriefReturnsTuple:
    """Issue 004 of the per-image-object-quantities PRD: ``generate_brief``
    now ALWAYS returns ``(brief, ReconcileSummary)``. Existing callers must
    unpack the tuple. ``ReconcileSummary`` is zero-zero when no
    ``previous_brief`` is supplied.
    """

    @pytest.mark.asyncio
    async def test_returns_tuple_with_zero_summary_when_no_previous_brief(self):
        from backend.models.design_brief import ReconcileSummary

        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({
                "global_instructions": "x",
                "object_palette": [
                    {
                        "name": "Pine",
                        "category": "tree",
                        "default_quantity": 2,
                        "size": "8 ft",
                        "placement": "back row",
                        "visual_notes": None,
                        "description": None,
                    }
                ],
                "placement_guide": {"back_row": "z"},
                "per_image_notes": {},
                "preserve_elements": [],
            })
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        result = await service.generate_brief(
            conversation_history=[], image_analyses=[_make_analysis()],
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        brief, summary = result
        assert isinstance(brief, DesignBrief)
        assert isinstance(summary, ReconcileSummary)
        assert summary.carried_forward == 0
        assert summary.dropped == 0


class TestPerImageObjectsParsing:
    """Issue 004: LLM may emit ``per_image_objects`` keyed by ``object_name``
    (NOT ``object_id``) when the conversation differentiates quantities or
    placement between specific images. The generator must:

    * Build a normalized name → assigned-UUID map from the just-built
      palette.
    * Walk LLM-emitted ``per_image_objects`` and substitute each
      ``object_name`` with the corresponding UUID.
    * Drop entries whose ``object_name`` doesn't match any palette entry.
    * Drop entries whose ``room_id`` isn't present in ``image_analyses``.
    * Skip palette entries whose normalized name is duplicated (ambiguous).
    * Drop malformed entries individually (a single bad row mustn't fail
      the whole brief).
    """

    @pytest.mark.asyncio
    async def test_object_name_substituted_to_object_id(self):
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({
                "global_instructions": "x",
                "object_palette": [
                    {
                        "name": "Lavender",
                        "category": "plant",
                        "default_quantity": 3,
                        "size": "2 ft",
                        "placement": "front",
                        "visual_notes": None,
                        "description": None,
                    }
                ],
                "placement_guide": {"back_row": "z"},
                "per_image_notes": {},
                "preserve_elements": [],
                "per_image_objects": {
                    "room-1": [
                        {
                            "object_name": "Lavender",
                            "quantity": 8,
                            "placement": "back row",
                            "enabled": True,
                        }
                    ]
                },
            })
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        brief, _ = await service.generate_brief(
            conversation_history=[], image_analyses=[_make_analysis("room-1")],
        )
        # The palette entry's UUID must now appear as object_id in the
        # carried-through override.
        palette_uuid = brief.object_palette[0].id
        overrides = brief.per_image_objects["room-1"]
        assert len(overrides) == 1
        assert overrides[0].object_id == palette_uuid
        assert overrides[0].quantity == 8
        assert overrides[0].placement == "back row"
        assert overrides[0].enabled is True

    @pytest.mark.asyncio
    async def test_unknown_object_name_dropped(self):
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({
                "global_instructions": "x",
                "object_palette": [
                    {
                        "name": "Lavender",
                        "category": "plant",
                        "default_quantity": 3,
                        "size": "2 ft",
                        "placement": "front",
                        "visual_notes": None,
                        "description": None,
                    }
                ],
                "placement_guide": {"back_row": "z"},
                "per_image_notes": {},
                "preserve_elements": [],
                "per_image_objects": {
                    "room-1": [
                        {"object_name": "Pine", "quantity": 5},  # Pine not in palette
                        {"object_name": "Lavender", "quantity": 8},
                    ]
                },
            })
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        brief, _ = await service.generate_brief(
            conversation_history=[], image_analyses=[_make_analysis("room-1")],
        )
        overrides = brief.per_image_objects["room-1"]
        # Pine entry dropped, Lavender survived.
        assert len(overrides) == 1
        assert overrides[0].object_id == brief.object_palette[0].id

    @pytest.mark.asyncio
    async def test_unknown_room_id_dropped(self):
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({
                "global_instructions": "x",
                "object_palette": [
                    {
                        "name": "Lavender",
                        "category": "plant",
                        "default_quantity": 3,
                        "size": "2 ft",
                        "placement": "front",
                        "visual_notes": None,
                        "description": None,
                    }
                ],
                "placement_guide": {"back_row": "z"},
                "per_image_notes": {},
                "preserve_elements": [],
                "per_image_objects": {
                    "room-1": [{"object_name": "Lavender", "quantity": 8}],
                    "room-9999-bogus": [{"object_name": "Lavender", "quantity": 5}],
                },
            })
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        brief, _ = await service.generate_brief(
            conversation_history=[], image_analyses=[_make_analysis("room-1")],
        )
        # Bogus room dropped wholesale.
        assert "room-9999-bogus" not in brief.per_image_objects
        assert len(brief.per_image_objects["room-1"]) == 1

    @pytest.mark.asyncio
    async def test_case_insensitive_name_substitution(self):
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({
                "global_instructions": "x",
                "object_palette": [
                    {
                        "name": "Lavender",
                        "category": "plant",
                        "default_quantity": 3,
                        "size": "2 ft",
                        "placement": "front",
                        "visual_notes": None,
                        "description": None,
                    }
                ],
                "placement_guide": {"back_row": "z"},
                "per_image_notes": {},
                "preserve_elements": [],
                "per_image_objects": {
                    "room-1": [
                        # "lavender" vs "Lavender" — must still match.
                        {"object_name": "  lavender  ", "quantity": 8}
                    ]
                },
            })
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        brief, _ = await service.generate_brief(
            conversation_history=[], image_analyses=[_make_analysis("room-1")],
        )
        assert len(brief.per_image_objects["room-1"]) == 1
        assert brief.per_image_objects["room-1"][0].object_id == brief.object_palette[0].id

    @pytest.mark.asyncio
    async def test_malformed_row_dropped_individually(self):
        """A single malformed override must be dropped without breaking
        the rest of the brief. Caught by rubber-duck review of the
        issue-004 plan: previously a missing ``quantity`` would have raised
        ValidationError and caused all 3 retry attempts to fail.
        """
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({
                "global_instructions": "x",
                "object_palette": [
                    {
                        "name": "Lavender",
                        "category": "plant",
                        "default_quantity": 3,
                        "size": "2 ft",
                        "placement": "front",
                        "visual_notes": None,
                        "description": None,
                    }
                ],
                "placement_guide": {"back_row": "z"},
                "per_image_notes": {},
                "preserve_elements": [],
                "per_image_objects": {
                    "room-1": [
                        "not-a-dict",  # malformed scalar
                        {"object_name": "Lavender"},  # missing quantity
                        {"quantity": 5},  # missing object_name
                        {"object_name": "Lavender", "quantity": "many"},  # non-int quantity
                        {"object_name": "Lavender", "quantity": 8},  # valid
                    ]
                },
            })
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        brief, _ = await service.generate_brief(
            conversation_history=[], image_analyses=[_make_analysis("room-1")],
        )
        # Only the one valid row survives.
        overrides = brief.per_image_objects["room-1"]
        assert len(overrides) == 1
        assert overrides[0].quantity == 8

    @pytest.mark.asyncio
    async def test_duplicate_normalized_palette_name_drops_referencing_overrides(self):
        """If the palette has two entries that normalize to the same name,
        we can't pick one unambiguously, so we drop any LLM-emitted
        override that targets that name.
        """
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({
                "global_instructions": "x",
                "object_palette": [
                    {
                        "name": "Pine",
                        "category": "tree",
                        "default_quantity": 2,
                        "size": "8 ft",
                        "placement": "back",
                        "visual_notes": None,
                        "description": None,
                    },
                    {
                        "name": "pine",
                        "category": "tree",
                        "default_quantity": 4,
                        "size": "6 ft",
                        "placement": "side",
                        "visual_notes": None,
                        "description": None,
                    },
                ],
                "placement_guide": {"back_row": "z"},
                "per_image_notes": {},
                "preserve_elements": [],
                "per_image_objects": {
                    "room-1": [{"object_name": "Pine", "quantity": 8}]
                },
            })
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        brief, _ = await service.generate_brief(
            conversation_history=[], image_analyses=[_make_analysis("room-1")],
        )
        # Empty (or missing) — ambiguous override dropped.
        assert brief.per_image_objects.get("room-1", []) == []


class TestGenerateBriefWithPreviousBrief:
    """Issue 004: when ``previous_brief`` is supplied, ``generate_brief``
    runs ``reconcile_overrides_by_name`` after assembling the new brief and
    surfaces non-zero ``carried_forward`` / ``dropped`` counts.
    """

    @pytest.mark.asyncio
    async def test_previous_brief_overrides_carried_forward(self):
        from backend.models.design_brief import (
            DesignBrief, ImageObjectOverride, ObjectEntry, PlacementGuide,
        )

        # Prev brief: user manually set Lavender qty=8 in room-1.
        prev_lavender = ObjectEntry(name="Lavender", default_quantity=3)
        prev_brief = DesignBrief(
            global_instructions="x",
            object_palette=[prev_lavender],
            placement_guide=PlacementGuide(back_row="z"),
            per_image_objects={
                "room-1": [ImageObjectOverride(object_id=prev_lavender.id, quantity=8)]
            },
        )

        # New LLM emits a new Lavender (different UUID, same name) and no
        # per_image_objects of its own.
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({
                "global_instructions": "x",
                "object_palette": [
                    {
                        "name": "Lavender",
                        "category": "plant",
                        "default_quantity": 3,
                        "size": "2 ft",
                        "placement": "front",
                        "visual_notes": None,
                        "description": None,
                    }
                ],
                "placement_guide": {"back_row": "z"},
                "per_image_notes": {},
                "preserve_elements": [],
            })
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        new_brief, summary = await service.generate_brief(
            conversation_history=[],
            image_analyses=[_make_analysis("room-1")],
            previous_brief=prev_brief,
        )
        assert summary.carried_forward == 1
        assert summary.dropped == 0
        new_lavender_id = new_brief.object_palette[0].id
        # The override now points at the NEW palette UUID.
        assert new_brief.per_image_objects["room-1"][0].object_id == new_lavender_id
        assert new_brief.per_image_objects["room-1"][0].quantity == 8

    @pytest.mark.asyncio
    async def test_previous_brief_renamed_object_dropped_with_summary(self):
        """User had a Pine override; new brief no longer has Pine — the
        override is dropped and ``summary.dropped == 1``.
        """
        from backend.models.design_brief import (
            DesignBrief, ImageObjectOverride, ObjectEntry, PlacementGuide,
        )

        prev_pine = ObjectEntry(name="Pine", default_quantity=2)
        prev_brief = DesignBrief(
            global_instructions="x",
            object_palette=[prev_pine],
            placement_guide=PlacementGuide(back_row="z"),
            per_image_objects={
                "room-1": [ImageObjectOverride(object_id=prev_pine.id, quantity=8)]
            },
        )

        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({
                "global_instructions": "x",
                "object_palette": [
                    {
                        "name": "Oak",
                        "category": "tree",
                        "default_quantity": 1,
                        "size": "10 ft",
                        "placement": "back",
                        "visual_notes": None,
                        "description": None,
                    }
                ],
                "placement_guide": {"back_row": "z"},
                "per_image_notes": {},
                "preserve_elements": [],
            })
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        new_brief, summary = await service.generate_brief(
            conversation_history=[],
            image_analyses=[_make_analysis("room-1")],
            previous_brief=prev_brief,
        )
        assert summary.carried_forward == 0
        assert summary.dropped == 1
        # No carried-forward overrides for room-1.
        assert new_brief.per_image_objects.get("room-1", []) == []

    @pytest.mark.asyncio
    async def test_previous_brief_reconcile_filters_invalid_room_ids_after_lift(self):
        """If a prev override exists for a room_id that's no longer in
        ``image_analyses`` (e.g., user removed an image before regenerate),
        the lift counts it as ``dropped`` so it doesn't pollute the new
        brief. Per rubber-duck review's non-blocking suggestion.
        """
        from backend.models.design_brief import (
            DesignBrief, ImageObjectOverride, ObjectEntry, PlacementGuide,
        )

        prev_lav = ObjectEntry(name="Lavender", default_quantity=3)
        prev_brief = DesignBrief(
            global_instructions="x",
            object_palette=[prev_lav],
            placement_guide=PlacementGuide(back_row="z"),
            per_image_objects={
                "room-1": [ImageObjectOverride(object_id=prev_lav.id, quantity=8)],
                "room-removed": [
                    ImageObjectOverride(object_id=prev_lav.id, quantity=99)
                ],
            },
        )

        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({
                "global_instructions": "x",
                "object_palette": [
                    {
                        "name": "Lavender",
                        "category": "plant",
                        "default_quantity": 3,
                        "size": "2 ft",
                        "placement": "front",
                        "visual_notes": None,
                        "description": None,
                    }
                ],
                "placement_guide": {"back_row": "z"},
                "per_image_notes": {},
                "preserve_elements": [],
            })
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        new_brief, _ = await service.generate_brief(
            conversation_history=[],
            image_analyses=[_make_analysis("room-1")],  # room-removed NOT here.
            previous_brief=prev_brief,
        )
        # room-1 carried; room-removed filtered out at the final step.
        assert "room-removed" not in new_brief.per_image_objects
        assert len(new_brief.per_image_objects["room-1"]) == 1


class TestBriefToPromptsWithRejectedPrompt:
    """Issue 003 of single-variation-regeneration PRD: ``brief_to_prompts``
    accepts an optional ``rejected_prompt`` and threads it through to the
    LLM call site so "Try Something New" diverges from the rejected
    aesthetic."""

    @pytest.mark.asyncio
    async def test_default_rejected_prompt_is_none_no_steering(self):
        """Existing first-time generation paths must keep working with no
        new argument: the default value is ``None`` and no steering is
        injected into the LLM call."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({"prompts": ["A", "B"]})
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        await service.brief_to_prompts(_make_brief(), [_make_analysis()], n_variations=2)
        sent = mock_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        # No "REJECTED" / "REGENERATION STEERING" markers when no prior prompt.
        assert "REJECTED_PRIOR_DIRECTION" not in sent
        assert "REGENERATION STEERING" not in sent

    @pytest.mark.asyncio
    async def test_rejected_prompt_appears_in_llm_call_site(self):
        """When a prior prompt is supplied, the actual ``messages[0].content``
        sent to the LLM must contain the rejected prompt as negative context.
        This is the integration acceptance criterion (rubber-duck blocking
        finding #2): mocking ``brief_to_prompts``' boundary alone wouldn't
        prove the LLM call site receives the steering. We assert at the
        LLM mock directly."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({"prompts": ["new direction A", "new direction B"]})
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        await service.brief_to_prompts(
            _make_brief(),
            [_make_analysis()],
            n_variations=2,
            rejected_prompt="MAGENTA-AND-CHROME MAXIMALIST AESTHETIC",
        )
        sent = mock_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "MAGENTA-AND-CHROME MAXIMALIST AESTHETIC" in sent
        assert "REJECTED_PRIOR_DIRECTION" in sent

    @pytest.mark.asyncio
    async def test_rejected_prompt_does_not_drop_brief_intent(self):
        """The user's existing brief intent (e.g. ``global_instructions``)
        must SURVIVE the steering wrapper — diversification biases the
        LLM away from the rejected aesthetic but keeps the user's intent
        intact."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({"prompts": ["x", "y"]})
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        brief = _make_brief()  # global_instructions = "Add drought-tolerant plants along the fence line"
        await service.brief_to_prompts(
            brief,
            [_make_analysis()],
            n_variations=2,
            rejected_prompt="industrial concrete jungle",
        )
        sent = mock_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert brief.global_instructions in sent
        # The original brief content survives alongside the steering.
        assert "Lavender" in sent  # palette object name
        assert "existing fence" in sent  # preserve element

    @pytest.mark.asyncio
    async def test_empty_rejected_prompt_treated_as_none(self):
        """Empty strings are normalized to "no rejected prompt" — caller
        can pass through ``variation.generation_metadata.adapted_prompt``
        even when it's an empty string without a guard."""
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            json.dumps({"prompts": ["A", "B"]})
        )
        service = BriefGeneratorService(async_llm_client=mock_llm, llm_deployment="gpt-5-4")
        await service.brief_to_prompts(
            _make_brief(),
            [_make_analysis()],
            n_variations=2,
            rejected_prompt="",
        )
        sent = mock_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "REJECTED_PRIOR_DIRECTION" not in sent
