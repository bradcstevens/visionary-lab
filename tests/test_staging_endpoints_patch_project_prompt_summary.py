"""Tests for ``PATCH /api/v1/staging/projects/{id}`` prompt-summary
maintenance — issue 013 of image-pipeline-and-project-ux-overhaul PRD.

Endpoint contract (extends the contract in
``test_staging_endpoints_patch_project.py``):

- When the client sends ``prompt`` but NOT ``prompt_summary``: the
  server regenerates the summary via ``PromptSummarizer.summarize``.
- When the client sends ``prompt_summary`` explicitly: the client's
  value wins (after server-side ≤240 normalization).
- When the client sends neither: the persisted ``prompt_summary`` is
  left untouched.
- PATCH NEVER triggers image regeneration — the staging pipeline
  dependency must not be invoked.
- Empty / whitespace-only ``prompt_summary`` raises 422 (validator).

The tests stub the ``PromptSummarizer`` dependency via
``app.dependency_overrides`` so we don't need a real LLM. The mock
returns a predictable string so we can assert pass-through wiring.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.endpoints.staging import get_prompt_summarizer
from backend.core.prompt_summarizer import PromptSummarizer


def _project_doc() -> dict:
    return {
        "id": "proj-ps",
        "name": "Original",
        "prompt": "modern minimalist",
        "prompt_summary": "Original modern minimalist summary",
        "status": "completed",
        "rooms": [],
        "settings": {
            "variations_per_room": 5,
            "model": "gpt-image-2",
            "quality": "high",
            "size": "auto",
        },
    }


def _captured_replace_body(mock_container):
    call = mock_container.replace_item.call_args
    return call.kwargs.get("body") or call.args[1]


@pytest.fixture
def stub_summarizer(app):
    """Replace the PromptSummarizer dependency with one that returns a
    deterministic, traceable summary so tests can assert wiring."""
    summarizer = MagicMock(spec=PromptSummarizer)
    summarizer.summarize = AsyncMock(return_value="GENERATED-SUMMARY")
    app.dependency_overrides[get_prompt_summarizer] = lambda: summarizer
    yield summarizer
    app.dependency_overrides.pop(get_prompt_summarizer, None)


# ---- prompt change → server regenerates summary ------------------------


def test_patch_prompt_only_regenerates_prompt_summary(
    client, mock_staging_deps, stub_summarizer
):
    container = mock_staging_deps["container"]
    container.read_item.return_value = _project_doc()
    container.replace_item.side_effect = lambda item, body: body

    new_prompt = "warm wood and lots of greenery, large pergola"
    response = client.patch(
        "/api/v1/staging/projects/proj-ps",
        json={"prompt": new_prompt},
    )
    assert response.status_code == 200, response.text

    stub_summarizer.summarize.assert_awaited_once_with(new_prompt)
    persisted = _captured_replace_body(container)
    assert persisted["prompt"] == new_prompt
    assert persisted["prompt_summary"] == "GENERATED-SUMMARY"
    assert response.json()["project"]["prompt_summary"] == "GENERATED-SUMMARY"


# ---- explicit prompt_summary wins --------------------------------------


def test_patch_explicit_prompt_summary_wins_over_regeneration(
    client, mock_staging_deps, stub_summarizer
):
    """When the client supplies prompt_summary alongside prompt, the
    client's value is persisted as-is and the summarizer is NOT called
    (avoids a wasted RU and respects user intent)."""
    container = mock_staging_deps["container"]
    container.read_item.return_value = _project_doc()
    container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-ps",
        json={
            "prompt": "new prompt",
            "prompt_summary": "Hand-tuned summary written by the user.",
        },
    )
    assert response.status_code == 200, response.text

    stub_summarizer.summarize.assert_not_called()
    persisted = _captured_replace_body(container)
    assert persisted["prompt"] == "new prompt"
    assert persisted["prompt_summary"] == "Hand-tuned summary written by the user."


# ---- summary untouched when not in PATCH body --------------------------


def test_patch_other_field_leaves_prompt_summary_untouched(
    client, mock_staging_deps, stub_summarizer
):
    """A name-only PATCH must NOT regenerate or clear prompt_summary."""
    container = mock_staging_deps["container"]
    container.read_item.return_value = _project_doc()
    container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-ps",
        json={"name": "Renamed"},
    )
    assert response.status_code == 200, response.text

    stub_summarizer.summarize.assert_not_called()
    persisted = _captured_replace_body(container)
    assert persisted["prompt_summary"] == "Original modern minimalist summary"
    assert persisted["name"] == "Renamed"


# ---- never triggers regeneration ---------------------------------------


def test_patch_prompt_does_not_invoke_pipeline(
    client, mock_staging_deps, stub_summarizer
):
    """Per the PRD: editing a prompt MUST NOT trigger image regeneration.
    The mock pipeline (used by the legacy SSE regenerate path) must be
    untouched by any PATCH /projects/{id} call."""
    container = mock_staging_deps["container"]
    container.read_item.return_value = _project_doc()
    container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-ps",
        json={"prompt": "new prompt"},
    )
    assert response.status_code == 200, response.text

    pipeline = mock_staging_deps["pipeline"]
    # No method on the pipeline mock should have been called. The mock
    # records every attribute access, but only callables that fire show
    # up in mock_calls.
    assert pipeline.mock_calls == []


# ---- length normalization ---------------------------------------------


def test_patch_explicit_prompt_summary_too_long_is_truncated(
    client, mock_staging_deps, stub_summarizer
):
    """A client overshoot (>240 chars) is normalized via the same
    truncate_to_summary helper rather than 422-ing the request — the
    user's intent ("set this summary") is honored at the contract's
    capacity."""
    container = mock_staging_deps["container"]
    container.read_item.return_value = _project_doc()
    container.replace_item.side_effect = lambda item, body: body

    overshoot = "the quick brown fox " * 30  # ~600 chars
    response = client.patch(
        "/api/v1/staging/projects/proj-ps",
        json={"prompt_summary": overshoot},
    )
    assert response.status_code == 200, response.text

    persisted = _captured_replace_body(container)
    assert len(persisted["prompt_summary"]) <= 240
    assert persisted["prompt_summary"].endswith("\u2026")


# ---- 422 on empty / null prompt_summary --------------------------------


def test_patch_empty_prompt_summary_returns_422(
    client, mock_staging_deps, stub_summarizer
):
    container = mock_staging_deps["container"]
    container.read_item.return_value = _project_doc()

    response = client.patch(
        "/api/v1/staging/projects/proj-ps",
        json={"prompt_summary": "   "},
    )
    assert response.status_code == 422


def test_patch_null_prompt_summary_returns_422(
    client, mock_staging_deps, stub_summarizer
):
    container = mock_staging_deps["container"]
    container.read_item.return_value = _project_doc()

    response = client.patch(
        "/api/v1/staging/projects/proj-ps",
        json={"prompt_summary": None},
    )
    assert response.status_code == 422


# ---- create_project also seeds prompt_summary --------------------------


def test_create_project_seeds_prompt_summary(
    client, mock_staging_deps, stub_summarizer
):
    """POST /projects must populate prompt_summary at creation time so
    the project page renders the collapsed view from the first read."""
    container = mock_staging_deps["container"]
    container.create_item.side_effect = lambda body: body

    response = client.post(
        "/api/v1/staging/projects",
        json={"name": "New Project", "prompt": "a brand new long prompt"},
    )
    assert response.status_code == 201, response.text

    stub_summarizer.summarize.assert_awaited_once_with("a brand new long prompt")
    assert response.json()["project"]["prompt_summary"] == "GENERATED-SUMMARY"
