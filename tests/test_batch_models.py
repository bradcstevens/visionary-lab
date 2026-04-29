"""Tests for batch request/response models."""
from backend.models.images import (
    ImageBatchRequest,
    ImageBatchResponse,
    ImagePipelineRequest,
    ImagePipelineResponse,
    PipelineAction,
)


def test_batch_request_accepts_list_of_pipeline_requests():
    req = ImageBatchRequest(requests=[
        ImagePipelineRequest(action=PipelineAction.GENERATE, prompt="a cat"),
        ImagePipelineRequest(action=PipelineAction.GENERATE, prompt="a dog"),
    ])
    assert len(req.requests) == 2


def test_batch_response_tracks_counts():
    resp = ImageBatchResponse(
        results=[],
        total=2,
        succeeded=1,
        failed=1,
    )
    assert resp.total == 2
    assert resp.succeeded == 1
    assert resp.failed == 1


def test_batch_request_enforces_min_length():
    """Empty requests list should be rejected."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ImageBatchRequest(requests=[])


def test_batch_request_enforces_max_length():
    """Lists > 20 items should be rejected."""
    import pytest
    from pydantic import ValidationError
    requests = [ImagePipelineRequest(action=PipelineAction.GENERATE, prompt=f"prompt {i}") for i in range(21)]
    with pytest.raises(ValidationError):
        ImageBatchRequest(requests=requests)
