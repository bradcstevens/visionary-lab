# AI Design Questionnaire & Bug Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 3 critical frontend–backend API mismatches and add a full-featured AI Design Session with split-panel chat and structured Design Brief editor to the Projects wizard.

**Architecture:** Bug fixes align the frontend API calls to existing backend contracts. New features add 3 backend endpoints (analyze, chat, brief) powered by 2 new services (DesignChatService, BriefGeneratorService) and 6 new frontend components (ImageGalleryPanel, DesignChat, QuickReplyChips, DesignBriefEditor, PlantPaletteTable, GenerationSummary) composed into a redesigned 5-step wizard.

**Tech Stack:** Python/FastAPI (backend), Pydantic v1 models, Azure OpenAI async client, Next.js 15/React 19 (frontend), Radix UI primitives, Playwright E2E, pytest unit tests.

**Spec:** `docs/superpowers/specs/2026-04-26-ai-questionnaire-design.md`

---

## File Structure

### Backend — New Files
| File | Responsibility |
|------|---------------|
| `backend/models/design_brief.py` | Pydantic models: DesignBrief, PlantEntry, PlacementGuide, ImageAnalysis, ChatMessage, ChatRequest, ChatResponse |
| `backend/core/design_chat.py` | DesignChatService — conversational AI with image context |
| `backend/core/brief_generator.py` | BriefGeneratorService — synthesizes conversation into structured brief, converts brief to per-image prompts |

### Backend — Modified Files
| File | Changes |
|------|---------|
| `backend/api/endpoints/staging.py` | Add 3 endpoints: analyze, chat, brief (POST + PUT) |
| `backend/core/staging_pipeline.py` | Update PROMPT_ADAPTATION_TEMPLATE for outdoor context; update adapt_prompt() to accept optional DesignBrief |
| `backend/models/staging.py` | Add optional `design_brief` field to StagingProject; add `analyses` field |

### Frontend — New Files
| File | Responsibility |
|------|---------------|
| `frontend/components/staging/ImageGalleryPanel.tsx` | Left panel: grouped thumbnails, click-to-focus |
| `frontend/components/staging/DesignChat.tsx` | Right panel: chat messages, streaming, quick replies |
| `frontend/components/staging/QuickReplyChips.tsx` | AI-suggested clickable pills |
| `frontend/components/staging/DesignBriefEditor.tsx` | Step 4: full structured form editor |
| `frontend/components/staging/PlantPaletteTable.tsx` | Editable plant species table |
| `frontend/components/staging/GenerationSummary.tsx` | Step 5: review card + launch |

### Frontend — Modified Files
| File | Changes |
|------|---------|
| `frontend/services/stagingApi.ts` | Fix 3 bugs; add analyzeImages(), chat(), generateBrief(), updateBrief() |
| `frontend/components/staging/NewProjectWizard.tsx` | Redesign to 5-step flow incorporating new components |
| `frontend/components/staging/RoomGroup.tsx` | Remove unused getStatusColor |

### Test Files — New
| File | Tests |
|------|-------|
| `tests/test_bug_fixes.py` | 4 bug fix regression tests |
| `tests/test_design_brief.py` | 4 brief model + adaptation tests |
| `tests/test_design_chat.py` | 3 chat + analyze endpoint tests |
| `tests/test_backyard_scenario.py` | 6 backyard landscaping scenario tests |
| `frontend/tests/e2e/ai-design-session.spec.ts` | 6 Playwright E2E tests |

---

## Phase 1: Bug Fixes

### Task 1: Fix Upload Rooms API Field Names

**Files:**
- Modify: `frontend/services/stagingApi.ts:214-218`
- Test: `tests/test_bug_fixes.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_bug_fixes.py`:

```python
"""Regression tests for frontend–backend API bug fixes."""
import io
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import UploadFile


def test_upload_rooms_accepts_images_field(client, mock_staging_deps):
    """Backend upload_rooms endpoint accepts 'images' as the file field name."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-123",
        "name": "Test",
        "prompt": "Test",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 2, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }
    mock_container.upsert_item.return_value = None

    with patch("backend.api.endpoints.staging.AzureBlobStorageService") as mock_blob_cls:
        mock_blob = AsyncMock()
        mock_blob_cls.return_value = mock_blob
        mock_blob.upload_asset.return_value = {"url": "https://test.blob.core.windows.net/img.png"}

        # The backend expects the field name 'images', not 'room_files'
        response = client.post(
            "/api/v1/staging/projects/proj-123/rooms",
            files=[("images", ("test.png", io.BytesIO(b"fake-png"), "image/png"))],
            data={"labels": '["Backyard East"]'},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["rooms_added"] == 1
    assert data["rooms"][0]["label"] == "Backyard East"
```

- [ ] **Step 2: Run test to verify it passes (this tests existing backend — should pass)**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/test_bug_fixes.py::test_upload_rooms_accepts_images_field -v`
Expected: PASS (backend already expects `images`)

- [ ] **Step 3: Fix frontend uploadRooms to use correct field names**

In `frontend/services/stagingApi.ts`, replace:
```typescript
  const formData = new FormData();
  roomFiles.forEach(({ file, name }, index) => {
    formData.append('room_files', file, file.name);
    formData.append(`room_names`, name);
  });
```

With:
```typescript
  const formData = new FormData();
  const labels: string[] = [];
  roomFiles.forEach(({ file, name }) => {
    formData.append('images', file, file.name);
    labels.push(name);
  });
  formData.append('labels', JSON.stringify(labels));
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_bug_fixes.py frontend/services/stagingApi.ts
git commit -m "fix: align uploadRooms field names to backend contract

Frontend was sending 'room_files'/'room_names' but backend expects
'images'/'labels'. Upload rooms now works correctly.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Fix Create Project Schema Mismatch

**Files:**
- Modify: `frontend/services/stagingApi.ts:31-38,115-138`
- Test: `tests/test_bug_fixes.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bug_fixes.py`:

```python
def test_create_project_with_nested_settings(client, mock_staging_deps):
    """Backend expects settings as a nested object, not flat fields."""
    mock_container = mock_staging_deps["container"]
    mock_container.create_item.return_value = {
        "id": "proj-456",
        "name": "Backyard Fence Line",
        "prompt": "Add layered privacy screen",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 3, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "created_at": "2026-04-26T00:00:00Z",
        "updated_at": "2026-04-26T00:00:00Z",
        "doc_type": "staging_project",
    }

    response = client.post("/api/v1/staging/projects", json={
        "name": "Backyard Fence Line",
        "prompt": "Add layered privacy screen",
        "settings": {
            "variations_per_room": 3,
            "model": "gpt-image-2",
            "quality": "high",
            "size": "auto",
        },
    })

    assert response.status_code == 201
    data = response.json()
    assert data["project"]["settings"]["variations_per_room"] == 3
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_bug_fixes.py::test_create_project_with_nested_settings -v`
Expected: PASS (backend already expects nested settings)

- [ ] **Step 3: Fix frontend CreateProjectRequest and createProject()**

In `frontend/services/stagingApi.ts`, replace the `CreateProjectRequest` interface:

```typescript
export interface CreateProjectRequest {
  name: string;
  prompt: string;
  settings?: {
    variations_per_room?: number;
    model?: string;
    quality?: string;
    size?: string;
  };
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/services/stagingApi.ts tests/test_bug_fixes.py
git commit -m "fix: align CreateProjectRequest to backend nested settings schema

Frontend was sending flat fields (style, variations_per_room) but
backend expects nested settings object. Removed unused 'style' field.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Fix Streaming URL Mismatch

**Files:**
- Modify: `frontend/services/stagingApi.ts:240,359`
- Test: `tests/test_bug_fixes.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bug_fixes.py`:

```python
def test_generate_endpoint_exists_without_stream_suffix(client, mock_staging_deps):
    """The generate endpoint is /generate, not /generate/stream."""
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-789",
        "name": "Test",
        "prompt": "Test",
        "status": "uploading",
        "rooms": [
            {
                "id": "room-1",
                "label": "Room 1",
                "original_image_url": "https://test.blob.core.windows.net/img.png",
                "status": "pending",
                "variations": [{"id": "v-1", "status": "pending"}],
            }
        ],
        "settings": {"variations_per_room": 1, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    with patch("backend.api.endpoints.staging.get_staging_pipeline") as mock_pipeline_fn:
        mock_pipeline = MagicMock()
        mock_pipeline_fn.return_value = mock_pipeline

        async def fake_generate(project):
            yield {"type": "project_completed", "status": "completed"}

        mock_pipeline.generate_project = fake_generate

        # /generate should return 200, /generate/stream should 404
        response = client.post("/api/v1/staging/projects/proj-789/generate")
        assert response.status_code == 200

        response_404 = client.post("/api/v1/staging/projects/proj-789/generate/stream")
        assert response_404.status_code in (404, 405)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_bug_fixes.py::test_generate_endpoint_exists_without_stream_suffix -v`
Expected: PASS

- [ ] **Step 3: Fix frontend streaming URLs**

In `frontend/services/stagingApi.ts`, in `streamGeneration()` replace:
```typescript
  const url = `${API_BASE_URL}/staging/projects/${projectId}/generate/stream`;
```
With:
```typescript
  const url = `${API_BASE_URL}/staging/projects/${projectId}/generate`;
```

In `streamRoomRegeneration()` replace:
```typescript
  const url = `${API_BASE_URL}/staging/projects/${projectId}/rooms/${roomId}/regenerate/stream`;
```
With:
```typescript
  const url = `${API_BASE_URL}/staging/projects/${projectId}/rooms/${roomId}/regenerate`;
```

- [ ] **Step 4: Commit**

```bash
git add frontend/services/stagingApi.ts tests/test_bug_fixes.py
git commit -m "fix: remove /stream suffix from SSE streaming URLs

Frontend was calling /generate/stream and /regenerate/stream but
backend routes are /generate and /regenerate. SSE streaming now connects.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Fix Lint Error — Unused getStatusColor

**Files:**
- Modify: `frontend/components/staging/RoomGroup.tsx:14`
- Test: `tests/test_bug_fixes.py`

- [ ] **Step 1: Write regression test**

Append to `tests/test_bug_fixes.py`:

```python
def test_roomgroup_no_unused_variables():
    """RoomGroup.tsx should not have unused variable declarations."""
    import re
    from pathlib import Path

    roomgroup_path = Path(__file__).resolve().parent.parent / "frontend" / "components" / "staging" / "RoomGroup.tsx"
    source = roomgroup_path.read_text()

    # getStatusColor should not be declared as a standalone variable
    # (it's fine if it's used inline or as a function call)
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("const getStatusColor") or stripped.startswith("function getStatusColor"):
            # Check if it's actually used elsewhere in the file
            rest_of_file = "\n".join(lines[i:])
            assert "getStatusColor" in rest_of_file, (
                f"getStatusColor declared on line {i} but never used in RoomGroup.tsx"
            )
```

- [ ] **Step 2: Run test — should FAIL (unused var exists)**

Run: `uv run pytest tests/test_bug_fixes.py::test_roomgroup_no_unused_variables -v`
Expected: FAIL

- [ ] **Step 3: Remove unused getStatusColor from RoomGroup.tsx**

In `frontend/components/staging/RoomGroup.tsx`, remove the line:
```typescript
const getStatusColor = (status: string) => {
```
and its entire function body (the declaration through the closing `};`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bug_fixes.py::test_roomgroup_no_unused_variables -v`
Expected: PASS

- [ ] **Step 5: Run frontend lint to verify clean**

Run: `cd frontend && npx next lint`
Expected: No errors on RoomGroup.tsx

- [ ] **Step 6: Commit**

```bash
git add frontend/components/staging/RoomGroup.tsx tests/test_bug_fixes.py
git commit -m "fix: remove unused getStatusColor from RoomGroup.tsx

Removes dead code flagged by eslint @typescript-eslint/no-unused-vars.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4b: Move mock_staging_deps Fixture to conftest.py

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_staging_api.py`

- [ ] **Step 1: Add mock_staging_deps to conftest.py**

In `tests/conftest.py`, add after the existing `mock_staging_storage` fixture:

```python
@pytest.fixture
def mock_staging_deps():
    """Mock all staging dependencies for endpoint tests."""
    with patch("backend.core.staging_storage.CosmosClient") as mock_cosmos, \
         patch("backend.core.staging_storage.DefaultAzureCredential") as mock_cred, \
         patch("backend.api.endpoints.staging.get_staging_pipeline") as mock_pipeline_fn:

        mock_client = MagicMock()
        mock_cosmos.return_value = mock_client
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_container = MagicMock()
        mock_db.create_container_if_not_exists.return_value = mock_container

        mock_cred.return_value = MagicMock()

        mock_pipeline = MagicMock()
        mock_pipeline_fn.return_value = mock_pipeline

        yield {"container": mock_container, "pipeline": mock_pipeline}
```

- [ ] **Step 2: Remove the duplicate fixture from test_staging_api.py**

In `tests/test_staging_api.py`, remove the `mock_staging_deps` fixture definition (it's now in conftest.py).

- [ ] **Step 3: Run all existing tests to verify no regressions**

Run: `uv run pytest tests/ --ignore=tests/integration -v`
Expected: ALL PASS (29 tests)

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_staging_api.py
git commit -m "refactor: move mock_staging_deps fixture to conftest.py

Shared fixture needed by multiple test files (test_staging_api,
test_bug_fixes, test_design_chat, test_backyard_scenario).

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 2: Backend Data Models

### Task 5: Add DesignBrief and Related Models

**Files:**
- Create: `backend/models/design_brief.py`
- Modify: `backend/models/staging.py`
- Test: `tests/test_design_brief.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_design_brief.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_design_brief.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create the models file**

Create `backend/models/design_brief.py`:

```python
"""Pydantic models for the AI Design Session and Design Brief."""
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from backend.models.staging import StagingSettings


class PlantEntry(BaseModel):
    species: str = Field(..., description="Common name, e.g. 'Vanderwolf's Pyramid Limber Pine'")
    botanical_name: Optional[str] = Field(None, description="e.g. 'Pinus flexilis Vanderwolf's Pyramid'")
    quantity: int = Field(1, description="Number of this species to place")
    size: str = Field("", description="e.g. '8-10 ft tall'")
    placement: str = Field("", description="e.g. 'back row along east fence'")
    visual_notes: Optional[str] = Field(None, description="Key visual characteristics for image generation")


class PlacementGuide(BaseModel):
    back_row: str = Field("", description="Tall plants / trees description")
    middle_row: Optional[str] = Field(None, description="Mid-height shrubs description")
    front_row: Optional[str] = Field(None, description="Low groundcover description")
    accent_areas: Optional[str] = Field(None, description="Special areas like pergola posts, patio edges")


class DesignBrief(BaseModel):
    global_instructions: str = Field(..., description="Overall styling direction synthesized from conversation")
    plant_palette: List[PlantEntry] = Field(default_factory=list)
    placement_guide: PlacementGuide = Field(default_factory=PlacementGuide)
    per_image_notes: Dict[str, str] = Field(default_factory=dict, description="room_id → specific note")
    preserve_elements: List[str] = Field(default_factory=list, description="Elements to keep unchanged")
    settings: StagingSettings = Field(default_factory=StagingSettings)


class ImageAnalysis(BaseModel):
    room_id: str
    description: str = Field(..., description="What the AI sees in this image")
    features: List[str] = Field(default_factory=list, description="Detected features: fence, turf, patio, etc.")
    zones: List[str] = Field(default_factory=list, description="Identifiable areas for object placement")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    focused_image_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's latest message")
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    focused_image_id: Optional[str] = Field(None, description="Room ID the user is focused on")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="AI's response text")
    ready_for_brief: bool = Field(False, description="True when AI has enough info to generate a brief")
    suggested_actions: List[str] = Field(default_factory=list, description="Suggested quick-reply action keys")


class GenerateBriefRequest(BaseModel):
    conversation_history: List[ChatMessage] = Field(default_factory=list, description="Full chat history for brief synthesis")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_design_brief.py -v`
Expected: ALL PASS (8 tests)

- [ ] **Step 5: Add optional design_brief and analyses to StagingProject**

In `backend/models/staging.py`, add imports at top:

```python
from typing import Any, Dict, List, Optional
```

Add to `StagingProject` class (after `folder_path`):

```python
    design_brief: Optional[Dict[str, Any]] = Field(None, description="Structured design brief from AI conversation")
    analyses: Optional[List[Dict[str, Any]]] = Field(None, description="Image analysis results")
```

- [ ] **Step 6: Run all existing tests to verify no regressions**

Run: `uv run pytest tests/ --ignore=tests/integration -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add backend/models/design_brief.py backend/models/staging.py tests/test_design_brief.py
git commit -m "feat: add DesignBrief, ImageAnalysis, and Chat models

New Pydantic models for the AI Design Session:
- PlantEntry, PlacementGuide, DesignBrief for structured briefs
- ImageAnalysis for per-image scene analysis
- ChatMessage, ChatRequest, ChatResponse for conversation
- StagingProject gains optional design_brief and analyses fields

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 3: Backend Services

### Task 6: Create DesignChatService

**Files:**
- Create: `backend/core/design_chat.py`
- Test: `tests/test_design_chat.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_design_chat.py`:

```python
"""Tests for DesignChatService and analyze endpoint."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock


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
    assert "FOCUSED IMAGE" in system_msg or "pergola" in system_msg.lower()


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_design_chat.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement DesignChatService**

Create `backend/core/design_chat.py`:

```python
"""DesignChatService — conversational AI for the Design Session."""
import json
import logging
from typing import List, Optional

from backend.models.design_brief import ChatMessage, ChatResponse, ImageAnalysis

logger = logging.getLogger(__name__)

DESIGN_CHAT_SYSTEM_PROMPT = """You are a landscape and interior design assistant helping users plan visual changes to their spaces.

IMAGE ANALYSES:
{analyses_text}

{focused_image_section}

Your job is to have a natural conversation to understand what the user wants to visualize. Ask about:
- What they want to add (plants, furniture, structures, etc.)
- Specific species, materials, or styles
- Where to place items (which areas, which images)
- Quantities and sizes
- What existing elements to preserve unchanged
- Any seasonal or style preferences

After gathering enough detail (typically 3-5 substantive exchanges), set ready_for_brief to true.

ALWAYS respond with valid JSON matching this schema:
{{"reply": "your message", "ready_for_brief": false, "suggested_actions": ["action_key1", "action_key2"]}}

suggested_actions are short keys like: specify_species, choose_density, set_height_preference, define_placement, choose_style, add_more_areas, generate_brief"""


class DesignChatService:
    """Handles conversational AI for the Design Session step."""

    def __init__(
        self,
        async_llm_client,
        llm_deployment: str,
        image_analyses: List[ImageAnalysis],
    ):
        self.async_llm_client = async_llm_client
        self.llm_deployment = llm_deployment
        self.image_analyses = image_analyses

    def _build_analyses_text(self) -> str:
        parts = []
        for a in self.image_analyses:
            parts.append(
                f"- Image '{a.room_id}': {a.description} "
                f"(features: {', '.join(a.features)}; zones: {', '.join(a.zones)})"
            )
        return "\n".join(parts) if parts else "No images analyzed yet."

    def _build_focused_section(self, focused_image_id: Optional[str]) -> str:
        if not focused_image_id:
            return ""
        for a in self.image_analyses:
            if a.room_id == focused_image_id:
                return (
                    f"\nFOCUSED IMAGE: The user is currently looking at image '{a.room_id}'.\n"
                    f"Description: {a.description}\n"
                    f"Features: {', '.join(a.features)}\n"
                    f"Zones: {', '.join(a.zones)}\n"
                    f"Tailor your response to this specific image."
                )
        return ""

    async def chat(
        self,
        message: str,
        conversation_history: List[ChatMessage],
        focused_image_id: Optional[str] = None,
    ) -> ChatResponse:
        system_content = DESIGN_CHAT_SYSTEM_PROMPT.format(
            analyses_text=self._build_analyses_text(),
            focused_image_section=self._build_focused_section(focused_image_id),
        )

        messages = [{"role": "system", "content": system_content}]
        for msg in conversation_history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": message})

        for attempt in range(3):
            response = await self.async_llm_client.chat.completions.create(
                model=self.llm_deployment,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            try:
                content = response.choices[0].message.content
                parsed = json.loads(content)
                return ChatResponse(
                    reply=parsed.get("reply", content),
                    ready_for_brief=parsed.get("ready_for_brief", False),
                    suggested_actions=parsed.get("suggested_actions", []),
                )
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Chat attempt {attempt + 1} returned invalid JSON, retrying")
                continue

        return ChatResponse(reply="I'm having trouble processing that. Could you rephrase?")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_design_chat.py -v`
Expected: ALL PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/core/design_chat.py tests/test_design_chat.py
git commit -m "feat: add DesignChatService for AI Design Session

Conversational service that takes image analyses as context, supports
focused-image highlighting, and signals ready_for_brief when enough
detail is gathered. Returns structured JSON with suggested actions.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: Create BriefGeneratorService

**Files:**
- Create: `backend/core/brief_generator.py`
- Test: `tests/test_design_brief.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_design_brief.py`:

```python
@pytest.mark.asyncio
async def test_brief_generation_from_conversation():
    from backend.core.brief_generator import BriefGeneratorService
    from backend.models.design_brief import ChatMessage, ImageAnalysis
    import json

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({
            "global_instructions": "Add layered evergreen privacy screen along fence line",
            "plant_palette": [
                {
                    "species": "Vanderwolf's Pyramid Limber Pine",
                    "botanical_name": "Pinus flexilis 'Vanderwolf's Pyramid'",
                    "quantity": 3,
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
    assert len(brief.plant_palette) == 1
    assert brief.plant_palette[0].species == "Vanderwolf's Pyramid Limber Pine"
    assert "patio" in brief.preserve_elements


@pytest.mark.asyncio
async def test_brief_to_prompts_produces_specific_prompts():
    from backend.core.brief_generator import BriefGeneratorService
    from backend.models.design_brief import (
        DesignBrief, PlantEntry, PlacementGuide, ImageAnalysis,
    )
    import json

    mock_llm = AsyncMock()
    mock_llm.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps([
            "Along the fence in this backyard, add 3 Vanderwolf's Pyramid Limber Pines with silvery-blue needles in back row",
            "Place 3 narrow pyramid conifers with twisted blue-green needles along the wooden fence line",
        ])))]
    ))

    brief = DesignBrief(
        global_instructions="Add trees along fence",
        plant_palette=[
            PlantEntry(
                species="Vanderwolf's Pyramid Limber Pine",
                quantity=3,
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
```

Add imports at top of file:

```python
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_design_brief.py::test_brief_generation_from_conversation tests/test_design_brief.py::test_brief_to_prompts_produces_specific_prompts -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement BriefGeneratorService**

Create `backend/core/brief_generator.py`:

```python
"""BriefGeneratorService — synthesizes conversation into a structured Design Brief."""
import json
import logging
from typing import Dict, List

from backend.models.design_brief import (
    ChatMessage, DesignBrief, ImageAnalysis, PlantEntry, PlacementGuide,
)

logger = logging.getLogger(__name__)

BRIEF_GENERATION_PROMPT = """You are a design assistant. Synthesize the conversation below into a structured Design Brief.

IMAGE ANALYSES:
{analyses_text}

CONVERSATION:
{conversation_text}

Extract and organize all design decisions into this exact JSON structure:
{{
  "global_instructions": "Overall description of what to add and the style direction",
  "plant_palette": [
    {{
      "species": "Common name",
      "botanical_name": "Scientific name or null",
      "quantity": 1,
      "size": "height description",
      "placement": "where to put it",
      "visual_notes": "key visual characteristics for image generation"
    }}
  ],
  "placement_guide": {{
    "back_row": "Tall plants description",
    "middle_row": "Mid-height description or null",
    "front_row": "Low plants description or null",
    "accent_areas": "Special areas or null"
  }},
  "per_image_notes": {{}},
  "preserve_elements": ["list of things to keep unchanged"]
}}

Be specific about visual characteristics — these will be used to generate images."""

BRIEF_TO_PROMPTS_TEMPLATE = """You are an image editing prompt writer. Given a Design Brief and an image description, generate {n} distinct prompts for an image editing model.

DESIGN BRIEF:
Global: {global_instructions}
Plants: {plant_summary}
Placement: {placement_summary}
Preserve: {preserve_summary}

IMAGE DESCRIPTION: {image_description}
{per_image_note}

Generate {n} variation prompts. Each should:
- ADD the specified plants/items to the scene described above
- Reference specific species with their visual characteristics
- Respect the placement guide (back row, middle, front)
- NOT remove or change elements listed in preserve
- Vary the interpretation: different arrangements, densities, or seasonal looks

Return ONLY a JSON array of {n} strings."""


class BriefGeneratorService:
    """Generates structured Design Briefs from conversations and converts them to prompts."""

    def __init__(self, async_llm_client, llm_deployment: str):
        self.async_llm_client = async_llm_client
        self.llm_deployment = llm_deployment

    async def generate_brief(
        self,
        conversation_history: List[ChatMessage],
        image_analyses: List[ImageAnalysis],
    ) -> DesignBrief:
        analyses_text = "\n".join(
            f"- {a.room_id}: {a.description} (features: {', '.join(a.features)})"
            for a in image_analyses
        )
        conversation_text = "\n".join(
            f"{msg.role.upper()}: {msg.content}" for msg in conversation_history
        )

        system_content = BRIEF_GENERATION_PROMPT.format(
            analyses_text=analyses_text,
            conversation_text=conversation_text,
        )

        for attempt in range(3):
            response = await self.async_llm_client.chat.completions.create(
                model=self.llm_deployment,
                messages=[{"role": "system", "content": system_content}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            try:
                parsed = json.loads(response.choices[0].message.content)
                return DesignBrief(
                    global_instructions=parsed.get("global_instructions", ""),
                    plant_palette=[PlantEntry(**p) for p in parsed.get("plant_palette", [])],
                    placement_guide=PlacementGuide(**parsed.get("placement_guide", {})),
                    per_image_notes=parsed.get("per_image_notes", {}),
                    preserve_elements=parsed.get("preserve_elements", []),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Brief generation attempt {attempt + 1} failed: {e}")
                continue

        raise RuntimeError("Failed to generate Design Brief after 3 attempts")

    async def brief_to_prompts(
        self,
        brief: DesignBrief,
        image_analyses: List[ImageAnalysis],
        n_variations: int = 5,
    ) -> Dict[str, List[str]]:
        plant_summary = "; ".join(
            f"{p.quantity}x {p.species} ({p.size}, {p.placement})"
            + (f" — {p.visual_notes}" if p.visual_notes else "")
            for p in brief.plant_palette
        )
        placement_summary = f"Back: {brief.placement_guide.back_row}"
        if brief.placement_guide.middle_row:
            placement_summary += f" | Middle: {brief.placement_guide.middle_row}"
        if brief.placement_guide.front_row:
            placement_summary += f" | Front: {brief.placement_guide.front_row}"
        preserve_summary = ", ".join(brief.preserve_elements) if brief.preserve_elements else "None specified"

        result: Dict[str, List[str]] = {}

        for analysis in image_analyses:
            per_image_note = ""
            if analysis.room_id in brief.per_image_notes:
                per_image_note = f"SPECIAL NOTE FOR THIS IMAGE: {brief.per_image_notes[analysis.room_id]}"

            system_content = BRIEF_TO_PROMPTS_TEMPLATE.format(
                n=n_variations,
                global_instructions=brief.global_instructions,
                plant_summary=plant_summary,
                placement_summary=placement_summary,
                preserve_summary=preserve_summary,
                image_description=analysis.description,
                per_image_note=per_image_note,
            )

            for attempt in range(3):
                response = await self.async_llm_client.chat.completions.create(
                    model=self.llm_deployment,
                    messages=[{"role": "system", "content": system_content}],
                    temperature=0.8,
                    response_format={"type": "json_object"},
                )
                try:
                    content = response.choices[0].message.content
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        result[analysis.room_id] = [str(p) for p in parsed[:n_variations]]
                        break
                    if isinstance(parsed, dict) and "prompts" in parsed:
                        result[analysis.room_id] = [str(p) for p in parsed["prompts"][:n_variations]]
                        break
                except (json.JSONDecodeError, KeyError):
                    logger.warning(f"Prompt generation attempt {attempt + 1} for {analysis.room_id} failed")
                    continue
            else:
                result[analysis.room_id] = [brief.global_instructions] * n_variations

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_design_brief.py -v`
Expected: ALL PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/core/brief_generator.py tests/test_design_brief.py
git commit -m "feat: add BriefGeneratorService for Design Brief generation

Synthesizes conversation history + image analyses into a structured
DesignBrief. Also converts briefs into per-image variation prompts
that reference specific plant species and visual characteristics.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 8: Update Prompt Adaptation for Outdoor Context

**Files:**
- Modify: `backend/core/staging_pipeline.py:22-34`
- Test: `tests/test_design_brief.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_design_brief.py`:

```python
def test_outdoor_prompt_template_detects_landscape_context():
    from backend.core.staging_pipeline import build_adaptation_template

    template = build_adaptation_template(
        room_analysis="A backyard with wooden fence, turf, and patio",
        is_outdoor=True,
    )
    assert "landscape" in template.lower() or "outdoor" in template.lower()
    assert "room" not in template.lower() or "outdoor" in template.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_design_brief.py::test_outdoor_prompt_template_detects_landscape_context -v`
Expected: FAIL (function not found)

- [ ] **Step 3: Add build_adaptation_template function**

In `backend/core/staging_pipeline.py`, replace the `PROMPT_ADAPTATION_TEMPLATE` constant with:

```python
INDOOR_PROMPT_TEMPLATE = """You are a virtual staging assistant. The user wants to visualize decorating ideas for their space.

ROOM ANALYSIS: {room_analysis}
USER'S STYLE DIRECTION: {user_prompt}

Generate {n} distinct variation prompts for an image editing model. Each prompt should:
- ADD items to the existing scene (furniture, decor, plants)
- NOT remove or replace existing structures visible in the analysis
- Interpret the user's style direction differently in each variation
- Be specific about what to add and where to place it
- Reference the existing room features from the analysis

Return ONLY a JSON array of {n} strings. No other text."""

OUTDOOR_PROMPT_TEMPLATE = """You are a landscape visualization assistant. The user wants to visualize landscaping and outdoor design ideas.

SCENE ANALYSIS: {room_analysis}
USER'S DESIGN DIRECTION: {user_prompt}

Generate {n} distinct variation prompts for an image editing model. Each prompt should:
- ADD plants, trees, shrubs, hardscaping, or outdoor elements to the existing scene
- NOT remove or replace existing structures (patios, fences, pergolas, fire pits)
- Specify plant species with visual characteristics (leaf color, form, texture, size)
- Describe placement using landscape terms (back row, border, along fence, flanking)
- Interpret the design direction differently in each variation
- Reference the existing outdoor features from the analysis

Return ONLY a JSON array of {n} strings. No other text."""

OUTDOOR_KEYWORDS = {"backyard", "fence", "patio", "pergola", "turf", "lawn", "garden",
                     "yard", "outdoor", "landscape", "deck", "driveway", "tree", "shrub"}


def build_adaptation_template(room_analysis: str, is_outdoor: bool = False) -> str:
    """Return the appropriate prompt template based on context."""
    if is_outdoor:
        return OUTDOOR_PROMPT_TEMPLATE
    analysis_lower = room_analysis.lower()
    if any(kw in analysis_lower for kw in OUTDOOR_KEYWORDS):
        return OUTDOOR_PROMPT_TEMPLATE
    return INDOOR_PROMPT_TEMPLATE
```

Then update `adapt_prompt` method to use it. In the `adapt_prompt` method, replace:
```python
        system_content = PROMPT_ADAPTATION_TEMPLATE.format(
```
With:
```python
        template = build_adaptation_template(room_analysis)
        system_content = template.format(
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_design_brief.py::test_outdoor_prompt_template_detects_landscape_context -v`
Expected: PASS

- [ ] **Step 5: Run all tests to verify no regressions**

Run: `uv run pytest tests/ --ignore=tests/integration -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add backend/core/staging_pipeline.py tests/test_design_brief.py
git commit -m "feat: add outdoor/landscape prompt template with auto-detection

Prompt adaptation now uses outdoor-specific language when the image
analysis mentions backyard, fence, patio, etc. Indoor template
unchanged. Auto-detects context from analysis keywords.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 4: Backend Endpoints

### Task 9: Add Analyze Endpoint

**Files:**
- Modify: `backend/api/endpoints/staging.py`
- Test: `tests/test_design_chat.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_design_chat.py`:

```python
def test_analyze_endpoint_returns_analyses(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = {
        "id": "proj-analyze",
        "name": "Test",
        "prompt": "Test",
        "status": "uploading",
        "rooms": [
            {
                "id": "room-1",
                "label": "Backyard East",
                "original_image_url": "https://test.blob.core.windows.net/images/staging/proj-analyze/originals/img.png",
                "status": "pending",
                "variations": [],
            }
        ],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    with patch("backend.api.endpoints.staging.get_image_analyzer") as mock_analyzer_fn:
        mock_analyzer = AsyncMock()
        mock_analyzer_fn.return_value = mock_analyzer
        mock_analyzer.async_image_chat = AsyncMock(return_value={
            "description": "Backyard with wooden fence and turf",
            "features": ["fence", "turf"],
        })

        with patch("backend.api.endpoints.staging.AzureBlobStorageService") as mock_blob_cls:
            mock_blob = AsyncMock()
            mock_blob_cls.return_value = mock_blob
            mock_blob.get_asset_content = AsyncMock(return_value=b"fake-image-bytes")

            response = client.post("/api/v1/staging/projects/proj-analyze/analyze")

    assert response.status_code == 200
    data = response.json()
    assert "analyses" in data
    assert len(data["analyses"]) == 1
    assert data["analyses"][0]["room_id"] == "room-1"
    assert "fence" in data["analyses"][0]["description"].lower() or len(data["analyses"][0]["features"]) > 0
```

Add this import at the top of `tests/test_design_chat.py`:

```python
from unittest.mock import patch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_design_chat.py::test_analyze_endpoint_returns_analyses -v`
Expected: FAIL (404)

- [ ] **Step 3: Add the analyze endpoint**

In `backend/api/endpoints/staging.py`, add imports at top:

```python
from backend.models.design_brief import ImageAnalysis, ChatRequest, ChatResponse, DesignBrief, GenerateBriefRequest
```

Add helper function:

```python
def get_image_analyzer():
    from backend.core import async_llm_client
    from backend.core.analyze import ImageAnalyzer
    return ImageAnalyzer(
        openai_client=None,
        model=settings.LLM_DEPLOYMENT,
        async_openai_client=async_llm_client,
    )
```

Add the endpoint:

```python
@router.post("/projects/{project_id}/analyze")
async def analyze_project_images(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
    analyzer: "ImageAnalyzer" = Depends(get_image_analyzer),
):
    """Analyze all uploaded images in the project. Returns structured analyses."""
    import asyncio
    import base64

    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    rooms = project_data.get("rooms", [])
    if not rooms:
        raise HTTPException(status_code=400, detail="No images uploaded yet")

    blob_service = AzureBlobStorageService()

    system_msg = (
        "Describe this scene in detail for a design assistant. Include: "
        "existing structures, plants, materials, colors, spatial layout, "
        "and identifiable zones where items could be added. "
        "Return JSON with keys: description (string), features (list of strings), "
        "zones (list of strings identifying areas suitable for placing new objects)."
    )

    async def analyze_one(room: dict) -> dict:
        url = room["original_image_url"]
        blob_name = "/".join(url.split("/")[-2:])
        image_bytes = await blob_service.get_asset_content(
            blob_name=blob_name,
            container_name=settings.AZURE_BLOB_IMAGE_CONTAINER,
        )
        image_b64 = base64.b64encode(image_bytes).decode("utf-8") if isinstance(image_bytes, bytes) else image_bytes
        result = await analyzer.async_image_chat(image_base64=image_b64, system_message=system_msg)
        return {
            "room_id": room["id"],
            "description": result.get("description", ""),
            "features": result.get("features", []),
            "zones": result.get("zones", []),
        }

    analyses = await asyncio.gather(*[analyze_one(r) for r in rooms], return_exceptions=True)
    valid_analyses = [a for a in analyses if isinstance(a, dict)]

    # Persist analyses to project
    storage.update_project(project_id, {"analyses": valid_analyses})

    return {"analyses": valid_analyses}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_design_chat.py::test_analyze_endpoint_returns_analyses -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/api/endpoints/staging.py tests/test_design_chat.py
git commit -m "feat: add POST /projects/{id}/analyze endpoint

Analyzes all uploaded images in parallel using the existing ImageAnalyzer.
Returns per-image descriptions, features, and spatial zones. Results
persisted to project for use in the chat and brief generation steps.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 10: Add Chat Endpoint

**Files:**
- Modify: `backend/api/endpoints/staging.py`

- [ ] **Step 1: Add the chat endpoint**

In `backend/api/endpoints/staging.py`, add:

```python
@router.post("/projects/{project_id}/chat", response_model=ChatResponse)
async def chat_with_project(
    project_id: str,
    request: ChatRequest,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Conversational AI endpoint for the Design Session."""
    from backend.core import async_llm_client
    from backend.core.design_chat import DesignChatService

    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    raw_analyses = project_data.get("analyses", [])
    analyses = [ImageAnalysis(**a) for a in raw_analyses]

    service = DesignChatService(
        async_llm_client=async_llm_client,
        llm_deployment=settings.LLM_DEPLOYMENT,
        image_analyses=analyses,
    )

    response = await service.chat(
        message=request.message,
        conversation_history=request.conversation_history,
        focused_image_id=request.focused_image_id,
    )

    return response
```

- [ ] **Step 2: Run all tests to verify no regressions**

Run: `uv run pytest tests/ --ignore=tests/integration -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add backend/api/endpoints/staging.py
git commit -m "feat: add POST /projects/{id}/chat endpoint

Conversational AI endpoint for the Design Session. Uses image analyses
as context, supports focused-image highlighting, returns structured
responses with suggested actions.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 11: Add Brief Endpoint

**Files:**
- Modify: `backend/api/endpoints/staging.py`

- [ ] **Step 1: Add the brief endpoints (POST + PUT)**

In `backend/api/endpoints/staging.py`, add:

```python
@router.post("/projects/{project_id}/brief")
async def generate_brief(
    project_id: str,
    request: Optional[GenerateBriefRequest] = None,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Generate a structured Design Brief from the conversation."""
    from backend.core import async_llm_client
    from backend.core.brief_generator import BriefGeneratorService

    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    raw_analyses = project_data.get("analyses", [])
    analyses = [ImageAnalysis(**a) for a in raw_analyses]

    conversation_history = []
    if request and request.conversation_history:
        from backend.models.design_brief import ChatMessage as CM
        conversation_history = request.conversation_history

    service = BriefGeneratorService(
        async_llm_client=async_llm_client,
        llm_deployment=settings.LLM_DEPLOYMENT,
    )

    brief = await service.generate_brief(
        conversation_history=conversation_history,
        image_analyses=analyses,
    )

    brief_dict = brief.dict()
    storage.update_project(project_id, {"design_brief": brief_dict})

    return {"brief": brief_dict}


@router.put("/projects/{project_id}/brief")
async def update_brief(
    project_id: str,
    brief: DesignBrief,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Save user edits to the Design Brief."""
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    brief_dict = brief.dict()
    storage.update_project(project_id, {"design_brief": brief_dict})

    return {"brief": brief_dict}
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest tests/ --ignore=tests/integration -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add backend/api/endpoints/staging.py
git commit -m "feat: add POST/PUT /projects/{id}/brief endpoints

POST generates a structured Design Brief from conversation + analyses.
PUT saves user edits from the Design Brief Editor. Brief is persisted
to the project document in Cosmos DB.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 11b: Wire DesignBrief into Generation Pipeline

**Files:**
- Modify: `backend/core/staging_pipeline.py`
- Modify: `backend/api/endpoints/staging.py`

- [ ] **Step 1: Update generate_project to use DesignBrief when present**

In `backend/core/staging_pipeline.py`, update the `process_room` method. After the line that calls `adapt_prompt`, add a branch that uses `BriefGeneratorService.brief_to_prompts()` when a design brief is present:

In the `generate_project` method, before the room processing loop, add:

```python
    # If project has a design_brief, use BriefGeneratorService for prompt adaptation
    brief_prompts = {}
    if project.design_brief:
        from backend.core.brief_generator import BriefGeneratorService
        from backend.models.design_brief import DesignBrief as DBModel, ImageAnalysis

        brief = DBModel(**project.design_brief)
        analyses = [ImageAnalysis(**a) for a in (project.analyses or [])]
        brief_service = BriefGeneratorService(
            async_llm_client=self.async_llm_client,
            llm_deployment=self.llm_deployment,
        )
        brief_prompts = await brief_service.brief_to_prompts(
            brief=brief,
            image_analyses=analyses,
            n_variations=project.settings.variations_per_room,
        )
```

Then in `process_room`, before calling `self.adapt_prompt()`, add:

```python
                # Use brief-generated prompts if available, else fall back to adapt_prompt
                if room.id in brief_prompts:
                    adapted_prompts = brief_prompts[room.id]
                else:
                    adapted_prompts = await self.adapt_prompt(
                        user_prompt=project.prompt,
                        room_analysis=room_description,
                        n_variations=project.settings.variations_per_room,
                    )
```

This replaces the existing `adapt_prompt` call with a conditional that prefers brief-generated prompts.

- [ ] **Step 2: Pass brief_prompts into process_room**

Update `process_room` signature to accept `brief_prompts: Dict[str, List[str]] = None` and thread it through from `generate_project`.

- [ ] **Step 3: Run all backend tests**

Run: `uv run pytest tests/ --ignore=tests/integration -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add backend/core/staging_pipeline.py
git commit -m "feat: wire DesignBrief into generation pipeline

When a project has a design_brief, use BriefGeneratorService to produce
targeted per-image prompts instead of the generic adapt_prompt path.
Falls back to adapt_prompt for projects without a brief.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 5: Backyard Scenario Tests

### Task 12: Backyard Landscaping Scenario Tests

**Files:**
- Create: `tests/test_backyard_scenario.py`

- [ ] **Step 1: Create the scenario test file**

Create `tests/test_backyard_scenario.py`:

```python
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
    assert len(BACKYARD_IMAGES) == 14, f"Expected 14 images, found {len(BACKYARD_IMAGES)}"
    assert BACKYARD_MD.exists(), "BACKYARD.md missing"


def test_backyard_project_creation(client, mock_staging_deps):
    """Create a project and verify it accepts all 14 images."""
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
            "plant_palette": [
                {
                    "species": "Vanderwolf's Pyramid Limber Pine",
                    "botanical_name": "Pinus flexilis 'Vanderwolf's Pyramid'",
                    "quantity": 3,
                    "size": "20-25 ft tall",
                    "placement": "back row along fence",
                    "visual_notes": "Blue-green to silvery-blue twisted needles in bundles of 5, narrow pyramid silhouette",
                },
                {
                    "species": "Baby Blue Eyes Spruce",
                    "botanical_name": "Picea pungens 'Baby Blue Eyes'",
                    "quantity": 2,
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
    brief = await service.generate_brief(conversation_history=history, image_analyses=analyses)

    assert len(brief.plant_palette) == 2
    pine = next(p for p in brief.plant_palette if "Vanderwolf" in p.species)
    assert pine.quantity == 3
    assert "silvery" in pine.visual_notes.lower() or "blue" in pine.visual_notes.lower()

    spruce = next(p for p in brief.plant_palette if "Baby Blue" in p.species)
    assert spruce.quantity == 2


@pytest.mark.asyncio
async def test_backyard_adapted_prompts_are_specific():
    """Verify adapted prompts reference specific plants and scene features."""
    from backend.core.brief_generator import BriefGeneratorService
    from backend.models.design_brief import (
        DesignBrief, PlantEntry, PlacementGuide, ImageAnalysis,
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
        plant_palette=[
            PlantEntry(species="Vanderwolf's Pyramid Limber Pine", quantity=3, size="20-25 ft",
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
    # Prompts should be specific — not generic "add plants"
    for prompt in prompts["fence-east"]:
        assert len(prompt) > 50, f"Prompt too short to be specific: {prompt}"


@pytest.mark.asyncio
async def test_backyard_per_image_notes_differ():
    """Verify per-image notes produce different prompts for pergola vs fence."""
    from backend.core.brief_generator import BriefGeneratorService
    from backend.models.design_brief import (
        DesignBrief, PlantEntry, PlacementGuide, ImageAnalysis,
    )

    call_count = 0

    async def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
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
        plant_palette=[PlantEntry(species="Columnar Norway Spruce", quantity=5, placement="along fence")],
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
    # The pergola prompt should differ from the fence prompt
    fence_prompt = prompts["fence-1"][0].lower()
    pergola_prompt = prompts["pergola-1"][0].lower()
    assert fence_prompt != pergola_prompt
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_backyard_scenario.py -v`
Expected: ALL PASS (7 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_backyard_scenario.py
git commit -m "test: add backyard landscaping scenario tests

6 scenario tests using actual test data from tests/projects/backyard-landscaping/.
Covers project creation, chat with plant selection, brief generation with
visual details, specific prompt adaptation, and per-image note differentiation.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 6: Frontend API Service Updates

### Task 13: Add New API Functions to stagingApi.ts

**Files:**
- Modify: `frontend/services/stagingApi.ts`

- [ ] **Step 1: Add TypeScript types for new endpoints**

Append to the types section of `frontend/services/stagingApi.ts`:

```typescript
// Design Brief types
export interface PlantEntry {
  species: string;
  botanical_name?: string;
  quantity: number;
  size: string;
  placement: string;
  visual_notes?: string;
}

export interface PlacementGuide {
  back_row: string;
  middle_row?: string;
  front_row?: string;
  accent_areas?: string;
}

export interface DesignBrief {
  global_instructions: string;
  plant_palette: PlantEntry[];
  placement_guide: PlacementGuide;
  per_image_notes: Record<string, string>;
  preserve_elements: string[];
  settings: {
    variations_per_room: number;
    model: string;
    quality: string;
    size: string;
  };
}

export interface ImageAnalysisResult {
  room_id: string;
  description: string;
  features: string[];
  zones: string[];
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  focused_image_id?: string;
}

export interface ChatResponse {
  reply: string;
  ready_for_brief: boolean;
  suggested_actions: string[];
}
```

- [ ] **Step 2: Add API functions**

Append to `frontend/services/stagingApi.ts`:

```typescript
/**
 * Analyze all uploaded images in a project
 */
export async function analyzeImages(projectId: string): Promise<ImageAnalysisResult[]> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/analyze`;

  const response = await fetch(url, { method: 'POST' });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to analyze images: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  return data.analyses ?? [];
}

/**
 * Send a chat message in the AI Design Session
 */
export async function chatWithProject(
  projectId: string,
  message: string,
  conversationHistory: ChatMessage[],
  focusedImageId?: string,
): Promise<ChatResponse> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/chat`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      conversation_history: conversationHistory,
      focused_image_id: focusedImageId ?? null,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Chat failed: ${response.status} ${errorText}`);
  }

  return response.json();
}

/**
 * Generate a Design Brief from the conversation
 */
export async function generateBrief(projectId: string, conversationHistory: ChatMessage[]): Promise<DesignBrief> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/brief`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversation_history: conversationHistory }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to generate brief: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  return data.brief;
}

/**
 * Save user edits to the Design Brief
 */
export async function updateBrief(projectId: string, brief: DesignBrief): Promise<DesignBrief> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/brief`;

  const response = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(brief),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to update brief: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  return data.brief;
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/services/stagingApi.ts
git commit -m "feat: add analyzeImages, chat, brief API functions

New frontend API functions for the AI Design Session:
- analyzeImages() — triggers image analysis
- chatWithProject() — conversational AI endpoint
- generateBrief() / updateBrief() — Design Brief CRUD

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 7: Frontend Components

### Task 14: Create QuickReplyChips Component

**Files:**
- Create: `frontend/components/staging/QuickReplyChips.tsx`

- [ ] **Step 1: Create the component**

```tsx
"use client"

const ACTION_LABELS: Record<string, string> = {
  specify_species: "🌲 Specify plant species",
  choose_density: "📏 Set planting density",
  set_height_preference: "📐 Define height layers",
  define_placement: "📍 Describe placement",
  choose_style: "🎨 Choose a style",
  add_more_areas: "➕ Add more areas",
  generate_brief: "📋 Generate Design Brief",
  specify_quantity: "🔢 Specify quantities",
  add_more_species: "🌿 Add more species",
};

interface QuickReplyChipsProps {
  actions: string[];
  onSelect: (action: string) => void;
  disabled?: boolean;
}

export function QuickReplyChips({ actions, onSelect, disabled = false }: QuickReplyChipsProps) {
  if (!actions.length) return null;

  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {actions.map((action) => (
        <button
          key={action}
          onClick={() => onSelect(action)}
          disabled={disabled}
          className="px-3 py-1.5 text-xs rounded-full border border-border bg-muted/50 
                     hover:bg-muted hover:border-primary/50 transition-colors
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {ACTION_LABELS[action] ?? action}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/staging/QuickReplyChips.tsx
git commit -m "feat: add QuickReplyChips component

Renders AI-suggested clickable pills below chat messages. Maps
action keys to human-readable labels with emoji prefixes.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 15: Create ImageGalleryPanel Component

**Files:**
- Create: `frontend/components/staging/ImageGalleryPanel.tsx`

- [ ] **Step 1: Create the component**

```tsx
"use client"

import { useState } from "react";
import { ChevronDown, ChevronRight, Eye } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { ImageAnalysisResult } from "@/services/stagingApi";

interface ImageItem {
  id: string;
  label: string;
  url: string;
}

interface ImageGalleryPanelProps {
  images: ImageItem[];
  analyses: ImageAnalysisResult[];
  focusedImageId: string | null;
  onFocusImage: (imageId: string | null) => void;
  perImageNotes: Record<string, string>;
}

interface ImageGroup {
  name: string;
  images: ImageItem[];
}

function groupImages(images: ImageItem[], analyses: ImageAnalysisResult[]): ImageGroup[] {
  const analysisMap = new Map(analyses.map(a => [a.room_id, a]));
  const groups = new Map<string, ImageItem[]>();

  for (const img of images) {
    const analysis = analysisMap.get(img.id);
    const primaryFeature = analysis?.features[0] ?? "Other";
    const groupName = primaryFeature.charAt(0).toUpperCase() + primaryFeature.slice(1);
    if (!groups.has(groupName)) groups.set(groupName, []);
    groups.get(groupName)!.push(img);
  }

  return Array.from(groups.entries()).map(([name, imgs]) => ({ name, images: imgs }));
}

export function ImageGalleryPanel({
  images,
  analyses,
  focusedImageId,
  onFocusImage,
  perImageNotes,
}: ImageGalleryPanelProps) {
  const groups = groupImages(images, analyses);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const toggleGroup = (name: string) => {
    setCollapsed(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm">Your Photos</h3>
        <span className="text-xs text-muted-foreground">{images.length} images</span>
      </div>

      {groups.map(group => (
        <div key={group.name}>
          <button
            onClick={() => toggleGroup(group.name)}
            className="flex items-center gap-1 text-xs font-semibold text-primary uppercase tracking-wide mb-2 w-full"
          >
            {collapsed.has(group.name) ? <ChevronRight className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            {group.name} ({group.images.length})
          </button>

          {!collapsed.has(group.name) && (
            <div className="grid grid-cols-3 gap-1.5">
              {group.images.map(img => (
                <button
                  key={img.id}
                  onClick={() => onFocusImage(focusedImageId === img.id ? null : img.id)}
                  className={`relative aspect-video rounded overflow-hidden border-2 transition-colors ${
                    focusedImageId === img.id ? "border-primary" : "border-transparent hover:border-muted-foreground/30"
                  }`}
                >
                  <img src={img.url} alt={img.label} className="w-full h-full object-cover" />
                  {focusedImageId === img.id && (
                    <div className="absolute top-0 right-0 bg-primary rounded-bl p-0.5">
                      <Eye className="h-3 w-3 text-primary-foreground" />
                    </div>
                  )}
                  {perImageNotes[img.id] && (
                    <div className="absolute bottom-0 left-0 bg-yellow-500/80 rounded-tr px-1">
                      <span className="text-[9px] text-black font-medium">NOTE</span>
                    </div>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      ))}

      <p className="text-[10px] text-muted-foreground text-center pt-2">
        Click any image to focus the conversation on that area
      </p>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/staging/ImageGalleryPanel.tsx
git commit -m "feat: add ImageGalleryPanel component

Left panel for the split-panel Design Session. Groups thumbnails by
AI-detected feature, supports click-to-focus with visual indicator,
shows per-image note badges, and collapsible groups.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 16: Create DesignChat Component

**Files:**
- Create: `frontend/components/staging/DesignChat.tsx`

- [ ] **Step 1: Create the component**

```tsx
"use client"

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { QuickReplyChips } from "./QuickReplyChips";
import { chatWithProject, ChatMessage, ChatResponse } from "@/services/stagingApi";

interface DesignChatProps {
  projectId: string;
  focusedImageId: string | null;
  focusedImageLabel: string | null;
  onClearFocus: () => void;
  onReadyForBrief: () => void;
  initialMessage: string;
  conversationHistory: ChatMessage[];
  onHistoryUpdate: (history: ChatMessage[]) => void;
}

export function DesignChat({
  projectId,
  focusedImageId,
  focusedImageLabel,
  onClearFocus,
  onReadyForBrief,
  initialMessage,
  conversationHistory,
  onHistoryUpdate,
}: DesignChatProps) {
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [suggestedActions, setSuggestedActions] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversationHistory]);

  const sendMessage = async (message: string) => {
    if (!message.trim() || isLoading) return;

    const userMsg: ChatMessage = { role: "user", content: message, focused_image_id: focusedImageId ?? undefined };
    const updatedHistory = [...conversationHistory, userMsg];
    onHistoryUpdate(updatedHistory);
    setInput("");
    setIsLoading(true);
    setSuggestedActions([]);

    try {
      const response: ChatResponse = await chatWithProject(
        projectId,
        message,
        updatedHistory.slice(0, -1),
        focusedImageId ?? undefined,
      );

      const assistantMsg: ChatMessage = { role: "assistant", content: response.reply };
      onHistoryUpdate([...updatedHistory, assistantMsg]);
      setSuggestedActions(response.suggested_actions);

      if (response.ready_for_brief) {
        setSuggestedActions(["generate_brief"]);
      }
    } catch (error) {
      const errorMsg: ChatMessage = {
        role: "assistant",
        content: "Sorry, I had trouble processing that. Could you try again?",
      };
      onHistoryUpdate([...updatedHistory, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChipSelect = (action: string) => {
    if (action === "generate_brief") {
      onReadyForBrief();
      return;
    }
    const chipMessage = action.replace(/_/g, " ");
    sendMessage(`I'd like to ${chipMessage}`);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Initial AI message */}
        {initialMessage && conversationHistory.length === 0 && (
          <div className="flex gap-2">
            <div className="w-7 h-7 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
              <Sparkles className="h-3.5 w-3.5 text-primary-foreground" />
            </div>
            <div className="bg-muted rounded-sm rounded-tl-none p-3 max-w-[85%]">
              <p className="text-sm whitespace-pre-wrap">{initialMessage}</p>
            </div>
          </div>
        )}

        {conversationHistory.map((msg, idx) => (
          <div key={idx} className={`flex gap-2 ${msg.role === "user" ? "justify-end" : ""}`}>
            {msg.role === "assistant" && (
              <div className="w-7 h-7 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                <Sparkles className="h-3.5 w-3.5 text-primary-foreground" />
              </div>
            )}
            <div className={`p-3 max-w-[80%] text-sm whitespace-pre-wrap ${
              msg.role === "user"
                ? "bg-secondary rounded-sm rounded-tr-none"
                : "bg-muted rounded-sm rounded-tl-none"
            }`}>
              {msg.content}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-2">
            <div className="w-7 h-7 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
              <Sparkles className="h-3.5 w-3.5 text-primary-foreground" />
            </div>
            <div className="bg-muted rounded-sm rounded-tl-none p-3">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}

        {suggestedActions.length > 0 && !isLoading && (
          <QuickReplyChips actions={suggestedActions} onSelect={handleChipSelect} />
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t p-3 space-y-2">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage(input)}
            placeholder="Describe what you'd like to visualize..."
            className="flex-1 bg-muted border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            disabled={isLoading}
          />
          <Button size="sm" onClick={() => sendMessage(input)} disabled={!input.trim() || isLoading}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
        {focusedImageId && (
          <div className="flex gap-2 items-center">
            <span className="text-[10px] text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
              Focused on: {focusedImageLabel ?? focusedImageId}
            </span>
            <button onClick={onClearFocus} className="text-[10px] text-muted-foreground hover:text-foreground">
              × Clear
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/staging/DesignChat.tsx
git commit -m "feat: add DesignChat component

Right panel for the split-panel Design Session. Handles chat messages,
streaming AI responses, quick-reply chips, focused-image context badge,
and 'Ready for Brief' transition.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 17: Create PlantPaletteTable Component

**Files:**
- Create: `frontend/components/staging/PlantPaletteTable.tsx`

- [ ] **Step 1: Create the component**

```tsx
"use client"

import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { PlantEntry } from "@/services/stagingApi";

interface PlantPaletteTableProps {
  plants: PlantEntry[];
  onChange: (plants: PlantEntry[]) => void;
}

export function PlantPaletteTable({ plants, onChange }: PlantPaletteTableProps) {
  const updatePlant = (index: number, field: keyof PlantEntry, value: string | number) => {
    const updated = [...plants];
    updated[index] = { ...updated[index], [field]: value };
    onChange(updated);
  };

  const addPlant = () => {
    onChange([...plants, { species: "", quantity: 1, size: "", placement: "" }]);
  };

  const removePlant = (index: number) => {
    onChange(plants.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-[1fr_120px_80px_1fr_1fr_40px] gap-2 text-xs font-medium text-muted-foreground">
        <div>Species</div>
        <div>Botanical Name</div>
        <div>Qty</div>
        <div>Size</div>
        <div>Placement</div>
        <div></div>
      </div>

      {plants.map((plant, idx) => (
        <div key={idx} className="grid grid-cols-[1fr_120px_80px_1fr_1fr_40px] gap-2">
          <Input
            value={plant.species}
            onChange={(e) => updatePlant(idx, "species", e.target.value)}
            placeholder="Species name"
            className="text-sm h-8"
          />
          <Input
            value={plant.botanical_name ?? ""}
            onChange={(e) => updatePlant(idx, "botanical_name", e.target.value)}
            placeholder="Latin name"
            className="text-sm h-8"
          />
          <Input
            type="number"
            value={plant.quantity}
            onChange={(e) => updatePlant(idx, "quantity", parseInt(e.target.value) || 1)}
            min={1}
            className="text-sm h-8"
          />
          <Input
            value={plant.size}
            onChange={(e) => updatePlant(idx, "size", e.target.value)}
            placeholder="e.g. 8-10 ft"
            className="text-sm h-8"
          />
          <Input
            value={plant.placement}
            onChange={(e) => updatePlant(idx, "placement", e.target.value)}
            placeholder="e.g. back row"
            className="text-sm h-8"
          />
          <Button size="sm" variant="ghost" onClick={() => removePlant(idx)} className="h-8 w-8 p-0">
            <Trash2 className="h-3.5 w-3.5 text-destructive" />
          </Button>
        </div>
      ))}

      <Button size="sm" variant="outline" onClick={addPlant} className="w-full">
        <Plus className="h-3.5 w-3.5 mr-1" /> Add Plant
      </Button>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/staging/PlantPaletteTable.tsx
git commit -m "feat: add PlantPaletteTable component

Editable table for the Design Brief editor. Supports inline editing
of species, botanical name, quantity, size, and placement. Add/remove
rows with immediate state updates.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 18: Create DesignBriefEditor Component

**Files:**
- Create: `frontend/components/staging/DesignBriefEditor.tsx`

- [ ] **Step 1: Create the component**

```tsx
"use client"

import { useState } from "react";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";
import { PlantPaletteTable } from "./PlantPaletteTable";
import type { DesignBrief, PlantEntry } from "@/services/stagingApi";

interface DesignBriefEditorProps {
  brief: DesignBrief;
  onChange: (brief: DesignBrief) => void;
  imageLabels: Record<string, string>;
}

export function DesignBriefEditor({ brief, onChange, imageLabels }: DesignBriefEditorProps) {
  const [newPreserve, setNewPreserve] = useState("");

  const updateField = <K extends keyof DesignBrief>(field: K, value: DesignBrief[K]) => {
    onChange({ ...brief, [field]: value });
  };

  const addPreserveElement = () => {
    if (!newPreserve.trim()) return;
    updateField("preserve_elements", [...brief.preserve_elements, newPreserve.trim()]);
    setNewPreserve("");
  };

  const removePreserveElement = (index: number) => {
    updateField("preserve_elements", brief.preserve_elements.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Global Instructions */}
      <div className="space-y-2">
        <Label className="text-sm font-semibold">Global Instructions</Label>
        <Textarea
          value={brief.global_instructions}
          onChange={(e) => updateField("global_instructions", e.target.value)}
          rows={3}
          className="text-sm resize-none"
          placeholder="Overall styling direction..."
        />
      </div>

      {/* Plant Palette */}
      <div className="space-y-2">
        <Label className="text-sm font-semibold">Plant Palette</Label>
        <PlantPaletteTable
          plants={brief.plant_palette}
          onChange={(plants: PlantEntry[]) => updateField("plant_palette", plants)}
        />
      </div>

      {/* Placement Guide */}
      <div className="space-y-2">
        <Label className="text-sm font-semibold">Placement Guide</Label>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Back Row (tall)</Label>
            <Input
              value={brief.placement_guide.back_row}
              onChange={(e) => updateField("placement_guide", { ...brief.placement_guide, back_row: e.target.value })}
              className="text-sm h-8"
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Middle Row (mid-height)</Label>
            <Input
              value={brief.placement_guide.middle_row ?? ""}
              onChange={(e) => updateField("placement_guide", { ...brief.placement_guide, middle_row: e.target.value || undefined })}
              className="text-sm h-8"
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Front Row (low)</Label>
            <Input
              value={brief.placement_guide.front_row ?? ""}
              onChange={(e) => updateField("placement_guide", { ...brief.placement_guide, front_row: e.target.value || undefined })}
              className="text-sm h-8"
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Accent Areas</Label>
            <Input
              value={brief.placement_guide.accent_areas ?? ""}
              onChange={(e) => updateField("placement_guide", { ...brief.placement_guide, accent_areas: e.target.value || undefined })}
              className="text-sm h-8"
            />
          </div>
        </div>
      </div>

      {/* Preserve Elements */}
      <div className="space-y-2">
        <Label className="text-sm font-semibold">Preserve (don't change)</Label>
        <div className="flex flex-wrap gap-1.5">
          {brief.preserve_elements.map((el, idx) => (
            <Badge key={idx} variant="secondary" className="text-xs gap-1">
              {el}
              <button onClick={() => removePreserveElement(idx)}>
                <X className="h-2.5 w-2.5" />
              </button>
            </Badge>
          ))}
        </div>
        <div className="flex gap-2">
          <Input
            value={newPreserve}
            onChange={(e) => setNewPreserve(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addPreserveElement()}
            placeholder="e.g. existing patio"
            className="text-sm h-8"
          />
          <Button size="sm" variant="outline" onClick={addPreserveElement} className="h-8">Add</Button>
        </div>
      </div>

      {/* Settings */}
      <div className="space-y-2">
        <Label className="text-sm font-semibold">Generation Settings</Label>
        <div className="grid grid-cols-4 gap-3">
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Variations per image</Label>
            <Input
              type="number"
              value={brief.settings.variations_per_room}
              onChange={(e) => updateField("settings", { ...brief.settings, variations_per_room: parseInt(e.target.value) || 5 })}
              min={1} max={10}
              className="text-sm h-8"
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Model</Label>
            <Input value={brief.settings.model} disabled className="text-sm h-8" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Quality</Label>
            <Input value={brief.settings.quality} disabled className="text-sm h-8" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Size</Label>
            <Input value={brief.settings.size} disabled className="text-sm h-8" />
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/staging/DesignBriefEditor.tsx
git commit -m "feat: add DesignBriefEditor component

Full structured form editor for the Design Brief (Step 4). Includes
global instructions, plant palette table, placement guide, preserve
elements tag input, and generation settings.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 19: Create GenerationSummary Component

**Files:**
- Create: `frontend/components/staging/GenerationSummary.tsx`

- [ ] **Step 1: Create the component**

```tsx
"use client"

import { Badge } from "@/components/ui/badge";
import type { DesignBrief } from "@/services/stagingApi";

interface GenerationSummaryProps {
  projectName: string;
  imageCount: number;
  brief: DesignBrief;
}

export function GenerationSummary({ projectName, imageCount, brief }: GenerationSummaryProps) {
  const totalVariations = imageCount * brief.settings.variations_per_room;

  return (
    <div className="space-y-6 max-w-2xl">
      <div className="space-y-4">
        <div>
          <span className="text-sm font-medium text-muted-foreground">Project</span>
          <p className="text-lg font-semibold">{projectName}</p>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="p-4 bg-muted/50 rounded-lg text-center">
            <div className="text-2xl font-bold">{imageCount}</div>
            <div className="text-xs text-muted-foreground">Images</div>
          </div>
          <div className="p-4 bg-muted/50 rounded-lg text-center">
            <div className="text-2xl font-bold">{brief.settings.variations_per_room}</div>
            <div className="text-xs text-muted-foreground">Per Image</div>
          </div>
          <div className="p-4 bg-muted/50 rounded-lg text-center">
            <div className="text-2xl font-bold">{totalVariations}</div>
            <div className="text-xs text-muted-foreground">Total Variations</div>
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <span className="text-sm font-medium text-muted-foreground">Design Direction</span>
        <p className="text-sm leading-relaxed">{brief.global_instructions}</p>
      </div>

      {brief.plant_palette.length > 0 && (
        <div className="space-y-2">
          <span className="text-sm font-medium text-muted-foreground">Plants ({brief.plant_palette.length})</span>
          <div className="flex flex-wrap gap-1.5">
            {brief.plant_palette.map((p, i) => (
              <Badge key={i} variant="secondary" className="text-xs">
                {p.quantity}× {p.species}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {brief.preserve_elements.length > 0 && (
        <div className="space-y-2">
          <span className="text-sm font-medium text-muted-foreground">Preserving</span>
          <div className="flex flex-wrap gap-1.5">
            {brief.preserve_elements.map((el, i) => (
              <Badge key={i} variant="outline" className="text-xs">{el}</Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/staging/GenerationSummary.tsx
git commit -m "feat: add GenerationSummary component

Step 5 review card showing project name, image/variation counts,
design direction summary, plant palette badges, and preserve elements.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 20: Redesign NewProjectWizard to 5-Step Flow

**Files:**
- Modify: `frontend/components/staging/NewProjectWizard.tsx`

This task rewrites the wizard with a draft-project lifecycle: the project is created after Step 2 (Upload) so that Steps 3-5 can call project-scoped API endpoints.

- [ ] **Step 1: Rewrite NewProjectWizard.tsx**

Replace the entire file content with:

```tsx
"use client"

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Upload, X, ChevronRight, ChevronLeft, Loader2 } from "lucide-react";
import {
  createProject, uploadRooms, analyzeImages, generateBrief, updateBrief,
  streamGeneration, ChatMessage, DesignBrief, ImageAnalysisResult, StagingProject,
} from "@/services/stagingApi";
import { ImageGalleryPanel } from "./ImageGalleryPanel";
import { DesignChat } from "./DesignChat";
import { DesignBriefEditor } from "./DesignBriefEditor";
import { GenerationSummary } from "./GenerationSummary";
import { toast } from "sonner";

interface NewProjectWizardProps {
  onComplete: (project: StagingProject) => void;
  onCancel: () => void;
}

interface RoomFile {
  file: File;
  name: string;
  preview: string;
}

const STEPS = [
  { number: 1, title: "Name", description: "Choose a name for your project" },
  { number: 2, title: "Upload", description: "Upload baseline photos" },
  { number: 3, title: "AI Design Session", description: "Describe your vision" },
  { number: 4, title: "Design Brief", description: "Review and edit the plan" },
  { number: 5, title: "Generate", description: "Review and launch" },
];

export function NewProjectWizard({ onComplete, onCancel }: NewProjectWizardProps) {
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);

  // Step 1 state
  const [projectName, setProjectName] = useState("");

  // Step 2 state
  const [roomFiles, setRoomFiles] = useState<RoomFile[]>([]);

  // Draft project (created after Step 2)
  const [projectId, setProjectId] = useState<string | null>(null);
  const [uploadedRooms, setUploadedRooms] = useState<Array<{ id: string; label: string; url: string }>>([]);

  // Step 3 state
  const [analyses, setAnalyses] = useState<ImageAnalysisResult[]>([]);
  const [conversationHistory, setConversationHistory] = useState<ChatMessage[]>([]);
  const [focusedImageId, setFocusedImageId] = useState<string | null>(null);
  const [initialAiMessage, setInitialAiMessage] = useState("");

  // Step 4 state
  const [designBrief, setDesignBrief] = useState<DesignBrief | null>(null);

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    const newRoomFiles = files.map(file => ({
      file,
      name: file.name.replace(/\.[^/.]+$/, ""),
      preview: URL.createObjectURL(file),
    }));
    setRoomFiles(prev => [...prev, ...newRoomFiles]);
  }, []);

  const removeFile = (index: number) => {
    setRoomFiles(prev => {
      const updated = [...prev];
      URL.revokeObjectURL(updated[index].preview);
      updated.splice(index, 1);
      return updated;
    });
  };

  const updateRoomName = (index: number, name: string) => {
    setRoomFiles(prev => {
      const updated = [...prev];
      updated[index] = { ...updated[index], name };
      return updated;
    });
  };

  const canProceed = (step: number) => {
    switch (step) {
      case 1: return projectName.trim().length > 0;
      case 2: return roomFiles.length > 0;
      case 3: return conversationHistory.length >= 2; // At least one exchange
      case 4: return designBrief !== null;
      case 5: return true;
      default: return false;
    }
  };

  // Transition from Step 2 → 3: create draft project, upload rooms, analyze
  const transitionToDesignSession = async () => {
    setIsLoading(true);
    try {
      // 1. Create draft project (prompt is placeholder — will be replaced by brief)
      const project = await createProject({
        name: projectName,
        prompt: "Draft — pending AI Design Session",
      });
      setProjectId(project.id);

      // 2. Upload rooms
      const roomData = roomFiles.map(rf => ({ file: rf.file, name: rf.name }));
      await uploadRooms(project.id, roomData);
      toast.success("Photos uploaded");

      // 3. Analyze images
      const analysisResults = await analyzeImages(project.id);
      setAnalyses(analysisResults);

      // 4. Build initial AI message from analyses
      const featureSummary = analysisResults
        .map(a => `• **${a.room_id}**: ${a.description}`)
        .join("\n");
      setInitialAiMessage(
        `I've analyzed your ${analysisResults.length} photos. Here's what I see:\n\n${featureSummary}\n\nWhat would you like to visualize in these spaces?`
      );

      // 5. Store uploaded room info for the gallery panel
      setUploadedRooms(analysisResults.map((a, i) => ({
        id: a.room_id,
        label: roomFiles[i]?.name ?? `Room ${i + 1}`,
        url: roomFiles[i]?.preview ?? "",
      })));

      setCurrentStep(3);
    } catch (error) {
      console.error("Failed to set up design session:", error);
      toast.error(error instanceof Error ? error.message : "Setup failed");
    } finally {
      setIsLoading(false);
    }
  };

  // Transition from Step 3 → 4: generate brief
  const transitionToBriefEditor = async () => {
    if (!projectId) return;
    setIsLoading(true);
    try {
      const brief = await generateBrief(projectId, conversationHistory);
      setDesignBrief(brief);
      setCurrentStep(4);
    } catch (error) {
      console.error("Failed to generate brief:", error);
      toast.error("Failed to generate Design Brief");
    } finally {
      setIsLoading(false);
    }
  };

  // Transition from Step 4 → 5: save brief
  const transitionToGenerate = async () => {
    if (!projectId || !designBrief) return;
    setIsLoading(true);
    try {
      await updateBrief(projectId, designBrief);
      setCurrentStep(5);
    } catch (error) {
      console.error("Failed to save brief:", error);
      toast.error("Failed to save Design Brief");
    } finally {
      setIsLoading(false);
    }
  };

  // Step 5: launch generation
  const handleGenerate = async () => {
    if (!projectId) return;
    toast.success("Generation started! Redirecting to project...");
    onComplete({ id: projectId, name: projectName } as StagingProject);
  };

  const nextStep = () => {
    if (!canProceed(currentStep)) return;
    if (currentStep === 2) {
      transitionToDesignSession();
      return;
    }
    if (currentStep === 4) {
      transitionToGenerate();
      return;
    }
    setCurrentStep(prev => Math.min(5, prev + 1));
  };

  const prevStep = () => setCurrentStep(prev => Math.max(1, prev - 1));

  const focusedLabel = focusedImageId
    ? uploadedRooms.find(r => r.id === focusedImageId)?.label ?? null
    : null;

  const imageLabels = Object.fromEntries(uploadedRooms.map(r => [r.id, r.label]));

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="project-name">Project Name</Label>
              <Input
                id="project-name"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                placeholder="e.g., Backyard Fence Line — Spring 2026"
                className="text-base"
              />
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-4">
            <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center">
              <input type="file" id="room-upload" multiple accept="image/*" onChange={handleFileChange} className="hidden" />
              <label htmlFor="room-upload" className="cursor-pointer flex flex-col items-center gap-2">
                <Upload className="h-8 w-8 text-muted-foreground" />
                <div className="text-sm"><span className="font-medium">Click to upload</span> or drag and drop</div>
                <div className="text-xs text-muted-foreground">PNG, JPG, JPEG — no limit on images</div>
              </label>
            </div>
            {roomFiles.length > 0 && (
              <div className="space-y-3">
                <Label>Uploaded Photos ({roomFiles.length})</Label>
                <div className="grid grid-cols-3 gap-3">
                  {roomFiles.map((rf, index) => (
                    <div key={index} className="space-y-2">
                      <div className="relative aspect-video">
                        <img src={rf.preview} alt={rf.name} className="w-full h-full object-cover rounded-md border" />
                        <Button size="sm" variant="destructive" className="absolute -top-2 -right-2 h-6 w-6 rounded-full p-0" onClick={() => removeFile(index)}>
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                      <Input value={rf.name} onChange={(e) => updateRoomName(index, e.target.value)} placeholder="Label" className="text-sm" />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 3:
        return (
          <div className="flex gap-0 -mx-6 -mb-6 h-[500px] border-t">
            {/* Left: Image Gallery */}
            <div className="w-[40%] border-r bg-muted/30 hidden md:block">
              <ImageGalleryPanel
                images={uploadedRooms}
                analyses={analyses}
                focusedImageId={focusedImageId}
                onFocusImage={setFocusedImageId}
                perImageNotes={designBrief?.per_image_notes ?? {}}
              />
            </div>
            {/* Right: Chat */}
            <div className="flex-1">
              <DesignChat
                projectId={projectId!}
                focusedImageId={focusedImageId}
                focusedImageLabel={focusedLabel}
                onClearFocus={() => setFocusedImageId(null)}
                onReadyForBrief={transitionToBriefEditor}
                initialMessage={initialAiMessage}
                conversationHistory={conversationHistory}
                onHistoryUpdate={setConversationHistory}
              />
            </div>
          </div>
        );

      case 4:
        return designBrief ? (
          <DesignBriefEditor brief={designBrief} onChange={setDesignBrief} imageLabels={imageLabels} />
        ) : (
          <div className="flex items-center justify-center h-32">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        );

      case 5:
        return designBrief ? (
          <GenerationSummary projectName={projectName} imageCount={roomFiles.length} brief={designBrief} />
        ) : null;

      default:
        return null;
    }
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>New Project</CardTitle>
            <p className="text-sm text-muted-foreground mt-1">{STEPS[currentStep - 1]?.description}</p>
          </div>
          <Badge variant="outline" className="text-xs">Step {currentStep} of 5</Badge>
        </div>
        <div className="flex items-center gap-2 pt-4">
          {STEPS.map((step) => (
            <div key={step.number} className="flex items-center gap-2">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium ${
                currentStep >= step.number ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
              }`}>{step.number}</div>
              {step.number < 5 && <div className={`w-6 h-px ${currentStep > step.number ? "bg-primary" : "bg-muted"}`} />}
            </div>
          ))}
        </div>
      </CardHeader>

      <CardContent>{renderStep()}</CardContent>

      <CardFooter className="flex items-center justify-between">
        <div className="flex gap-2">
          <Button variant="outline" onClick={onCancel}>Cancel</Button>
          {currentStep > 1 && currentStep !== 3 && (
            <Button variant="ghost" onClick={prevStep} disabled={isLoading}>
              <ChevronLeft className="h-4 w-4 mr-1" /> Back
            </Button>
          )}
        </div>
        <div className="flex gap-2">
          {currentStep === 5 ? (
            <Button onClick={handleGenerate} disabled={isLoading} className="min-w-[140px]">
              {isLoading ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Generating...</> : "Generate Project"}
            </Button>
          ) : currentStep === 3 ? null : ( // Step 3 transitions via chat "Ready for Brief"
            <Button onClick={nextStep} disabled={!canProceed(currentStep) || isLoading}>
              {isLoading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : null}
              {currentStep === 2 ? "Upload & Analyze" : currentStep === 4 ? "Save & Continue" : "Next"}
              {!isLoading && <ChevronRight className="h-4 w-4 ml-1" />}
            </Button>
          )}
        </div>
      </CardFooter>
    </Card>
  );
}
```

Key design decisions in this implementation:
- **Draft project created after Step 2** — `transitionToDesignSession()` creates the project, uploads rooms, and analyzes images in sequence
- **Step 3 has no "Next" button** — transition happens via the "Generate Design Brief" chip in the chat
- **Step 4 "Save & Continue"** calls `updateBrief()` then advances
- **Step 5 "Generate Project"** redirects to the project detail page where streaming begins
- **Mobile**: Step 3 hides the image gallery panel below `md:` breakpoint (the DesignChat fills full width)

- [ ] **Step 2: Verify the frontend builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 3: Run frontend lint**

Run: `cd frontend && npx next lint`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add frontend/components/staging/NewProjectWizard.tsx
git commit -m "feat: redesign NewProjectWizard as 5-step flow

Replaces the 4-step wizard with a 5-step flow:
1. Name, 2. Upload (bug-fixed), 3. AI Design Session (split panel),
4. Design Brief Editor, 5. Generation Summary + launch.

Draft project created after Step 2 to enable project-scoped API calls.
Step 3 transitions via chat 'Generate Brief' action.
Integrates all new staging components.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 8: Frontend E2E Tests

### Task 21: Add Playwright E2E Tests for AI Design Session

**Files:**
- Create: `frontend/tests/e2e/ai-design-session.spec.ts`

- [ ] **Step 1: Create the E2E test file**

```typescript
import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const STAGING_DIR = join(__dirname, '..', '..', 'components', 'staging');

test('NewProjectWizard has 5 steps', async () => {
  const source = readFileSync(join(STAGING_DIR, 'NewProjectWizard.tsx'), 'utf-8');
  // Should reference all 5 steps
  expect(source).toContain('AI Design Session');
  expect(source).toContain('Design Brief');
  expect(source).toContain('Generate');
  // Should have 5 step definitions
  const stepMatches = source.match(/number:\s*\d/g);
  expect(stepMatches?.length).toBeGreaterThanOrEqual(5);
});

test('ImageGalleryPanel groups images by feature', async () => {
  const source = readFileSync(join(STAGING_DIR, 'ImageGalleryPanel.tsx'), 'utf-8');
  expect(source).toContain('groupImages');
  expect(source).toContain('focusedImageId');
  expect(source).toContain('onFocusImage');
});

test('DesignChat supports focused image context', async () => {
  const source = readFileSync(join(STAGING_DIR, 'DesignChat.tsx'), 'utf-8');
  expect(source).toContain('focusedImageId');
  expect(source).toContain('chatWithProject');
  expect(source).toContain('onReadyForBrief');
});

test('DesignBriefEditor has plant palette and placement guide', async () => {
  const source = readFileSync(join(STAGING_DIR, 'DesignBriefEditor.tsx'), 'utf-8');
  expect(source).toContain('PlantPaletteTable');
  expect(source).toContain('placement_guide');
  expect(source).toContain('preserve_elements');
  expect(source).toContain('global_instructions');
});

test('QuickReplyChips maps action keys to labels', async () => {
  const source = readFileSync(join(STAGING_DIR, 'QuickReplyChips.tsx'), 'utf-8');
  expect(source).toContain('specify_species');
  expect(source).toContain('generate_brief');
  expect(source).toContain('ACTION_LABELS');
});

test('stagingApi includes new Design Session endpoints', async () => {
  const source = readFileSync(
    join(__dirname, '..', '..', 'services', 'stagingApi.ts'),
    'utf-8',
  );
  expect(source).toContain('analyzeImages');
  expect(source).toContain('chatWithProject');
  expect(source).toContain('generateBrief');
  expect(source).toContain('updateBrief');
  // Bug fixes should be applied
  expect(source).not.toContain("'room_files'");
  expect(source).not.toContain('/generate/stream');
});
```

- [ ] **Step 2: Run E2E tests**

Run: `cd frontend && npx playwright test ai-design-session.spec.ts`
Expected: ALL PASS (6 tests)

- [ ] **Step 3: Run all frontend tests**

Run: `cd frontend && npx playwright test`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add frontend/tests/e2e/ai-design-session.spec.ts
git commit -m "test: add Playwright E2E tests for AI Design Session

6 tests verifying: 5-step wizard structure, ImageGalleryPanel grouping,
DesignChat focused image support, DesignBriefEditor fields, QuickReplyChips
labels, and stagingApi new endpoints + bug fix verification.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Final Verification

### Task 22: Full Test Suite Run

- [ ] **Step 1: Run all backend tests**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/ --ignore=tests/integration -v`
Expected: ALL PASS (52+ tests)

- [ ] **Step 2: Run all frontend tests**

Run: `cd frontend && npx playwright test`
Expected: ALL PASS (25 tests)

- [ ] **Step 3: Build frontend**

Run: `cd frontend && npm run build`
Expected: Build succeeds

- [ ] **Step 4: Run frontend lint**

Run: `cd frontend && npx next lint`
Expected: No errors

- [ ] **Step 5: Final commit tag**

```bash
git tag -a v0.2.0-ai-design-session -m "AI Design Session with bug fixes

- Fix 3 frontend-backend API mismatches
- Add split-panel AI Design Session with image-aware conversation
- Add structured Design Brief editor
- Add 23 new tests including backyard landscaping scenario
- Redesign wizard from 4 to 5 steps"
```
