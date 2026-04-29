# Edit Existing Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an edit experience for existing virtual-staging projects so users can rename, change settings, edit the design brief (with a regeneration prompt), resume the design chat, manage rooms, and roll back to saved versions.

**Architecture:** The project page becomes a tabbed view (Gallery | Design Brief | Settings | History). New backend endpoints expose project/room mutations and an explicit version-snapshot history. Shared edit primitives (form, chat panel, settings form) get extracted into hooks + components and reused by both the create wizard and the edit page (Approach C: shared core + thin wrappers). Versions live in the same Cosmos container as projects with `doc_type="staging_project_version"` and a `project_id` property; the partition key remains `/id`.

**Tech Stack:** Python / FastAPI / pytest (backend), Next.js 14 / React / shadcn-ui / Tailwind / Sonner (frontend), Playwright (e2e).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `backend/models/staging.py` | Modify | Add `UpdateProjectRequest`, `UpdateRoomRequest`, `CreateVersionRequest`, `VersionSnapshot`, `StagingProjectVersion`; add `conversation_history: Optional[List[Dict[str, Any]]]` to `StagingProject`. |
| `backend/models/design_brief.py` | (no change) | No edits required — `staging.py` stores chat history as plain dicts to avoid the import cycle. |
| `backend/core/staging_storage.py` | Modify | Add `create_version`, `list_versions`, `get_version`, `delete_version`. |
| `backend/api/endpoints/staging.py` | Modify | Add 7 endpoints (PATCH project, DELETE room, PATCH room, POST/GET/DELETE version, POST revert) + persist `conversation_history` from chat. |
| `frontend/services/stagingApi.ts` | Modify | Add update/room/version client functions and types; add `conversation_history` to `StagingProject`. |
| `frontend/hooks/staging/useBriefEditor.ts` | Create | Brief draft state + dirty + save logic. |
| `frontend/hooks/staging/useDesignChat.ts` | Create | Chat send / typing / history state. |
| `frontend/hooks/staging/useProjectSettings.ts` | Create | Settings form state + save. |
| `frontend/hooks/staging/useProjectVersions.ts` | Create | Versions list + create / revert / delete. |
| `frontend/components/staging/DesignChatPanel.tsx` | Create | Pure chat-UI (extracted from `DesignChat`). |
| `frontend/components/staging/DesignChat.tsx` | Modify | Becomes thin wrapper using `DesignChatPanel` + proceed-intent detection. |
| `frontend/components/staging/DesignBriefEditor.tsx` | Modify | Add `disabled?: boolean` prop; thread through inputs/buttons. |
| `frontend/components/staging/GenerationSettingsForm.tsx` | Create | Editable settings form for the Settings tab. |
| `frontend/components/staging/edit/ProjectTabs.tsx` | Create | URL-synced tab shell (`?tab=`). |
| `frontend/components/staging/edit/EditableProjectName.tsx` | Create | Inline-editable project title. |
| `frontend/components/staging/edit/RegeneratePrompt.tsx` | Create | Banner+dialog asking to regenerate after brief edits. |
| `frontend/components/staging/edit/ProjectGalleryTab.tsx` | Create | Wraps existing gallery body. |
| `frontend/components/staging/edit/ProjectBriefTab.tsx` | Create | Brief editor + chat panel + regenerate. |
| `frontend/components/staging/edit/ProjectSettingsTab.tsx` | Create | Settings form + add/remove room. |
| `frontend/components/staging/edit/ProjectHistoryTab.tsx` | Create | Versions list + Save/Revert/Delete. |
| `frontend/app/projects/[id]/page.tsx` | Modify | Renders tab shell, wires hooks, replaces title with `EditableProjectName`. |
| `frontend/components/staging/NewProjectWizard.tsx` | Modify | Reuse new hooks/components (no behavior change). |
| `tests/test_staging_models.py` | Create | Pydantic tests for new request/snapshot/version models. |
| `tests/test_staging_storage_versions.py` | Create | Cosmos storage tests for version CRUD. |
| `tests/test_staging_api.py` | Modify | Add tests for new endpoints + chat persistence. |
| `frontend/tests/e2e/edit-project.spec.ts` | Create | Playwright e2e for the edit flow. |

---

## Phase 1 — Backend Models

### Task 1: Add edit, room, and version request/response models

**Files:**
- Modify: `backend/models/staging.py`
- Test: `tests/test_staging_models.py` (Create)

- [ ] **Step 1.1: Write the failing test**

```python
# tests/test_staging_models.py
"""Tests for new staging request/version models."""
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.models.staging import (
    CreateVersionRequest,
    StagingProject,
    StagingProjectVersion,
    StagingSettings,
    UpdateProjectRequest,
    UpdateRoomRequest,
    VersionSnapshot,
)


def test_update_project_request_all_fields_optional():
    req = UpdateProjectRequest()
    assert req.name is None
    assert req.prompt is None
    assert req.settings is None
    assert req.design_brief is None
    assert req.conversation_history is None


def test_update_project_request_accepts_partial():
    req = UpdateProjectRequest(name="Renamed")
    assert req.name == "Renamed"
    assert req.prompt is None


def test_update_room_request_requires_label():
    with pytest.raises(ValidationError):
        UpdateRoomRequest()
    req = UpdateRoomRequest(label="Front Yard")
    assert req.label == "Front Yard"


def test_create_version_request_defaults():
    req = CreateVersionRequest()
    assert req.label is None
    assert req.note is None


def test_version_snapshot_round_trip():
    snap = VersionSnapshot(
        name="My Project",
        prompt="Modern",
        settings=StagingSettings(),
        design_brief={"global_instructions": "x"},
        room_labels={"room-1": "Living Room"},
        conversation_history=[],
    )
    data = snap.dict()
    assert data["room_labels"] == {"room-1": "Living Room"}


def test_staging_project_version_model_round_trip():
    snap = VersionSnapshot(
        name="My Project",
        prompt="Modern",
        settings=StagingSettings(),
    )
    v = StagingProjectVersion(
        id="ver-1",
        project_id="proj-1",
        snapshot=snap,
        label="Before refresh",
        note="initial",
    )
    assert v.project_id == "proj-1"
    assert v.snapshot.name == "My Project"


def test_staging_project_includes_conversation_history():
    p = StagingProject(id="p1", name="N", prompt="P")
    assert p.conversation_history is None
    p2 = StagingProject(
        id="p2", name="N", prompt="P",
        conversation_history=[{"role": "user", "content": "hi"}],
    )
    assert p2.conversation_history is not None
    assert p2.conversation_history[0]["role"] == "user"
```

> **Note on dict shape:** `conversation_history` is stored as a list of plain dicts to avoid the staging↔design_brief import cycle. The above test reflects that — the assertion uses `["role"]` (dict access), not `.role` (attribute access).

- [ ] **Step 1.2: Run test to verify it fails**

```
uv run pytest tests/test_staging_models.py -v
```
Expected: ImportError / AttributeError — new models don't exist yet.

- [ ] **Step 1.3: Add models in `backend/models/staging.py`**

Replace the `StagingProject` block (lines 63-74) with:

```python
class StagingProject(BaseModel):
    id: str
    name: str
    prompt: str
    status: str = Field(ProjectStatus.UPLOADING, description="Overall project status")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    rooms: List[Room] = Field(default_factory=list)
    settings: StagingSettings = Field(default_factory=StagingSettings)
    folder_path: Optional[str] = None
    design_brief: Optional[Dict[str, Any]] = Field(None, description="Structured design brief from AI conversation")
    analyses: Optional[List[Dict[str, Any]]] = Field(None, description="Image analysis results")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        None, description="Persisted design-chat history (denormalized ChatMessage dicts)."
    )
```

Append to the end of `backend/models/staging.py`:

```python
class UpdateProjectRequest(BaseModel):
    name: Optional[str] = Field(None, description="New project name.")
    prompt: Optional[str] = Field(None, description="New overall styling prompt.")
    settings: Optional[StagingSettings] = Field(None, description="Updated generation settings.")
    design_brief: Optional[Dict[str, Any]] = Field(None, description="Replacement design brief (denormalized dict).")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(None, description="Replacement chat history (denormalized).")


class UpdateRoomRequest(BaseModel):
    label: str = Field(..., description="New room label.")


class CreateVersionRequest(BaseModel):
    label: Optional[str] = Field(None, description="Human-friendly version label.")
    note: Optional[str] = Field(None, description="Optional note describing the snapshot.")


class VersionSnapshot(BaseModel):
    name: str
    prompt: str
    settings: StagingSettings
    design_brief: Optional[Dict[str, Any]] = None
    room_labels: Dict[str, str] = Field(default_factory=dict, description="room_id -> label at snapshot time")
    conversation_history: Optional[List[Dict[str, Any]]] = None


class StagingProjectVersion(BaseModel):
    id: str
    project_id: str
    snapshot: VersionSnapshot
    label: Optional[str] = None
    note: Optional[str] = None
    created_at: Optional[datetime] = None
```

> **Note:** `conversation_history` is stored as `List[Dict[str, Any]]` rather than `List[ChatMessage]` to avoid an import cycle between `staging.py` and `design_brief.py`. The chat endpoint receives validated `ChatMessage` instances from `ChatRequest` and serializes them to dicts before persisting.

- [ ] **Step 1.4: Update the conversation_history test to match dict shape**

In the test you wrote in Step 1.1, change the final assertion in `test_staging_project_includes_conversation_history` to read the dict directly (not via attribute access):

```python
def test_staging_project_includes_conversation_history():
    p = StagingProject(id="p1", name="N", prompt="P")
    assert p.conversation_history is None
    p2 = StagingProject(
        id="p2", name="N", prompt="P",
        conversation_history=[{"role": "user", "content": "hi"}],
    )
    assert p2.conversation_history is not None
    assert p2.conversation_history[0]["role"] == "user"
```

- [ ] **Step 1.5: Run test to verify it passes**

```
uv run pytest tests/test_staging_models.py -v
```
Expected: 7 passed.

- [ ] **Step 1.6: Run the full backend test suite**

```
uv run pytest tests/ --ignore=tests/integration -v
```
Expected: All previously-passing tests still pass.

- [ ] **Step 1.7: Commit**

```
git add backend/models/staging.py tests/test_staging_models.py
git commit -m "feat(staging): add edit, room, and version request/snapshot models

Adds UpdateProjectRequest, UpdateRoomRequest, CreateVersionRequest,
VersionSnapshot, and StagingProjectVersion. Also adds an optional
conversation_history field to StagingProject so the design chat
can be resumed after the project is created.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 2 — Backend Storage

### Task 2: Add version CRUD to `StagingStorageService`

**Files:**
- Modify: `backend/core/staging_storage.py`
- Test: `tests/test_staging_storage_versions.py` (Create)

- [ ] **Step 2.1: Write the failing test**

```python
# tests/test_staging_storage_versions.py
"""Tests for project version storage."""
from unittest.mock import MagicMock

from azure.cosmos import exceptions

from backend.core.staging_storage import StagingStorageService


def _service():
    container = MagicMock()
    return StagingStorageService(container=container), container


def test_create_version_assigns_metadata():
    svc, container = _service()
    container.create_item.side_effect = lambda body: body

    body = {
        "project_id": "proj-1",
        "snapshot": {
            "name": "X",
            "prompt": "Y",
            "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
            "room_labels": {},
        },
        "label": "Before refresh",
    }
    created = svc.create_version(body)

    assert created["id"]
    assert created["project_id"] == "proj-1"
    assert created["doc_type"] == "staging_project_version"
    assert created["created_at"]
    container.create_item.assert_called_once()


def test_list_versions_filters_by_project_and_orders_desc():
    svc, container = _service()
    container.query_items.return_value = [
        {"id": "v2", "project_id": "p1", "doc_type": "staging_project_version", "created_at": "2026-04-29T01:00:00Z"},
        {"id": "v1", "project_id": "p1", "doc_type": "staging_project_version", "created_at": "2026-04-29T00:00:00Z"},
    ]

    versions = svc.list_versions(project_id="p1")

    assert [v["id"] for v in versions] == ["v2", "v1"]
    args, kwargs = container.query_items.call_args
    query = kwargs.get("query") or args[0]
    params = kwargs.get("parameters") or args[1]
    assert "c.doc_type = 'staging_project_version'" in query
    assert "c.project_id = @pid" in query
    assert "ORDER BY c.created_at DESC" in query
    assert {"name": "@pid", "value": "p1"} in params
    assert kwargs.get("enable_cross_partition_query") is True


def test_get_version_returns_doc_or_none():
    svc, container = _service()
    container.read_item.return_value = {"id": "v1", "doc_type": "staging_project_version", "project_id": "p1"}
    assert svc.get_version("v1")["id"] == "v1"

    container.read_item.side_effect = exceptions.CosmosResourceNotFoundError(status_code=404, message="x")
    assert svc.get_version("missing") is None


def test_delete_version_returns_bool():
    svc, container = _service()
    container.delete_item.return_value = None
    assert svc.delete_version("v1") is True

    container.delete_item.side_effect = exceptions.CosmosResourceNotFoundError(status_code=404, message="x")
    assert svc.delete_version("missing") is False
```

- [ ] **Step 2.2: Run test to verify it fails**

```
uv run pytest tests/test_staging_storage_versions.py -v
```
Expected: AttributeError — methods don't exist.

- [ ] **Step 2.3: Add version CRUD methods**

Append to `backend/core/staging_storage.py`:

```python
    # ----- Project version snapshots -----

    def create_version(self, version_data: Dict[str, Any]) -> Dict[str, Any]:
        if "id" not in version_data:
            version_data["id"] = str(uuid.uuid4())
        version_data["doc_type"] = "staging_project_version"
        version_data["created_at"] = datetime.now(timezone.utc).isoformat()
        return self.container.create_item(body=version_data)

    def list_versions(self, project_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        query = (
            "SELECT * FROM c "
            "WHERE c.doc_type = 'staging_project_version' AND c.project_id = @pid "
            "ORDER BY c.created_at DESC "
            "OFFSET @offset LIMIT @limit"
        )
        params = [
            {"name": "@pid", "value": project_id},
            {"name": "@offset", "value": offset},
            {"name": "@limit", "value": limit},
        ]
        return list(
            self.container.query_items(
                query=query,
                parameters=params,
                enable_cross_partition_query=True,
            )
        )

    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.container.read_item(item=version_id, partition_key=version_id)
        except exceptions.CosmosResourceNotFoundError:
            return None

    def delete_version(self, version_id: str) -> bool:
        try:
            self.container.delete_item(item=version_id, partition_key=version_id)
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False
```

- [ ] **Step 2.4: Run tests**

```
uv run pytest tests/test_staging_storage_versions.py -v
```
Expected: 4 passed.

- [ ] **Step 2.5: Run the full backend suite**

```
uv run pytest tests/ --ignore=tests/integration -v
```
Expected: All passing.

- [ ] **Step 2.6: Commit**

```
git add backend/core/staging_storage.py tests/test_staging_storage_versions.py
git commit -m "feat(staging): add project version snapshot CRUD

Stores versions in the same Cosmos container as projects, using
doc_type='staging_project_version' and a project_id property
(partition key remains /id). Supports create, list (newest-first),
get, and delete.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 3 — Backend Endpoints

### Task 3: PATCH /projects/{project_id} — update name, prompt, settings

**Files:**
- Modify: `backend/api/endpoints/staging.py`
- Modify: `tests/test_staging_api.py`

- [ ] **Step 3.1: Write the failing test**

Append to `tests/test_staging_api.py`:

```python
# --- Project edit (PATCH) tests ---

def _completed_project():
    return {
        "id": "proj-123",
        "name": "Original",
        "prompt": "Original prompt",
        "status": "completed",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "created_at": "2026-04-26T00:00:00Z",
        "updated_at": "2026-04-26T00:00:00Z",
        "doc_type": "staging_project",
    }


def test_patch_project_updates_name(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _completed_project()
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-123",
        json={"name": "Renamed"},
    )

    assert response.status_code == 200
    assert response.json()["project"]["name"] == "Renamed"


def test_patch_project_updates_prompt_and_settings(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _completed_project()
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-123",
        json={
            "prompt": "New direction",
            "settings": {"variations_per_room": 3, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        },
    )

    assert response.status_code == 200
    body = response.json()["project"]
    assert body["prompt"] == "New direction"
    assert body["settings"]["variations_per_room"] == 3


def test_patch_project_updates_design_brief_and_history(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _completed_project()
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-123",
        json={
            "design_brief": {
                "global_instructions": "Modern",
                "plant_palette": [],
                "placement_guide": {"back_row": ""},
                "per_image_notes": {},
                "preserve_elements": [],
                "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
            },
            "conversation_history": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 200
    body = response.json()["project"]
    assert body["design_brief"]["global_instructions"] == "Modern"
    assert body["conversation_history"][0]["content"] == "hello"


def test_patch_project_404(client, mock_staging_deps):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.side_effect = CosmosResourceNotFoundError(status_code=404, message="x")
    response = client.patch("/api/v1/staging/projects/missing", json={"name": "x"})
    assert response.status_code == 404


def test_patch_project_rejects_empty_payload(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _completed_project()
    response = client.patch("/api/v1/staging/projects/proj-123", json={})
    assert response.status_code == 400
```

- [ ] **Step 3.2: Verify the tests fail**

```
uv run pytest tests/test_staging_api.py::test_patch_project_updates_name -v
```
Expected: 405 Method Not Allowed (route doesn't exist).

- [ ] **Step 3.3: Add the endpoint**

In `backend/api/endpoints/staging.py`, add to the staging-models import block (line 21-30):

```python
from backend.models.staging import (
    CreateProjectRequest,
    CreateVersionRequest,
    ItemStatus,
    ProjectListResponse,
    ProjectResponse,
    Room,
    StagingProject,
    StagingProjectVersion,
    UpdateProjectRequest,
    UpdateRoomRequest,
    UploadRoomsResponse,
    Variation,
    VersionSnapshot,
)
```

Insert this endpoint immediately before the existing `@router.delete("/projects/{project_id}")` route:

```python
@router.patch("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    request: UpdateProjectRequest,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Update project name, prompt, and/or settings."""
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    payload = request.dict(exclude_unset=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No fields to update")

    updates: dict = {}
    if "name" in payload and payload["name"] is not None:
        updates["name"] = payload["name"]
    if "prompt" in payload and payload["prompt"] is not None:
        updates["prompt"] = payload["prompt"]
    if "settings" in payload and payload["settings"] is not None:
        updates["settings"] = payload["settings"]
    if "design_brief" in payload and payload["design_brief"] is not None:
        updates["design_brief"] = payload["design_brief"]
    if "conversation_history" in payload and payload["conversation_history"] is not None:
        updates["conversation_history"] = payload["conversation_history"]

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated = storage.update_project(project_id, updates)
    clean = {k: v for k, v in updated.items() if k != "doc_type" and not k.startswith("_")}
    return {"project": StagingProject(**clean)}
```

- [ ] **Step 3.4: Run the new tests**

```
uv run pytest tests/test_staging_api.py -k "patch_project" -v
```
Expected: 4 passed.

- [ ] **Step 3.5: Run full backend suite**

```
uv run pytest tests/ --ignore=tests/integration -v
```
Expected: all passing.

- [ ] **Step 3.6: Commit**

```
git add backend/api/endpoints/staging.py tests/test_staging_api.py
git commit -m "feat(staging): add PATCH /projects/{id} for name, prompt, settings

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: PATCH and DELETE /projects/{project_id}/rooms/{room_id}

**Files:**
- Modify: `backend/api/endpoints/staging.py`
- Modify: `tests/test_staging_api.py`

- [ ] **Step 4.1: Write the failing tests**

Append to `tests/test_staging_api.py`:

```python
# --- Room edit / delete tests ---

def _project_with_two_rooms():
    p = _completed_project()
    p["rooms"] = [
        {"id": "room-1", "label": "Living", "original_image_url": "https://x/img1.png", "status": "completed", "variations": []},
        {"id": "room-2", "label": "Kitchen", "original_image_url": "https://x/img2.png", "status": "completed", "variations": []},
    ]
    return p


def test_patch_room_updates_label(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_two_rooms()
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.patch(
        "/api/v1/staging/projects/proj-123/rooms/room-2",
        json={"label": "Den"},
    )

    assert response.status_code == 200
    rooms = response.json()["project"]["rooms"]
    labels = {r["id"]: r["label"] for r in rooms}
    assert labels["room-2"] == "Den"
    assert labels["room-1"] == "Living"


def test_patch_room_404_room_missing(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_two_rooms()
    response = client.patch(
        "/api/v1/staging/projects/proj-123/rooms/room-9",
        json={"label": "x"},
    )
    assert response.status_code == 404


def test_delete_room_removes_only_that_room(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_two_rooms()
    mock_container.replace_item.side_effect = lambda item, body: body

    response = client.delete("/api/v1/staging/projects/proj-123/rooms/room-2")

    assert response.status_code == 200
    rooms = response.json()["project"]["rooms"]
    assert [r["id"] for r in rooms] == ["room-1"]


def test_delete_last_room_returns_400(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    p = _project_with_two_rooms()
    p["rooms"] = p["rooms"][:1]
    mock_container.read_item.return_value = p

    response = client.delete("/api/v1/staging/projects/proj-123/rooms/room-1")
    assert response.status_code == 400
```

- [ ] **Step 4.2: Verify the tests fail**

```
uv run pytest tests/test_staging_api.py -k "patch_room or delete_room" -v
```
Expected: 405s.

- [ ] **Step 4.3: Implement the endpoints**

Add immediately after the PATCH project endpoint:

```python
@router.patch("/projects/{project_id}/rooms/{room_id}", response_model=ProjectResponse)
async def update_room(
    project_id: str,
    room_id: str,
    request: UpdateRoomRequest,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Rename a room."""
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    rooms = project_data.get("rooms", [])
    target = next((r for r in rooms if r.get("id") == room_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Room not found")

    target["label"] = request.label
    updated = storage.update_project(project_id, {"rooms": rooms})
    clean = {k: v for k, v in updated.items() if k != "doc_type" and not k.startswith("_")}
    return {"project": StagingProject(**clean)}


@router.delete("/projects/{project_id}/rooms/{room_id}", response_model=ProjectResponse)
async def delete_room(
    project_id: str,
    room_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Remove a room from the project."""
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    rooms = project_data.get("rooms", [])
    if not any(r.get("id") == room_id for r in rooms):
        raise HTTPException(status_code=404, detail="Room not found")
    if len(rooms) <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the last remaining room")

    new_rooms = [r for r in rooms if r.get("id") != room_id]
    updated = storage.update_project(project_id, {"rooms": new_rooms})
    clean = {k: v for k, v in updated.items() if k != "doc_type" and not k.startswith("_")}
    return {"project": StagingProject(**clean)}
```

- [ ] **Step 4.4: Run the room tests**

```
uv run pytest tests/test_staging_api.py -k "patch_room or delete_room" -v
```
Expected: 4 passed.

- [ ] **Step 4.5: Run full backend suite**

```
uv run pytest tests/ --ignore=tests/integration -v
```
Expected: all passing.

- [ ] **Step 4.6: Commit**

```
git add backend/api/endpoints/staging.py tests/test_staging_api.py
git commit -m "feat(staging): add PATCH and DELETE endpoints for rooms

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: POST and GET /projects/{project_id}/versions

**Files:**
- Modify: `backend/api/endpoints/staging.py`
- Modify: `tests/test_staging_api.py`

- [ ] **Step 5.1: Write the failing tests**

Append to `tests/test_staging_api.py`:

```python
# --- Version create / list tests ---

def test_create_version_snapshot_includes_room_labels(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _project_with_two_rooms()

    captured = {}
    def capture_create(body):
        captured.update(body)
        return body
    mock_container.create_item.side_effect = capture_create

    response = client.post(
        "/api/v1/staging/projects/proj-123/versions",
        json={"label": "Before refresh", "note": "v1"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["version"]["label"] == "Before refresh"
    assert body["version"]["snapshot"]["room_labels"] == {"room-1": "Living", "room-2": "Kitchen"}
    assert captured["doc_type"] == "staging_project_version"
    assert captured["project_id"] == "proj-123"


def test_create_version_404_for_missing_project(client, mock_staging_deps):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.side_effect = CosmosResourceNotFoundError(status_code=404, message="x")
    response = client.post("/api/v1/staging/projects/nope/versions", json={})
    assert response.status_code == 404


def test_list_versions_returns_descending(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    mock_container.read_item.return_value = _completed_project()
    mock_container.query_items.return_value = [
        {
            "id": "v2", "project_id": "proj-123", "doc_type": "staging_project_version",
            "snapshot": {"name": "X", "prompt": "Y", "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"}, "room_labels": {}},
            "created_at": "2026-04-29T01:00:00Z",
        },
        {
            "id": "v1", "project_id": "proj-123", "doc_type": "staging_project_version",
            "snapshot": {"name": "X", "prompt": "Y", "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"}, "room_labels": {}},
            "created_at": "2026-04-29T00:00:00Z",
        },
    ]

    response = client.get("/api/v1/staging/projects/proj-123/versions")
    assert response.status_code == 200
    assert [v["id"] for v in response.json()["versions"]] == ["v2", "v1"]
```

- [ ] **Step 5.2: Verify the tests fail**

```
uv run pytest tests/test_staging_api.py -k "version" -v
```
Expected: 405s.

- [ ] **Step 5.3: Add a small response model**

Append to `backend/models/staging.py`:

```python
class VersionResponse(BaseModel):
    version: StagingProjectVersion


class VersionListResponse(BaseModel):
    versions: List[StagingProjectVersion]
    total: int
```

- [ ] **Step 5.4: Implement the endpoints**

In `backend/api/endpoints/staging.py`, extend the staging-models import to also pull in `VersionListResponse` and `VersionResponse`:

```python
from backend.models.staging import (
    CreateProjectRequest,
    CreateVersionRequest,
    ItemStatus,
    ProjectListResponse,
    ProjectResponse,
    Room,
    StagingProject,
    StagingProjectVersion,
    UpdateProjectRequest,
    UpdateRoomRequest,
    UploadRoomsResponse,
    Variation,
    VersionListResponse,
    VersionResponse,
    VersionSnapshot,
)
```

Add a helper near the top of the module (after `get_staging_pipeline`):

```python
def _build_snapshot(project_data: dict) -> VersionSnapshot:
    """Convert a project document into a VersionSnapshot value."""
    rooms = project_data.get("rooms", [])
    room_labels = {r["id"]: r.get("label", "") for r in rooms}
    return VersionSnapshot(
        name=project_data.get("name", ""),
        prompt=project_data.get("prompt", ""),
        settings=project_data.get("settings") or {},
        design_brief=project_data.get("design_brief"),
        room_labels=room_labels,
        conversation_history=project_data.get("conversation_history") or None,
    )
```

Append the version endpoints below the room endpoints:

```python
@router.post("/projects/{project_id}/versions", response_model=VersionResponse, status_code=201)
async def create_version(
    project_id: str,
    request: CreateVersionRequest,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Save the current project state as a named version."""
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    snapshot = _build_snapshot(project_data)
    version_doc = {
        "project_id": project_id,
        "snapshot": snapshot.dict(),
        "label": request.label,
        "note": request.note,
    }
    created = storage.create_version(version_doc)
    clean = {k: v for k, v in created.items() if k != "doc_type" and not k.startswith("_")}
    return {"version": StagingProjectVersion(**clean)}


@router.get("/projects/{project_id}/versions", response_model=VersionListResponse)
async def list_versions(
    project_id: str,
    limit: int = 50,
    offset: int = 0,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """List saved versions for a project, newest first."""
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    docs = storage.list_versions(project_id=project_id, limit=limit, offset=offset)
    versions: list[StagingProjectVersion] = []
    for d in docs:
        clean = {k: v for k, v in d.items() if k != "doc_type" and not k.startswith("_")}
        versions.append(StagingProjectVersion(**clean))
    return {"versions": versions, "total": len(versions)}
```

- [ ] **Step 5.5: Run the version tests**

```
uv run pytest tests/test_staging_api.py -k "version" -v
```
Expected: 3 passed.

- [ ] **Step 5.6: Run full backend suite**

```
uv run pytest tests/ --ignore=tests/integration -v
```
Expected: all passing.

- [ ] **Step 5.7: Commit**

```
git add backend/models/staging.py backend/api/endpoints/staging.py tests/test_staging_api.py
git commit -m "feat(staging): add POST and GET endpoints for project versions

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: POST /projects/{project_id}/versions/{version_id}/revert

**Files:**
- Modify: `backend/api/endpoints/staging.py`
- Modify: `tests/test_staging_api.py`

- [ ] **Step 6.1: Write the failing tests**

Append to `tests/test_staging_api.py`:

```python
# --- Version revert tests ---

def _stored_version(version_id="v1", project_id="proj-123", label_for_rooms=None):
    label_for_rooms = label_for_rooms or {"room-1": "Snapshot Living"}
    return {
        "id": version_id,
        "project_id": project_id,
        "doc_type": "staging_project_version",
        "snapshot": {
            "name": "Snapshot Name",
            "prompt": "Snapshot prompt",
            "settings": {"variations_per_room": 4, "model": "gpt-image-2", "quality": "high", "size": "auto"},
            "design_brief": {"global_instructions": "from snapshot"},
            "room_labels": label_for_rooms,
            "conversation_history": [{"role": "user", "content": "hello", "focused_image_id": None, "timestamp": None}],
        },
        "label": "Before refresh",
        "note": None,
        "created_at": "2026-04-29T00:00:00Z",
    }


def test_revert_version_auto_snapshots_then_applies(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]

    project = _project_with_two_rooms()
    version = _stored_version(label_for_rooms={"room-1": "Snapshot Living", "room-2": "Snapshot Kitchen"})

    def fake_read(item, partition_key):
        if item == "proj-123":
            return project
        if item == "v1":
            return version
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        raise CosmosResourceNotFoundError(status_code=404, message="x")

    mock_container.read_item.side_effect = fake_read
    mock_container.replace_item.side_effect = lambda item, body: body

    created_versions: list[dict] = []
    def capture_create(body):
        created_versions.append(body)
        return body
    mock_container.create_item.side_effect = capture_create

    response = client.post("/api/v1/staging/projects/proj-123/versions/v1/revert")

    assert response.status_code == 200
    body = response.json()["project"]
    assert body["name"] == "Snapshot Name"
    assert body["prompt"] == "Snapshot prompt"
    assert body["settings"]["variations_per_room"] == 4
    labels = {r["id"]: r["label"] for r in body["rooms"]}
    assert labels == {"room-1": "Snapshot Living", "room-2": "Snapshot Kitchen"}
    # Ensure an auto "before revert" snapshot was created prior to applying.
    assert len(created_versions) == 1
    assert created_versions[0]["label"].startswith("Before revert")


def test_revert_does_not_resurrect_deleted_rooms(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]

    project = _project_with_two_rooms()
    project["rooms"] = project["rooms"][:1]
    version = _stored_version(label_for_rooms={"room-1": "X", "room-2": "Y"})

    def fake_read(item, partition_key):
        if item == "proj-123":
            return project
        if item == "v1":
            return version
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        raise CosmosResourceNotFoundError(status_code=404, message="x")
    mock_container.read_item.side_effect = fake_read
    mock_container.replace_item.side_effect = lambda item, body: body
    mock_container.create_item.side_effect = lambda body: body

    response = client.post("/api/v1/staging/projects/proj-123/versions/v1/revert")
    assert response.status_code == 200
    rooms = response.json()["project"]["rooms"]
    assert [r["id"] for r in rooms] == ["room-1"]
    assert rooms[0]["label"] == "X"


def test_revert_404_when_version_missing(client, mock_staging_deps):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    mock_container = mock_staging_deps["container"]

    def fake_read(item, partition_key):
        if item == "proj-123":
            return _completed_project()
        raise CosmosResourceNotFoundError(status_code=404, message="x")
    mock_container.read_item.side_effect = fake_read

    response = client.post("/api/v1/staging/projects/proj-123/versions/missing/revert")
    assert response.status_code == 404


def test_revert_404_when_version_belongs_to_different_project(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]
    project = _completed_project()
    version = _stored_version(project_id="other-proj")

    def fake_read(item, partition_key):
        if item == "proj-123":
            return project
        if item == "v1":
            return version
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        raise CosmosResourceNotFoundError(status_code=404, message="x")
    mock_container.read_item.side_effect = fake_read

    response = client.post("/api/v1/staging/projects/proj-123/versions/v1/revert")
    assert response.status_code == 404
```

- [ ] **Step 6.2: Verify the tests fail**

```
uv run pytest tests/test_staging_api.py -k "revert" -v
```
Expected: 405s / 404s.

- [ ] **Step 6.3: Implement the endpoint**

Add below the version-list endpoint:

```python
@router.post("/projects/{project_id}/versions/{version_id}/revert", response_model=ProjectResponse)
async def revert_version(
    project_id: str,
    version_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Apply a saved version's snapshot to the live project.

    Auto-creates a 'Before revert' snapshot of the current state first so the
    revert itself is reversible.
    """
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    version_doc = storage.get_version(version_id)
    if not version_doc or version_doc.get("project_id") != project_id:
        raise HTTPException(status_code=404, detail="Version not found")

    auto_snapshot = _build_snapshot(project_data)
    storage.create_version({
        "project_id": project_id,
        "snapshot": auto_snapshot.dict(),
        "label": f"Before revert to {version_doc.get('label') or version_id}",
        "note": "Auto-created before revert",
    })

    snap = version_doc.get("snapshot") or {}
    updates: dict = {
        "name": snap.get("name", project_data.get("name")),
        "prompt": snap.get("prompt", project_data.get("prompt")),
        "settings": snap.get("settings", project_data.get("settings")),
        "design_brief": snap.get("design_brief"),
        "conversation_history": snap.get("conversation_history"),
    }

    snapshot_labels = snap.get("room_labels") or {}
    rooms = project_data.get("rooms", [])
    for r in rooms:
        if r["id"] in snapshot_labels:
            r["label"] = snapshot_labels[r["id"]]
    updates["rooms"] = rooms

    updated = storage.update_project(project_id, updates)
    clean = {k: v for k, v in updated.items() if k != "doc_type" and not k.startswith("_")}
    return {"project": StagingProject(**clean)}
```

- [ ] **Step 6.4: Run the revert tests**

```
uv run pytest tests/test_staging_api.py -k "revert" -v
```
Expected: 4 passed.

- [ ] **Step 6.5: Run full backend suite**

```
uv run pytest tests/ --ignore=tests/integration -v
```
Expected: all passing.

- [ ] **Step 6.6: Commit**

```
git add backend/api/endpoints/staging.py tests/test_staging_api.py
git commit -m "feat(staging): add POST /versions/{id}/revert with auto pre-revert snapshot

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: DELETE /projects/{project_id}/versions/{version_id}

**Files:**
- Modify: `backend/api/endpoints/staging.py`
- Modify: `tests/test_staging_api.py`

- [ ] **Step 7.1: Write the failing tests**

```python
def test_delete_version_succeeds(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]

    def fake_read(item, partition_key):
        if item == "v1":
            return _stored_version()
        if item == "proj-123":
            return _completed_project()
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        raise CosmosResourceNotFoundError(status_code=404, message="x")
    mock_container.read_item.side_effect = fake_read
    mock_container.delete_item.return_value = None

    response = client.delete("/api/v1/staging/projects/proj-123/versions/v1")

    assert response.status_code == 200
    mock_container.delete_item.assert_called_once_with(item="v1", partition_key="v1")


def test_delete_version_404_when_missing(client, mock_staging_deps):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    mock_container = mock_staging_deps["container"]

    def fake_read(item, partition_key):
        if item == "proj-123":
            return _completed_project()
        raise CosmosResourceNotFoundError(status_code=404, message="x")
    mock_container.read_item.side_effect = fake_read

    response = client.delete("/api/v1/staging/projects/proj-123/versions/missing")
    assert response.status_code == 404


def test_delete_version_404_when_belongs_to_other_project(client, mock_staging_deps):
    mock_container = mock_staging_deps["container"]

    def fake_read(item, partition_key):
        if item == "proj-123":
            return _completed_project()
        if item == "v1":
            return _stored_version(project_id="other")
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        raise CosmosResourceNotFoundError(status_code=404, message="x")
    mock_container.read_item.side_effect = fake_read

    response = client.delete("/api/v1/staging/projects/proj-123/versions/v1")
    assert response.status_code == 404
```

- [ ] **Step 7.2: Verify the tests fail**

```
uv run pytest tests/test_staging_api.py -k "delete_version" -v
```
Expected: 405s.

- [ ] **Step 7.3: Implement the endpoint**

Add below the revert endpoint:

```python
@router.delete("/projects/{project_id}/versions/{version_id}")
async def delete_version(
    project_id: str,
    version_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    """Delete a saved version snapshot."""
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    version_doc = storage.get_version(version_id)
    if not version_doc or version_doc.get("project_id") != project_id:
        raise HTTPException(status_code=404, detail="Version not found")

    deleted = storage.delete_version(version_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"deleted": True, "id": version_id}
```

- [ ] **Step 7.4: Run the delete tests**

```
uv run pytest tests/test_staging_api.py -k "delete_version" -v
```
Expected: 3 passed.

- [ ] **Step 7.5: Run full backend suite**

```
uv run pytest tests/ --ignore=tests/integration -v
```
Expected: all passing.

- [ ] **Step 7.6: Commit**

```
git add backend/api/endpoints/staging.py tests/test_staging_api.py
git commit -m "feat(staging): add DELETE /versions/{id}

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 8: Persist conversation history on the chat endpoint

**Files:**
- Modify: `backend/api/endpoints/staging.py:539` (existing `chat_with_project`)
- Modify: `tests/test_staging_api.py`

- [ ] **Step 8.1: Write the failing test**

```python
# --- Chat persistence tests ---

def test_chat_persists_conversation_history(client, mock_staging_deps, monkeypatch):
    from backend.api.endpoints import staging as staging_module

    mock_container = mock_staging_deps["container"]
    p = _completed_project()
    p["analyses"] = []
    mock_container.read_item.return_value = p

    captured_updates: list[dict] = []
    def fake_replace(item, body):
        captured_updates.append(body)
        return body
    mock_container.replace_item.side_effect = fake_replace

    class _FakeService:
        def __init__(self, **kwargs):
            pass

        async def chat(self, message, conversation_history, focused_image_id):
            from backend.models.design_brief import ChatResponse
            return ChatResponse(reply="Hi there", ready_for_brief=False, suggested_actions=[])

    monkeypatch.setattr(staging_module, "DesignChatService", _FakeService, raising=False)
    # Ensure the lazy import inside the endpoint uses the patched class.
    import backend.core.design_chat as dc
    monkeypatch.setattr(dc, "DesignChatService", _FakeService, raising=False)

    response = client.post(
        "/api/v1/staging/projects/proj-123/chat",
        json={"message": "Hello", "conversation_history": []},
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "Hi there"
    assert any("conversation_history" in u for u in captured_updates)
    saved = next(u["conversation_history"] for u in captured_updates if "conversation_history" in u)
    roles = [m["role"] for m in saved]
    contents = [m["content"] for m in saved]
    assert roles == ["user", "assistant"]
    assert contents == ["Hello", "Hi there"]
```

- [ ] **Step 8.2: Verify the test fails**

```
uv run pytest tests/test_staging_api.py -k "chat_persists" -v
```
Expected: assertion failure (no `conversation_history` update is captured).

- [ ] **Step 8.3: Update the chat endpoint**

Replace the body of `chat_with_project` (around line 539) with:

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
    from backend.models.design_brief import ChatMessage

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

    user_msg = ChatMessage(
        role="user",
        content=request.message,
        focused_image_id=request.focused_image_id,
    )
    assistant_msg = ChatMessage(role="assistant", content=response.reply)
    full_history = [m.dict() for m in request.conversation_history] + [user_msg.dict(), assistant_msg.dict()]
    storage.update_project(project_id, {"conversation_history": full_history})

    return response
```

- [ ] **Step 8.4: Run the chat test**

```
uv run pytest tests/test_staging_api.py -k "chat_persists" -v
```
Expected: PASS.

- [ ] **Step 8.5: Run the full backend suite**

```
uv run pytest tests/ --ignore=tests/integration -v
```
Expected: all passing.

- [ ] **Step 8.6: Commit**

```
git add backend/api/endpoints/staging.py tests/test_staging_api.py
git commit -m "feat(staging): persist design chat history on the project doc

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 4 — Frontend Service Layer

### Task 9: Add update / room functions to `stagingApi`

**Files:**
- Modify: `frontend/services/stagingApi.ts`

- [ ] **Step 9.1: Update the `StagingProject` interface and request types**

Locate lines 68-79 (`StagingProject` interface) and replace with:

```ts
export interface StagingProject {
  id: string;
  name: string;
  prompt: string;
  status: 'uploading' | 'pending' | 'processing' | 'completed' | 'failed';
  settings: StagingSettings;
  rooms: Room[];
  created_at?: string;
  updated_at?: string;
  total_variations?: number;
  completed_variations?: number;
  design_brief?: DesignBrief | null;
  conversation_history?: ChatMessage[] | null;
}
```

Add immediately after `CreateProjectRequest` (around line 90):

```ts
export interface UpdateProjectRequest {
  name?: string;
  prompt?: string;
  settings?: Partial<{
    variations_per_room: number;
    model: string;
    quality: string;
    size: string;
  }>;
  design_brief?: DesignBrief | null;
  conversation_history?: ChatMessage[] | null;
}

export interface UpdateRoomRequest {
  label: string;
}
```

- [ ] **Step 9.2: Add update + room API functions**

Append to `frontend/services/stagingApi.ts` (after `deleteProject`):

```ts
/**
 * Update a project's name, prompt, or settings.
 */
export async function updateProject(
  projectId: string,
  payload: UpdateProjectRequest,
): Promise<StagingProject> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}`;
  if (API_DEBUG) console.log(`PATCH ${url}`, payload);
  const response = await fetch(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to update project: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return data.project ?? data;
}

/**
 * Rename a single room.
 */
export async function updateRoom(
  projectId: string,
  roomId: string,
  payload: UpdateRoomRequest,
): Promise<StagingProject> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/rooms/${roomId}`;
  if (API_DEBUG) console.log(`PATCH ${url}`, payload);
  const response = await fetch(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to update room: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return data.project ?? data;
}

/**
 * Remove a room from the project.
 */
export async function removeRoom(
  projectId: string,
  roomId: string,
): Promise<StagingProject> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/rooms/${roomId}`;
  if (API_DEBUG) console.log(`DELETE ${url}`);
  const response = await fetch(url, { method: 'DELETE' });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to delete room: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return data.project ?? data;
}
```

- [ ] **Step 9.3: Type-check**

```
cd frontend && npx tsc --noEmit
```
Expected: no new errors.

- [ ] **Step 9.4: Build**

```
cd frontend && npm run build
```
Expected: build succeeds.

- [ ] **Step 9.5: Commit**

```
git add frontend/services/stagingApi.ts
git commit -m "feat(staging-ui): add updateProject, updateRoom, removeRoom helpers

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 10: Add version client functions + types

**Files:**
- Modify: `frontend/services/stagingApi.ts`

- [ ] **Step 10.1: Add version types**

Append after `UpdateRoomRequest`:

```ts
export interface VersionSnapshot {
  name: string;
  prompt: string;
  settings: {
    variations_per_room: number;
    model: string;
    quality: string;
    size: string;
  };
  design_brief?: DesignBrief | null;
  room_labels: Record<string, string>;
  conversation_history?: ChatMessage[] | null;
}

export interface StagingProjectVersion {
  id: string;
  project_id: string;
  snapshot: VersionSnapshot;
  label?: string | null;
  note?: string | null;
  created_at?: string;
}

export interface CreateVersionRequest {
  label?: string;
  note?: string;
}
```

- [ ] **Step 10.2: Add version API functions**

Append to the bottom of `frontend/services/stagingApi.ts`:

```ts
/**
 * List saved versions of a project (newest first).
 */
export async function listVersions(projectId: string): Promise<StagingProjectVersion[]> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/versions`;
  if (API_DEBUG) console.log(`GET ${url}`);
  const response = await fetch(url);
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to list versions: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return data.versions ?? [];
}

/**
 * Save the current project state as a version snapshot.
 */
export async function createVersion(
  projectId: string,
  payload: CreateVersionRequest = {},
): Promise<StagingProjectVersion> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/versions`;
  if (API_DEBUG) console.log(`POST ${url}`, payload);
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to create version: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return data.version;
}

/**
 * Apply a saved version to the live project (auto-snapshots first).
 */
export async function revertVersion(
  projectId: string,
  versionId: string,
): Promise<StagingProject> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/versions/${versionId}/revert`;
  if (API_DEBUG) console.log(`POST ${url}`);
  const response = await fetch(url, { method: 'POST' });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to revert: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return data.project ?? data;
}

/**
 * Delete a saved version.
 */
export async function deleteVersion(
  projectId: string,
  versionId: string,
): Promise<void> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/versions/${versionId}`;
  if (API_DEBUG) console.log(`DELETE ${url}`);
  const response = await fetch(url, { method: 'DELETE' });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to delete version: ${response.status} ${errorText}`);
  }
}
```

- [ ] **Step 10.3: Build**

```
cd frontend && npm run build
```
Expected: build succeeds.

- [ ] **Step 10.4: Commit**

```
git add frontend/services/stagingApi.ts
git commit -m "feat(staging-ui): add version list/create/revert/delete helpers

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 5 — Frontend Shared Extraction

### Task 11: Add `disabled` prop to `DesignBriefEditor`

**Files:**
- Modify: `frontend/components/staging/DesignBriefEditor.tsx`

- [ ] **Step 11.1: Add the prop and thread through**

Replace lines 13-19 of `DesignBriefEditor.tsx` with:

```tsx
interface DesignBriefEditorProps {
  brief: DesignBrief;
  onChange: (brief: DesignBrief) => void;
  imageLabels: Record<string, string>;
  disabled?: boolean;
}

export function DesignBriefEditor({ brief, onChange, imageLabels, disabled = false }: DesignBriefEditorProps) {
```

Now thread `disabled` into every editable input/textarea/button. Update the JSX:

- Global instructions textarea: add `disabled={disabled}`.
- `<PlantPaletteTable plants={...} onChange={...} />` — add `disabled={disabled}`. (This requires accepting `disabled` in PlantPaletteTable; if it's not currently supported, add it as an optional pass-through that disables the underlying inputs.)
- All four placement-guide inputs: add `disabled={disabled}`.
- Preserve elements: `disabled={disabled}` on the Input + add-button + remove-X buttons.
- Generation Settings variations_per_room input: add `disabled={disabled}`.

The other model/quality/size inputs already render with `disabled` and stay disabled regardless.

- [ ] **Step 11.2: Update `PlantPaletteTable` to accept `disabled`**

Open `frontend/components/staging/PlantPaletteTable.tsx`. Add `disabled?: boolean` to its props, default `false`, and apply it to every interactive control inside. (No behavior change when `disabled` is false.)

- [ ] **Step 11.3: Build**

```
cd frontend && npm run build
```
Expected: build succeeds.

- [ ] **Step 11.4: Commit**

```
git add frontend/components/staging/DesignBriefEditor.tsx frontend/components/staging/PlantPaletteTable.tsx
git commit -m "refactor(staging-ui): allow DesignBriefEditor and PlantPaletteTable to be disabled

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 12: Extract `DesignChatPanel` (pure UI) from `DesignChat`

**Files:**
- Create: `frontend/components/staging/DesignChatPanel.tsx`
- Modify: `frontend/components/staging/DesignChat.tsx`

- [ ] **Step 12.1: Create the pure-UI panel**

Create `frontend/components/staging/DesignChatPanel.tsx`:

```tsx
"use client"

import { useEffect, useRef } from "react";
import { Loader2, Send, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { QuickReplyChips } from "./QuickReplyChips";
import type { ChatMessage } from "@/services/stagingApi";

export interface DesignChatPanelProps {
  conversationHistory: ChatMessage[];
  /** Render this above the messages when the conversation is empty. */
  emptyState?: React.ReactNode;
  /** Sent when the user submits the input. */
  onSend: (message: string) => void;
  /** Currently-typed value (controlled). */
  inputValue: string;
  onInputChange: (value: string) => void;
  /** Disables the input + send button. */
  isDisabled?: boolean;
  /** Renders a typing indicator when true. */
  isThinking?: boolean;
  /** Optional content rendered above the input row (e.g., proceed button). */
  footerSlot?: React.ReactNode;
  /** Quick-reply suggestions; clicking calls onSuggestionSelect. */
  suggestedActions?: string[];
  onSuggestionSelect?: (action: string) => void;
  /** Optional focused-image badge. */
  focusedImageId?: string | null;
  focusedImageLabel?: string | null;
  onClearFocus?: () => void;
  /** Placeholder for the input. */
  placeholder?: string;
  /** Optional overlay (e.g., brief-generation loader). */
  overlay?: React.ReactNode;
}

export function DesignChatPanel({
  conversationHistory,
  emptyState,
  onSend,
  inputValue,
  onInputChange,
  isDisabled = false,
  isThinking = false,
  footerSlot,
  suggestedActions = [],
  onSuggestionSelect,
  focusedImageId,
  focusedImageLabel,
  onClearFocus,
  placeholder = "Type a message...",
  overlay,
}: DesignChatPanelProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversationHistory, isThinking]);

  return (
    <div className="flex flex-col h-full relative">
      {overlay}

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {conversationHistory.length === 0 && emptyState}

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

        {isThinking && (
          <div className="flex gap-2">
            <div className="w-7 h-7 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
              <Sparkles className="h-3.5 w-3.5 text-primary-foreground" />
            </div>
            <div className="bg-muted rounded-sm rounded-tl-none p-3">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}

        {suggestedActions.length > 0 && !isThinking && onSuggestionSelect && (
          <QuickReplyChips actions={suggestedActions} onSelect={onSuggestionSelect} />
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="border-t">
        {footerSlot}
        <div className="p-3 space-y-2">
          <div className="flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (inputValue.trim()) onSend(inputValue);
                }
              }}
              placeholder={placeholder}
              className="flex-1 bg-muted border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
              disabled={isDisabled}
            />
            <Button
              size="sm"
              onClick={() => inputValue.trim() && onSend(inputValue)}
              disabled={!inputValue.trim() || isDisabled}
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
          {focusedImageId && onClearFocus && (
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
    </div>
  );
}
```

- [ ] **Step 12.2: Refactor `DesignChat` to use the panel**

Replace `frontend/components/staging/DesignChat.tsx` with:

```tsx
"use client"

import { useState } from "react";
import { ChevronRight, FileText, Loader2 } from "lucide-react";
import { Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DesignChatPanel } from "./DesignChatPanel";
import { chatWithProject, ChatMessage, ChatResponse } from "@/services/stagingApi";

const PROCEED_PATTERNS = [
  'go ahead', 'proceed', 'generate brief', 'generate the brief', 'create brief',
  'create the brief', 'looks good', "let's go", "let's do it", "let's proceed",
  'move on', 'next step', "i'm happy", "i'm ready", "that's great", "thats great",
  'perfect', 'sounds good', 'do it', 'ready to go', 'good to go', "let's move on",
  "move to the brief", "move to brief", "design brief", "make the brief",
];

function isProceedIntent(message: string): boolean {
  const lower = message.toLowerCase().trim();
  return PROCEED_PATTERNS.some(phrase => lower.includes(phrase));
}

interface DesignChatProps {
  projectId: string;
  focusedImageId: string | null;
  focusedImageLabel: string | null;
  onClearFocus: () => void;
  onReadyForBrief: () => void;
  isGeneratingBrief?: boolean;
  canGenerateBrief?: boolean;
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
  isGeneratingBrief = false,
  canGenerateBrief = false,
  initialMessage,
  conversationHistory,
  onHistoryUpdate,
}: DesignChatProps) {
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [suggestedActions, setSuggestedActions] = useState<string[]>([]);

  const sendMessage = async (message: string) => {
    if (!message.trim() || isLoading || isGeneratingBrief) return;

    if (canGenerateBrief && isProceedIntent(message)) {
      setInput("");
      onReadyForBrief();
      return;
    }

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
      setSuggestedActions(response.ready_for_brief ? ["generate_brief"] : response.suggested_actions);
    } catch {
      onHistoryUpdate([...updatedHistory, { role: "assistant", content: "Sorry, I had trouble processing that. Could you try again?" }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChipSelect = (action: string) => {
    if (action === "generate_brief") {
      onReadyForBrief();
      return;
    }
    sendMessage(`I'd like to ${action.replace(/_/g, " ")}`);
  };

  const isDisabled = isLoading || isGeneratingBrief;
  const showBriefButton = canGenerateBrief && !isGeneratingBrief;
  const briefButtonProminent = suggestedActions.includes("generate_brief");

  const overlay = isGeneratingBrief ? (
    <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-10 rounded-b-lg">
      <div className="text-center space-y-3">
        <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto" />
        <div className="space-y-1">
          <p className="font-semibold text-sm">Creating your Design Brief...</p>
          <p className="text-xs text-muted-foreground">Analyzing the conversation to build your plan</p>
        </div>
      </div>
    </div>
  ) : null;

  const emptyState = initialMessage ? (
    <div className="flex gap-2">
      <div className="w-7 h-7 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
        <Sparkles className="h-3.5 w-3.5 text-primary-foreground" />
      </div>
      <div className="bg-muted rounded-sm rounded-tl-none p-3 max-w-[85%]">
        <p className="text-sm whitespace-pre-wrap">{initialMessage}</p>
      </div>
    </div>
  ) : null;

  const footerSlot = showBriefButton ? (
    <div className="px-3 pt-3">
      <Button
        onClick={onReadyForBrief}
        variant={briefButtonProminent ? "default" : "outline"}
        className={`w-full ${briefButtonProminent ? "animate-in fade-in slide-in-from-bottom-2 duration-300" : ""}`}
      >
        <FileText className="h-4 w-4 mr-2" />
        Generate Design Brief
        <ChevronRight className="h-4 w-4 ml-1" />
      </Button>
    </div>
  ) : null;

  return (
    <DesignChatPanel
      conversationHistory={conversationHistory}
      emptyState={emptyState}
      onSend={sendMessage}
      inputValue={input}
      onInputChange={setInput}
      isDisabled={isDisabled}
      isThinking={isLoading}
      footerSlot={footerSlot}
      suggestedActions={suggestedActions}
      onSuggestionSelect={handleChipSelect}
      focusedImageId={focusedImageId}
      focusedImageLabel={focusedImageLabel}
      onClearFocus={onClearFocus}
      placeholder={
        isGeneratingBrief
          ? "Generating design brief..."
          : canGenerateBrief
            ? 'Keep chatting, or say "go ahead" to generate the brief'
            : "Describe what you'd like to visualize..."
      }
      overlay={overlay}
    />
  );
}
```

- [ ] **Step 12.3: Build**

```
cd frontend && npm run build
```
Expected: build succeeds, no behavioral change in the wizard.

- [ ] **Step 12.4: Commit**

```
git add frontend/components/staging/DesignChat.tsx frontend/components/staging/DesignChatPanel.tsx
git commit -m "refactor(staging-ui): extract DesignChatPanel from DesignChat

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 13: Create `GenerationSettingsForm`

**Files:**
- Create: `frontend/components/staging/GenerationSettingsForm.tsx`

- [ ] **Step 13.1: Create the form**

Create `frontend/components/staging/GenerationSettingsForm.tsx`:

```tsx
"use client"

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { StagingProject } from "@/services/stagingApi";

type Settings = StagingProject["settings"];

interface GenerationSettingsFormProps {
  settings: Settings;
  onChange: (settings: Settings) => void;
  disabled?: boolean;
}

export function GenerationSettingsForm({ settings, onChange, disabled = false }: GenerationSettingsFormProps) {
  const update = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    onChange({ ...settings, [key]: value });
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-3xl">
      <div className="space-y-1">
        <Label htmlFor="settings-variations" className="text-xs text-muted-foreground">
          Variations per room
        </Label>
        <Input
          id="settings-variations"
          type="number"
          min={1}
          max={10}
          value={settings.variations_per_room}
          disabled={disabled}
          onChange={(e) => update("variations_per_room", Math.max(1, Math.min(10, parseInt(e.target.value) || 1)))}
        />
      </div>
      <div className="space-y-1">
        <Label htmlFor="settings-quality" className="text-xs text-muted-foreground">
          Quality
        </Label>
        <Input
          id="settings-quality"
          value={settings.quality ?? ""}
          disabled={disabled}
          onChange={(e) => update("quality", e.target.value)}
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 13.2: Build**

```
cd frontend && npm run build
```
Expected: build succeeds (component is unused — no warning expected because TS doesn't warn on unused exports).

- [ ] **Step 13.3: Commit**

```
git add frontend/components/staging/GenerationSettingsForm.tsx
git commit -m "feat(staging-ui): add GenerationSettingsForm

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 6 — Frontend Hooks

### Task 14: `useBriefEditor` hook

**Files:**
- Create: `frontend/hooks/staging/useBriefEditor.ts`

- [ ] **Step 14.1: Create the hook**

Create `frontend/hooks/staging/useBriefEditor.ts`:

```ts
"use client"

import { useCallback, useEffect, useMemo, useState } from "react";
import type { DesignBrief } from "@/services/stagingApi";

export interface UseBriefEditorOptions {
  initialBrief: DesignBrief | null;
  onSave: (brief: DesignBrief) => Promise<void>;
}

export interface UseBriefEditorResult {
  draft: DesignBrief | null;
  setDraft: (brief: DesignBrief) => void;
  isDirty: boolean;
  isSaving: boolean;
  error: string | null;
  save: () => Promise<void>;
  reset: () => void;
}

function deepEqual(a: unknown, b: unknown) {
  return JSON.stringify(a) === JSON.stringify(b);
}

export function useBriefEditor({ initialBrief, onSave }: UseBriefEditorOptions): UseBriefEditorResult {
  const [draft, setDraftState] = useState<DesignBrief | null>(initialBrief);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraftState(initialBrief);
  }, [initialBrief]);

  const isDirty = useMemo(() => !deepEqual(draft, initialBrief), [draft, initialBrief]);

  const setDraft = useCallback((brief: DesignBrief) => {
    setDraftState(brief);
  }, []);

  const reset = useCallback(() => {
    setDraftState(initialBrief);
    setError(null);
  }, [initialBrief]);

  const save = useCallback(async () => {
    if (!draft) return;
    setIsSaving(true);
    setError(null);
    try {
      await onSave(draft);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save brief");
      throw e;
    } finally {
      setIsSaving(false);
    }
  }, [draft, onSave]);

  return { draft, setDraft, isDirty, isSaving, error, save, reset };
}
```

- [ ] **Step 14.2: Build + commit**

```
cd frontend && npm run build
```
Expected: build succeeds.

```
git add frontend/hooks/staging/useBriefEditor.ts
git commit -m "feat(staging-ui): add useBriefEditor hook

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 15: `useDesignChat` hook

**Files:**
- Create: `frontend/hooks/staging/useDesignChat.ts`

- [ ] **Step 15.1: Create the hook**

Create `frontend/hooks/staging/useDesignChat.ts`:

```ts
"use client"

import { useCallback, useEffect, useState } from "react";
import { chatWithProject, ChatMessage, ChatResponse } from "@/services/stagingApi";

export interface UseDesignChatOptions {
  projectId: string;
  initialHistory?: ChatMessage[];
  onHistoryChange?: (history: ChatMessage[]) => void;
}

export interface UseDesignChatResult {
  history: ChatMessage[];
  setHistory: (history: ChatMessage[]) => void;
  send: (message: string, focusedImageId?: string | null) => Promise<ChatResponse | null>;
  isLoading: boolean;
  suggestedActions: string[];
  setSuggestedActions: (actions: string[]) => void;
  error: string | null;
  clear: () => void;
}

export function useDesignChat({ projectId, initialHistory = [], onHistoryChange }: UseDesignChatOptions): UseDesignChatResult {
  const [history, setHistoryState] = useState<ChatMessage[]>(initialHistory);
  const [isLoading, setIsLoading] = useState(false);
  const [suggestedActions, setSuggestedActions] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setHistoryState(initialHistory);
  }, [initialHistory]);

  const setHistory = useCallback((next: ChatMessage[]) => {
    setHistoryState(next);
    onHistoryChange?.(next);
  }, [onHistoryChange]);

  const send = useCallback(async (message: string, focusedImageId: string | null = null): Promise<ChatResponse | null> => {
    if (!message.trim() || isLoading) return null;

    const userMsg: ChatMessage = { role: "user", content: message, focused_image_id: focusedImageId ?? undefined };
    const updatedHistory = [...history, userMsg];
    setHistory(updatedHistory);
    setIsLoading(true);
    setError(null);

    try {
      const response = await chatWithProject(projectId, message, updatedHistory.slice(0, -1), focusedImageId ?? undefined);
      const assistantMsg: ChatMessage = { role: "assistant", content: response.reply };
      setHistory([...updatedHistory, assistantMsg]);
      setSuggestedActions(response.ready_for_brief ? ["generate_brief"] : response.suggested_actions);
      return response;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Chat failed");
      setHistory([...updatedHistory, { role: "assistant", content: "Sorry, I had trouble processing that. Could you try again?" }]);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [projectId, history, isLoading, setHistory]);

  const clear = useCallback(() => {
    setHistory([]);
    setSuggestedActions([]);
    setError(null);
  }, [setHistory]);

  return { history, setHistory, send, isLoading, suggestedActions, setSuggestedActions, error, clear };
}
```

- [ ] **Step 15.2: Build + commit**

```
cd frontend && npm run build
```

```
git add frontend/hooks/staging/useDesignChat.ts
git commit -m "feat(staging-ui): add useDesignChat hook

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 16: `useProjectSettings` hook

**Files:**
- Create: `frontend/hooks/staging/useProjectSettings.ts`

- [ ] **Step 16.1: Create the hook**

Create `frontend/hooks/staging/useProjectSettings.ts`:

```ts
"use client"

import { useCallback, useEffect, useMemo, useState } from "react";
import type { StagingProject, UpdateProjectRequest } from "@/services/stagingApi";

export interface UseProjectSettingsOptions {
  project: StagingProject | null;
  onSave: (payload: UpdateProjectRequest) => Promise<StagingProject>;
}

interface SettingsDraft {
  name: string;
  prompt: string;
  settings: StagingProject["settings"];
}

function buildDraft(project: StagingProject | null): SettingsDraft {
  return {
    name: project?.name ?? "",
    prompt: project?.prompt ?? "",
    settings: project?.settings ?? {
      variations_per_room: 5,
      style: "",
      room_count: 0,
    },
  };
}

export function useProjectSettings({ project, onSave }: UseProjectSettingsOptions) {
  const [draft, setDraft] = useState<SettingsDraft>(() => buildDraft(project));
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraft(buildDraft(project));
  }, [project]);

  const isDirty = useMemo(() => {
    if (!project) return false;
    return (
      draft.name !== project.name ||
      draft.prompt !== project.prompt ||
      JSON.stringify(draft.settings) !== JSON.stringify(project.settings)
    );
  }, [draft, project]);

  const update = useCallback(<K extends keyof SettingsDraft>(key: K, value: SettingsDraft[K]) => {
    setDraft((prev) => ({ ...prev, [key]: value }));
  }, []);

  const reset = useCallback(() => {
    setDraft(buildDraft(project));
    setError(null);
  }, [project]);

  const save = useCallback(async () => {
    if (!project || !isDirty) return;
    setIsSaving(true);
    setError(null);
    try {
      const payload: UpdateProjectRequest = {};
      if (draft.name !== project.name) payload.name = draft.name;
      if (draft.prompt !== project.prompt) payload.prompt = draft.prompt;
      if (JSON.stringify(draft.settings) !== JSON.stringify(project.settings)) {
        payload.settings = {
          variations_per_room: draft.settings.variations_per_room,
          quality: draft.settings.quality,
        };
      }
      await onSave(payload);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save settings");
      throw e;
    } finally {
      setIsSaving(false);
    }
  }, [draft, project, isDirty, onSave]);

  return { draft, update, isDirty, isSaving, error, save, reset };
}
```

- [ ] **Step 16.2: Build + commit**

```
cd frontend && npm run build
```

```
git add frontend/hooks/staging/useProjectSettings.ts
git commit -m "feat(staging-ui): add useProjectSettings hook

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 17: `useProjectVersions` hook

**Files:**
- Create: `frontend/hooks/staging/useProjectVersions.ts`

- [ ] **Step 17.1: Create the hook**

Create `frontend/hooks/staging/useProjectVersions.ts`:

```ts
"use client"

import { useCallback, useEffect, useState } from "react";
import {
  createVersion,
  deleteVersion,
  listVersions,
  revertVersion,
  type CreateVersionRequest,
  type StagingProject,
  type StagingProjectVersion,
} from "@/services/stagingApi";

export interface UseProjectVersionsResult {
  versions: StagingProjectVersion[];
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  save: (payload?: CreateVersionRequest) => Promise<StagingProjectVersion>;
  revert: (versionId: string) => Promise<StagingProject>;
  remove: (versionId: string) => Promise<void>;
}

export function useProjectVersions(projectId: string): UseProjectVersionsResult {
  const [versions, setVersions] = useState<StagingProjectVersion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      setVersions(await listVersions(projectId));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load versions");
    } finally {
      setIsLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const save = useCallback(async (payload: CreateVersionRequest = {}) => {
    const version = await createVersion(projectId, payload);
    await refresh();
    return version;
  }, [projectId, refresh]);

  const revert = useCallback(async (versionId: string) => {
    const project = await revertVersion(projectId, versionId);
    await refresh();
    return project;
  }, [projectId, refresh]);

  const remove = useCallback(async (versionId: string) => {
    await deleteVersion(projectId, versionId);
    await refresh();
  }, [projectId, refresh]);

  return { versions, isLoading, error, refresh, save, revert, remove };
}
```

- [ ] **Step 17.2: Build + commit**

```
cd frontend && npm run build
```

```
git add frontend/hooks/staging/useProjectVersions.ts
git commit -m "feat(staging-ui): add useProjectVersions hook

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 7 — Frontend Edit Page Components

All components in this phase live in `frontend/components/staging/edit/`.

### Task 18: `ProjectTabs` with URL `?tab=` sync

**Files:**
- Create: `frontend/components/staging/edit/ProjectTabs.tsx`

- [ ] **Step 18.1: Create the component**

Create `frontend/components/staging/edit/ProjectTabs.tsx`:

```tsx
"use client"

import { ReactNode, useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export type ProjectTabId = "gallery" | "brief" | "settings" | "history";

const VALID_TABS: ProjectTabId[] = ["gallery", "brief", "settings", "history"];

interface ProjectTabsProps {
  defaultTab?: ProjectTabId;
  galleryContent: ReactNode;
  briefContent: ReactNode;
  settingsContent: ReactNode;
  historyContent: ReactNode;
}

export function ProjectTabs({
  defaultTab = "gallery",
  galleryContent,
  briefContent,
  settingsContent,
  historyContent,
}: ProjectTabsProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const param = searchParams.get("tab");
  const initial: ProjectTabId = param && (VALID_TABS as string[]).includes(param)
    ? (param as ProjectTabId)
    : defaultTab;
  const [active, setActive] = useState<ProjectTabId>(initial);

  useEffect(() => {
    if (param && (VALID_TABS as string[]).includes(param) && param !== active) {
      setActive(param as ProjectTabId);
    }
  }, [param, active]);

  const onChange = useCallback((value: string) => {
    const next = (VALID_TABS as string[]).includes(value) ? (value as ProjectTabId) : defaultTab;
    setActive(next);
    const params = new URLSearchParams(searchParams.toString());
    if (next === defaultTab) {
      params.delete("tab");
    } else {
      params.set("tab", next);
    }
    const query = params.toString();
    router.replace(query ? `?${query}` : "?", { scroll: false });
  }, [router, searchParams, defaultTab]);

  return (
    <Tabs value={active} onValueChange={onChange} className="w-full">
      <TabsList>
        <TabsTrigger value="gallery">Gallery</TabsTrigger>
        <TabsTrigger value="brief">Design Brief</TabsTrigger>
        <TabsTrigger value="settings">Settings</TabsTrigger>
        <TabsTrigger value="history">History</TabsTrigger>
      </TabsList>
      <TabsContent value="gallery">{galleryContent}</TabsContent>
      <TabsContent value="brief">{briefContent}</TabsContent>
      <TabsContent value="settings">{settingsContent}</TabsContent>
      <TabsContent value="history">{historyContent}</TabsContent>
    </Tabs>
  );
}
```

- [ ] **Step 18.2: Build + commit**

```
cd frontend && npm run build
```

```
git add frontend/components/staging/edit/ProjectTabs.tsx
git commit -m "feat(staging-ui): add ProjectTabs with URL ?tab= sync

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 19: `EditableProjectName`

**Files:**
- Create: `frontend/components/staging/edit/EditableProjectName.tsx`

- [ ] **Step 19.1: Create the component**

Create `frontend/components/staging/edit/EditableProjectName.tsx`:

```tsx
"use client"

import { useEffect, useRef, useState } from "react";
import { Check, Pencil, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface EditableProjectNameProps {
  name: string;
  onSave: (next: string) => Promise<void> | void;
  disabled?: boolean;
}

export function EditableProjectName({ name, onSave, disabled = false }: EditableProjectNameProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState(name);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isEditing) inputRef.current?.focus();
  }, [isEditing]);

  useEffect(() => {
    if (!isEditing) setDraft(name);
  }, [name, isEditing]);

  const beginEdit = () => {
    if (disabled) return;
    setError(null);
    setDraft(name);
    setIsEditing(true);
  };

  const cancel = () => {
    setIsEditing(false);
    setDraft(name);
    setError(null);
  };

  const commit = async () => {
    const trimmed = draft.trim();
    if (!trimmed) {
      setError("Name cannot be empty");
      return;
    }
    if (trimmed === name) {
      setIsEditing(false);
      return;
    }
    setIsSaving(true);
    setError(null);
    try {
      await onSave(trimmed);
      setIsEditing(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to update name");
    } finally {
      setIsSaving(false);
    }
  };

  if (!isEditing) {
    return (
      <div className="flex items-center gap-2">
        <h1 className="text-2xl font-semibold tracking-tight truncate">{name}</h1>
        <Button
          aria-label="Edit project name"
          size="sm"
          variant="ghost"
          onClick={beginEdit}
          disabled={disabled}
        >
          <Pencil className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <Input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void commit();
            if (e.key === "Escape") cancel();
          }}
          className="text-2xl font-semibold h-10"
          disabled={isSaving}
        />
        <Button size="sm" onClick={commit} disabled={isSaving} aria-label="Save name">
          <Check className="h-4 w-4" />
        </Button>
        <Button size="sm" variant="ghost" onClick={cancel} disabled={isSaving} aria-label="Cancel rename">
          <X className="h-4 w-4" />
        </Button>
      </div>
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
```

- [ ] **Step 19.2: Build + commit**

```
cd frontend && npm run build
```

```
git add frontend/components/staging/edit/EditableProjectName.tsx
git commit -m "feat(staging-ui): add EditableProjectName

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 20: `RegeneratePrompt`

**Files:**
- Create: `frontend/components/staging/edit/RegeneratePrompt.tsx`

- [ ] **Step 20.1: Create the component**

Create `frontend/components/staging/edit/RegeneratePrompt.tsx`:

```tsx
"use client"

import { Loader2, Sparkles, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface RegeneratePromptProps {
  visible: boolean;
  message?: string;
  onRegenerate: () => Promise<void> | void;
  onDismiss: () => void;
  isWorking?: boolean;
}

export function RegeneratePrompt({
  visible,
  message = "Your changes affect the rendered staging. Regenerate to apply them now, or save them and regenerate later.",
  onRegenerate,
  onDismiss,
  isWorking = false,
}: RegeneratePromptProps) {
  if (!visible) return null;

  return (
    <div className="border border-amber-500/40 bg-amber-50 dark:bg-amber-950/30 text-amber-900 dark:text-amber-100 rounded-md px-4 py-3 flex items-start gap-3">
      <Sparkles className="h-4 w-4 mt-0.5 flex-shrink-0" />
      <div className="flex-1 text-sm">{message}</div>
      <div className="flex items-center gap-2">
        <Button size="sm" onClick={() => void onRegenerate()} disabled={isWorking}>
          {isWorking ? (
            <>
              <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              Regenerating
            </>
          ) : (
            "Regenerate"
          )}
        </Button>
        <Button size="sm" variant="ghost" onClick={onDismiss} disabled={isWorking} aria-label="Dismiss">
          <X className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
```

- [ ] **Step 20.2: Build + commit**

```
cd frontend && npm run build
```

```
git add frontend/components/staging/edit/RegeneratePrompt.tsx
git commit -m "feat(staging-ui): add RegeneratePrompt banner

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 21: `ProjectGalleryTab`

**Files:**
- Create: `frontend/components/staging/edit/ProjectGalleryTab.tsx`

- [ ] **Step 21.1: Create the component**

The existing project page already has gallery rendering. We extract that block into this component so the page becomes a thin shell. Keep behavior identical — just move the JSX.

Create `frontend/components/staging/edit/ProjectGalleryTab.tsx`:

```tsx
"use client"

import Image from "next/image";
import { Loader2, RefreshCw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { StagingProject } from "@/services/stagingApi";

export interface ProjectGalleryTabProps {
  project: StagingProject;
  isRegenerating: boolean;
  onRegenerate: () => void;
}

export function ProjectGalleryTab({ project, isRegenerating, onRegenerate }: ProjectGalleryTabProps) {
  const rooms = project.rooms ?? [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          {rooms.length} {rooms.length === 1 ? "room" : "rooms"} · {project.settings.variations_per_room} variations each
        </p>
        <Button onClick={onRegenerate} disabled={isRegenerating} variant="outline" size="sm">
          {isRegenerating ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Regenerating
            </>
          ) : (
            <>
              <RefreshCw className="h-4 w-4 mr-2" />
              Regenerate all
            </>
          )}
        </Button>
      </div>

      {rooms.length === 0 ? (
        <div className="border border-dashed rounded-md p-8 text-center text-sm text-muted-foreground">
          No rooms yet. Add a room in the Design Brief tab to start generating staged views.
        </div>
      ) : (
        <div className="space-y-8">
          {rooms.map((room) => (
            <section key={room.id} className="space-y-3">
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-semibold">{room.label}</h2>
                <Badge variant="outline" className="text-xs">{room.variations.length} variations</Badge>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {room.variations.map((variation) => (
                  <div key={variation.id} className="border rounded-md overflow-hidden bg-muted/30">
                    {variation.status === "completed" && variation.image_url ? (
                      <Image
                        src={variation.image_url}
                        alt={`${room.label} variation`}
                        width={512}
                        height={512}
                        className="w-full h-auto"
                        unoptimized
                      />
                    ) : (
                      <div className="aspect-square flex items-center justify-center text-muted-foreground text-xs">
                        {variation.status === "failed" ? "Failed" : "Pending"}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
```

> **Note:** `Variation.image_url` is a blob-storage URL. If the deployment relies on `sasTokenService` (see existing `[id]/page.tsx`), wrap the URL through it before rendering. Match the helper used by the page you're replacing.

- [ ] **Step 21.2: Build + commit**

```
cd frontend && npm run build
```

```
git add frontend/components/staging/edit/ProjectGalleryTab.tsx
git commit -m "feat(staging-ui): add ProjectGalleryTab

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 22: `ProjectBriefTab`

**Files:**
- Create: `frontend/components/staging/edit/ProjectBriefTab.tsx`

- [ ] **Step 22.1: Create the component**

Create `frontend/components/staging/edit/ProjectBriefTab.tsx`:

```tsx
"use client"

import { useEffect, useState } from "react";
import { Loader2, MessageSquare, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DesignBriefEditor } from "@/components/staging/DesignBriefEditor";
import { DesignChatPanel } from "@/components/staging/DesignChatPanel";
import { RegeneratePrompt } from "./RegeneratePrompt";
import { useBriefEditor } from "@/hooks/staging/useBriefEditor";
import { useDesignChat } from "@/hooks/staging/useDesignChat";
import { updateProject } from "@/services/stagingApi";
import type { StagingProject } from "@/services/stagingApi";

interface ProjectBriefTabProps {
  project: StagingProject;
  imageLabels: Record<string, string>;
  onProjectUpdate: (project: StagingProject) => void;
  onRequestRegenerate: () => Promise<void> | void;
  isRegenerating: boolean;
}

export function ProjectBriefTab({
  project,
  imageLabels,
  onProjectUpdate,
  onRequestRegenerate,
  isRegenerating,
}: ProjectBriefTabProps) {
  const [showRegeneratePrompt, setShowRegeneratePrompt] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState("");

  const editor = useBriefEditor({
    initialBrief: project.design_brief ?? null,
    onSave: async (brief) => {
      const updated = await updateProject(project.id, { design_brief: brief });
      onProjectUpdate(updated);
      setShowRegeneratePrompt(true);
    },
  });

  // Chat history is persisted server-side by the chat endpoint (see Task 8),
  // so the Brief tab does not need to PATCH /projects after every message.

  const chat = useDesignChat({
    projectId: project.id,
    initialHistory: project.conversation_history ?? [],
  });

  useEffect(() => {
    if (!chatOpen) setChatInput("");
  }, [chatOpen]);

  return (
    <div className="space-y-6">
      <RegeneratePrompt
        visible={showRegeneratePrompt}
        onRegenerate={async () => {
          await onRequestRegenerate();
          setShowRegeneratePrompt(false);
        }}
        onDismiss={() => setShowRegeneratePrompt(false)}
        isWorking={isRegenerating}
      />

      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Design Brief</h2>
          <p className="text-sm text-muted-foreground">Edit the structured plan that drives every render.</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => setChatOpen((v) => !v)}>
            <MessageSquare className="h-4 w-4 mr-2" />
            {chatOpen ? "Hide chat" : "Resume chat"}
          </Button>
          <Button size="sm" onClick={() => void editor.save()} disabled={!editor.isDirty || editor.isSaving}>
            {editor.isSaving ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Saving
              </>
            ) : (
              <>
                <Save className="h-4 w-4 mr-2" />
                Save brief
              </>
            )}
          </Button>
        </div>
      </div>

      {editor.error && <p className="text-sm text-destructive">{editor.error}</p>}

      {editor.draft && (
        <DesignBriefEditor
          brief={editor.draft}
          onChange={editor.setDraft}
          imageLabels={imageLabels}
          disabled={editor.isSaving}
        />
      )}

      {chatOpen && (
        <div className="border rounded-md h-[480px] overflow-hidden">
          <DesignChatPanel
            conversationHistory={chat.history}
            onSend={(msg) => void chat.send(msg)}
            inputValue={chatInput}
            onInputChange={setChatInput}
            isDisabled={chat.isLoading}
            isThinking={chat.isLoading}
            placeholder="Iterate on the brief — your messages are saved with the project."
          />
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 22.2: Build + commit**

```
cd frontend && npm run build
```

```
git add frontend/components/staging/edit/ProjectBriefTab.tsx
git commit -m "feat(staging-ui): add ProjectBriefTab with persistent chat

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 23: `ProjectSettingsTab`

**Files:**
- Create: `frontend/components/staging/edit/ProjectSettingsTab.tsx`

- [ ] **Step 23.1: Create the component**

Create `frontend/components/staging/edit/ProjectSettingsTab.tsx`:

```tsx
"use client"

import { useState } from "react";
import { Loader2, Save, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { GenerationSettingsForm } from "@/components/staging/GenerationSettingsForm";
import { RegeneratePrompt } from "./RegeneratePrompt";
import { useProjectSettings } from "@/hooks/staging/useProjectSettings";
import { deleteProject, updateProject } from "@/services/stagingApi";
import type { StagingProject } from "@/services/stagingApi";

interface ProjectSettingsTabProps {
  project: StagingProject;
  onProjectUpdate: (project: StagingProject) => void;
  onProjectDeleted: () => void;
  onRequestRegenerate: () => Promise<void> | void;
  isRegenerating: boolean;
}

export function ProjectSettingsTab({
  project,
  onProjectUpdate,
  onProjectDeleted,
  onRequestRegenerate,
  isRegenerating,
}: ProjectSettingsTabProps) {
  const [showRegeneratePrompt, setShowRegeneratePrompt] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const settings = useProjectSettings({
    project,
    onSave: async (payload) => {
      const updated = await updateProject(project.id, payload);
      onProjectUpdate(updated);
      if (payload.prompt !== undefined || payload.settings !== undefined) {
        setShowRegeneratePrompt(true);
      }
      return updated;
    },
  });

  const onDelete = async () => {
    setIsDeleting(true);
    setDeleteError(null);
    try {
      await deleteProject(project.id);
      onProjectDeleted();
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : "Failed to delete project");
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-6 max-w-3xl">
      <RegeneratePrompt
        visible={showRegeneratePrompt}
        onRegenerate={async () => {
          await onRequestRegenerate();
          setShowRegeneratePrompt(false);
        }}
        onDismiss={() => setShowRegeneratePrompt(false)}
        isWorking={isRegenerating}
      />

      <section className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold">Project details</h2>
          <p className="text-sm text-muted-foreground">Name and high-level prompt that frames every generation.</p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="settings-name">Project name</Label>
          <Input
            id="settings-name"
            value={settings.draft.name}
            onChange={(e) => settings.update("name", e.target.value)}
            disabled={settings.isSaving}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="settings-prompt">Project prompt</Label>
          <Textarea
            id="settings-prompt"
            rows={5}
            value={settings.draft.prompt}
            onChange={(e) => settings.update("prompt", e.target.value)}
            disabled={settings.isSaving}
          />
        </div>
      </section>

      <section className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold">Generation settings</h2>
          <p className="text-sm text-muted-foreground">How many variations are produced per room when regenerating.</p>
        </div>
        <GenerationSettingsForm
          settings={settings.draft.settings}
          onChange={(next) => settings.update("settings", next)}
          disabled={settings.isSaving}
        />
      </section>

      {settings.error && <p className="text-sm text-destructive">{settings.error}</p>}

      <div className="flex items-center gap-2">
        <Button onClick={() => void settings.save()} disabled={!settings.isDirty || settings.isSaving}>
          {settings.isSaving ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Saving
            </>
          ) : (
            <>
              <Save className="h-4 w-4 mr-2" />
              Save settings
            </>
          )}
        </Button>
        <Button variant="outline" onClick={settings.reset} disabled={!settings.isDirty || settings.isSaving}>
          Discard changes
        </Button>
      </div>

      <section className="border-t pt-6 space-y-3">
        <div>
          <h2 className="text-lg font-semibold text-destructive">Danger zone</h2>
          <p className="text-sm text-muted-foreground">Permanently delete this project, all rooms, variations, and history.</p>
        </div>
        {!deleteOpen ? (
          <Button variant="destructive" onClick={() => setDeleteOpen(true)}>
            <Trash2 className="h-4 w-4 mr-2" />
            Delete project
          </Button>
        ) : (
          <div className="border border-destructive/40 rounded-md p-4 space-y-3">
            <p className="text-sm">
              This will permanently delete <strong>{project.name}</strong>. This action cannot be undone.
            </p>
            {deleteError && <p className="text-sm text-destructive">{deleteError}</p>}
            <div className="flex gap-2">
              <Button variant="destructive" onClick={() => void onDelete()} disabled={isDeleting}>
                {isDeleting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Deleting
                  </>
                ) : (
                  "Yes, delete this project"
                )}
              </Button>
              <Button variant="outline" onClick={() => setDeleteOpen(false)} disabled={isDeleting}>
                Cancel
              </Button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
```

- [ ] **Step 23.2: Build + commit**

```
cd frontend && npm run build
```

```
git add frontend/components/staging/edit/ProjectSettingsTab.tsx
git commit -m "feat(staging-ui): add ProjectSettingsTab

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 24: `ProjectHistoryTab`

**Files:**
- Create: `frontend/components/staging/edit/ProjectHistoryTab.tsx`

- [ ] **Step 24.1: Create the component**

Create `frontend/components/staging/edit/ProjectHistoryTab.tsx`:

```tsx
"use client"

import { useState } from "react";
import { Loader2, RotateCcw, Save, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useProjectVersions } from "@/hooks/staging/useProjectVersions";
import type { StagingProject } from "@/services/stagingApi";

interface ProjectHistoryTabProps {
  project: StagingProject;
  onProjectUpdate: (project: StagingProject) => void;
}

function formatDate(value: string) {
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

export function ProjectHistoryTab({ project, onProjectUpdate }: ProjectHistoryTabProps) {
  const versions = useProjectVersions(project.id);
  const [label, setLabel] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [revertingId, setRevertingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onSaveSnapshot = async () => {
    setIsSaving(true);
    setError(null);
    try {
      await versions.save({ label: label.trim() || undefined });
      setLabel("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save snapshot");
    } finally {
      setIsSaving(false);
    }
  };

  const onRevert = async (versionId: string) => {
    setRevertingId(versionId);
    setError(null);
    try {
      const updated = await versions.revert(versionId);
      onProjectUpdate(updated);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to revert");
    } finally {
      setRevertingId(null);
    }
  };

  const onDelete = async (versionId: string) => {
    setDeletingId(versionId);
    setError(null);
    try {
      await versions.remove(versionId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to delete version");
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="space-y-6 max-w-3xl">
      <section className="space-y-3">
        <div>
          <h2 className="text-lg font-semibold">Save a snapshot</h2>
          <p className="text-sm text-muted-foreground">Capture the current brief, prompt, and settings so you can revert later.</p>
        </div>
        <div className="flex gap-2 items-end">
          <div className="flex-1 space-y-1">
            <Label htmlFor="version-label" className="text-xs text-muted-foreground">
              Optional label
            </Label>
            <Input
              id="version-label"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="e.g. Before Scandinavian variation"
              disabled={isSaving}
            />
          </div>
          <Button onClick={() => void onSaveSnapshot()} disabled={isSaving}>
            {isSaving ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Save className="h-4 w-4 mr-2" />}
            Save snapshot
          </Button>
        </div>
      </section>

      {error && <p className="text-sm text-destructive">{error}</p>}

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">History</h2>
        {versions.isLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading versions
          </div>
        ) : versions.versions.length === 0 ? (
          <p className="text-sm text-muted-foreground">No snapshots yet.</p>
        ) : (
          <ul className="border rounded-md divide-y">
            {versions.versions.map((version) => (
              <li key={version.id} className="flex items-center gap-3 px-3 py-2">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{version.label || `Snapshot ${version.id.slice(0, 8)}`}</p>
                  <p className="text-xs text-muted-foreground">{formatDate(version.created_at)}</p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => void onRevert(version.id)}
                  disabled={revertingId === version.id || deletingId === version.id}
                >
                  {revertingId === version.id ? (
                    <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                  ) : (
                    <RotateCcw className="h-4 w-4 mr-1" />
                  )}
                  Revert
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => void onDelete(version.id)}
                  disabled={revertingId === version.id || deletingId === version.id}
                  aria-label="Delete version"
                >
                  {deletingId === version.id ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                </Button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
```

- [ ] **Step 24.2: Build + commit**

```
cd frontend && npm run build
```

```
git add frontend/components/staging/edit/ProjectHistoryTab.tsx
git commit -m "feat(staging-ui): add ProjectHistoryTab with snapshot/revert/delete

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 8 — Integration & Refactor

### Task 25: Refactor `frontend/app/projects/[id]/page.tsx` to use the new tabs

**Files:**
- Modify: `frontend/app/projects/[id]/page.tsx`

This task is the integration point. The page still owns top-level data fetching, regeneration kick-off, and toast/error handling, but its rendered body becomes a thin shell composed of:
- `EditableProjectName` for the title.
- `ProjectTabs` mounting the four tab components.

- [ ] **Step 25.1: Trim page body to use ProjectTabs**

Read the current `frontend/app/projects/[id]/page.tsx` end-to-end (the file is ~645 lines) and identify three regions:
1. Top-level state + effects (fetch, regenerate, toasts).
2. Title bar.
3. Body (gallery JSX, controls).

Keep region 1 as-is, but drop any state that the new tab components own (brief draft, settings draft, version list — those move into the hooks inside the tab components).

Replace region 2 with:

```tsx
<EditableProjectName
  name={project.name}
  onSave={async (next) => {
    const updated = await updateProject(project.id, { name: next });
    setProject(updated);
  }}
  disabled={isRegenerating}
/>
```

Replace region 3 (the entire gallery body) with:

```tsx
<ProjectTabs
  galleryContent={
    <ProjectGalleryTab
      project={project}
      isRegenerating={isRegenerating}
      onRegenerate={() => void regenerateAll()}
    />
  }
  briefContent={
    <ProjectBriefTab
      project={project}
      imageLabels={imageLabels}
      onProjectUpdate={setProject}
      onRequestRegenerate={regenerateAll}
      isRegenerating={isRegenerating}
    />
  }
  settingsContent={
    <ProjectSettingsTab
      project={project}
      onProjectUpdate={setProject}
      onProjectDeleted={() => router.push("/projects")}
      onRequestRegenerate={regenerateAll}
      isRegenerating={isRegenerating}
    />
  }
  historyContent={
    <ProjectHistoryTab project={project} onProjectUpdate={setProject} />
  }
/>
```

Make sure the imports at the top of the file include:

```tsx
import { useRouter } from "next/navigation";
import { Suspense } from "react";
import { ProjectTabs } from "@/components/staging/edit/ProjectTabs";
import { EditableProjectName } from "@/components/staging/edit/EditableProjectName";
import { ProjectGalleryTab } from "@/components/staging/edit/ProjectGalleryTab";
import { ProjectBriefTab } from "@/components/staging/edit/ProjectBriefTab";
import { ProjectSettingsTab } from "@/components/staging/edit/ProjectSettingsTab";
import { ProjectHistoryTab } from "@/components/staging/edit/ProjectHistoryTab";
import { updateProject } from "@/services/stagingApi";
```

- [ ] **Step 25.2: Wrap `ProjectTabs` in `<Suspense>`**

Next.js 14+ requires `useSearchParams` (used inside `ProjectTabs`) to live under a Suspense boundary during static optimization. Wrap the tabs JSX:

```tsx
<Suspense fallback={<div className="text-sm text-muted-foreground">Loading…</div>}>
  <ProjectTabs ... />
</Suspense>
```

- [ ] **Step 25.3: `imageLabels`**

If the page already builds `imageLabels` (a `Record<string, string>` keyed by image id) for the existing chat / brief flow, pass it to `ProjectBriefTab`. If not, derive it inline:

```tsx
const imageLabels = useMemo(() => {
  const map: Record<string, string> = {};
  project.rooms?.forEach((room) => {
    room.variations.forEach((v, i) => {
      map[v.id] = `${room.label} #${i + 1}`;
    });
  });
  return map;
}, [project]);
```

- [ ] **Step 25.4: Remove the now-dead delete dialog from page**

If the page had a previous "Delete project" button, remove it — the Settings tab owns delete now. Keep the "Regenerate" header button only if it adds value beyond the per-tab regenerate buttons; otherwise remove it for clarity.

- [ ] **Step 25.5: Build + lint**

```
cd frontend && npm run build && npx next lint
```
Expected: build succeeds, lint clean.

- [ ] **Step 25.6: Manual smoke test**

```
cd frontend && npm run dev
```
Open an existing project and verify:
1. Title appears with pencil icon → click → input → type new name → Enter → name updates.
2. Tabs visible: Gallery, Design Brief, Settings, History.
3. Switching tabs updates the URL (`?tab=brief` etc.).
4. Refreshing with `?tab=settings` lands on the Settings tab.
5. Brief tab shows the current brief, "Save brief" enables on edit, save triggers regeneration banner.
6. Settings tab edits name + prompt + variations_per_room; "Delete project" leads to the confirmation flow.
7. History tab loads versions list (empty initially), Save snapshot creates one, Revert restores it.

- [ ] **Step 25.7: Commit**

```
git add frontend/app/projects/[id]/page.tsx
git commit -m "feat(staging-ui): tab-based edit experience for existing projects

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 26: Refactor `NewProjectWizard` to use the shared `DesignChatPanel`

**Files:**
- Modify: `frontend/components/staging/NewProjectWizard.tsx`

The wizard already uses `DesignChat`. After Task 12 the inner `DesignChat` already delegates to `DesignChatPanel`, so no behavior change is strictly required. This task is a small DRY pass.

- [ ] **Step 26.1: Verify the wizard still compiles after Task 12**

```
cd frontend && npm run build
```
Expected: build succeeds with no changes to `NewProjectWizard.tsx`.

- [ ] **Step 26.2: (Optional) Adopt `useDesignChat` in the wizard**

If the wizard maintains its own `conversationHistory` state inline, replace it with `useDesignChat({ projectId: draftProjectId })`. Skip if it would meaningfully change the wizard's flow (e.g., non-trivial side effects on history change). If skipped, document why in the commit message.

- [ ] **Step 26.3: Commit (only if you made changes)**

```
git add frontend/components/staging/NewProjectWizard.tsx
git commit -m "refactor(staging-ui): adopt useDesignChat in NewProjectWizard

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Phase 9 — End-to-End Tests

### Task 27: Playwright E2E for editing an existing project

**Files:**
- Create: `frontend/tests/e2e/edit-project.spec.ts`

This test exercises the full UX of the new feature against the running app. It assumes a project already exists (created via wizard or seeded). Use a fixture or seed step consistent with existing E2E tests in the repo.

- [ ] **Step 27.1: Inspect existing E2E patterns**

Read `frontend/tests/e2e/` (or `frontend/playwright.config.ts`) to confirm how other tests bootstrap a project and how the dev server is launched. Match those patterns; do not invent a new harness.

- [ ] **Step 27.2: Write the spec**

Create `frontend/tests/e2e/edit-project.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

const PROJECT_NAME = "E2E Edit Project";

test.describe("Edit existing project", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/projects");
    // Create a project via wizard or reuse a seeded one. The repo's existing
    // helpers should be used if present; otherwise implement creation here.
    // For brevity, this assumes a helper `createProjectForTest` exists.
    // await createProjectForTest(page, PROJECT_NAME);
  });

  test("inline rename, tab navigation, brief edit, settings edit, history snapshot+revert", async ({ page }) => {
    await page.getByText(PROJECT_NAME).first().click();
    await expect(page.getByRole("heading", { name: PROJECT_NAME })).toBeVisible();

    // Inline rename
    await page.getByRole("button", { name: /edit project name/i }).click();
    const renamed = `${PROJECT_NAME} – renamed`;
    await page.locator('input[value="' + PROJECT_NAME + '"]').fill(renamed);
    await page.keyboard.press("Enter");
    await expect(page.getByRole("heading", { name: renamed })).toBeVisible();

    // Tab navigation persists in URL
    await page.getByRole("tab", { name: "Settings" }).click();
    await expect(page).toHaveURL(/\?tab=settings/);

    // Settings: edit prompt, save, expect regenerate banner
    const prompt = page.getByLabel("Project prompt");
    await prompt.fill("A modern Scandinavian feel with warm lighting.");
    await page.getByRole("button", { name: /save settings/i }).click();
    await expect(page.getByText(/regenerate/i)).toBeVisible();

    // Brief tab: open chat resume
    await page.getByRole("tab", { name: "Design Brief" }).click();
    await page.getByRole("button", { name: /resume chat/i }).click();
    await page.locator('input[placeholder*="iterate"]').fill("Add more soft textiles.");
    await page.keyboard.press("Enter");
    await expect(page.getByText("Add more soft textiles.")).toBeVisible();

    // History: snapshot + revert
    await page.getByRole("tab", { name: "History" }).click();
    await page.getByLabel("Optional label").fill("Pre-experiment");
    await page.getByRole("button", { name: /save snapshot/i }).click();
    await expect(page.getByText("Pre-experiment")).toBeVisible();
    await page.getByRole("button", { name: /revert/i }).first().click();
    // After revert, prompt should match snapshot
    await page.getByRole("tab", { name: "Settings" }).click();
    await expect(prompt).toHaveValue("A modern Scandinavian feel with warm lighting.");
  });
});
```

- [ ] **Step 27.3: Run the test**

```
cd frontend && npx playwright test tests/e2e/edit-project.spec.ts
```
Expected: PASS in all configured browsers. Per `AGENTS.md`, save the report to `tests/playwright/<YYYY-MM-DD-HHMMSS>/` if the project's Playwright config doesn't already do so.

- [ ] **Step 27.4: Commit**

```
git add frontend/tests/e2e/edit-project.spec.ts
git commit -m "test(staging-ui): e2e for editing existing projects

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Final Verification

Run the full local check matrix as per `AGENTS.md`:

```
uv run pytest tests/ --ignore=tests/integration -v
cd frontend && npm run build
cd frontend && npx next lint
cd frontend && npx playwright test
```

Expected: all green. Push the branch and let CI/CD `azd up` the deployment + run the full Playwright suite against the deployed environment per the user's CI/CD policy.

---

## Notes

- **Cosmos partition key:** Versions live in the same `STAGING_COSMOS_CONTAINER_ID` (partition key `/id`) with `doc_type="staging_project_version"` and a `project_id` property. Cross-partition queries (`enable_cross_partition_query=True`) are used for `list_versions`.
- **Chat persistence:** Chat history is stored on the `StagingProject` document under `conversation_history`. The `ProjectBriefTab` writes the latest history back via `updateProject` after every send (best-effort; failures don't break the chat UX).
- **Auto-snapshot on revert:** The backend `revert_version` endpoint creates a `"Before revert"` snapshot before applying the requested version, so a one-click "undo the revert" is always available.
- **Versions don't resurrect deleted rooms.** Reverting a brief only patches labels for rooms whose ids still exist on the project; deleted rooms stay deleted (as documented in the spec).
- **`SecurityControl=Ignore` tag** must be applied to any new Azure resource created or updated as part of this work (per global Copilot CLI instructions). No new Azure resources are required for this feature.

