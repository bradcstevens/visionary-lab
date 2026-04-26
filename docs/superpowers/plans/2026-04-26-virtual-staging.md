# Virtual Staging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent virtual staging feature where users upload room/backyard photos, provide a single styling prompt, and receive 5 AI-generated variations per room in a portfolio grid.

**Architecture:** New `StagingProject` Cosmos DB document model + dedicated staging pipeline that adapts the master prompt per room via LLM image analysis, then fans out to the existing `image_pipeline.process_pipeline()` EDIT flow. Frontend gets `/projects` routes with a wizard, portfolio grid, and SSE progress tracking.

**Tech Stack:** FastAPI (backend), Pydantic v1 validators, Azure Cosmos DB, Azure Blob Storage, gpt-5.4 (LLM), gpt-image-2 (image editing), Next.js 15 / React 19 / shadcn+Radix (frontend), Playwright (e2e tests), pytest (backend tests).

**Spec:** `docs/superpowers/specs/2026-04-26-virtual-staging-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `backend/models/staging.py` | Pydantic models: Room, Variation, StagingProject, CreateProjectRequest, UploadRoomsResponse, ProjectResponse, ProjectListResponse |
| `backend/core/staging_storage.py` | Cosmos DB CRUD for staging projects (create, get, list, update, delete) |
| `backend/core/staging_pipeline.py` | Orchestrator: image analysis → prompt adaptation → fan-out to image pipeline → SSE progress |
| `backend/api/endpoints/staging.py` | FastAPI router: project CRUD, room upload, generate, progress SSE, regenerate |
| `frontend/services/stagingApi.ts` | API client: createProject, uploadRooms, listProjects, getProject, deleteProject, streamGeneration |
| `frontend/app/projects/page.tsx` | Project list page with grid of ProjectCards |
| `frontend/app/projects/new/page.tsx` | New project wizard (name → upload → prompt → generate) |
| `frontend/app/projects/[id]/page.tsx` | Project detail — portfolio grid with RoomGroups |
| `frontend/components/staging/ProjectCard.tsx` | Card for project list — thumbnails, name, room count, variation count |
| `frontend/components/staging/RoomGroup.tsx` | Portfolio row — original pinned left + variation thumbnails |
| `frontend/components/staging/VariationThumbnail.tsx` | Clickable thumbnail with loading/failed/completed states |
| `frontend/components/staging/NewProjectWizard.tsx` | Multi-step form component |
| `frontend/components/staging/ProgressTracker.tsx` | SSE-driven per-room progress indicator |
| `tests/test_staging_models.py` | Pydantic model validation tests |
| `tests/test_staging_api.py` | Endpoint integration tests with mocked services |
| `tests/test_prompt_adaptation.py` | LLM prompt adaptation tests with mocked responses |
| `frontend/tests/e2e/staging-projects.spec.ts` | Playwright: project list + wizard flow |
| `frontend/tests/e2e/staging-portfolio.spec.ts` | Playwright: portfolio grid rendering |

### Modified Files

| File | Change |
|------|--------|
| `backend/main.py` | Add `app.include_router(staging.router, prefix=f"{settings.API_V1_STR}/staging", tags=["staging"])` |
| `backend/core/config.py` | Add `STAGING_MAX_ROOMS_PER_PROJECT`, `STAGING_MAX_VARIATIONS`, `STAGING_CONCURRENT_ROOMS` settings |
| `frontend/components/app-sidebar.tsx` | Add "Projects" nav item to `createItems` array |
| `tests/conftest.py` | Add `mock_staging_cosmos` fixture |

---

## Task 1: Backend Pydantic Models

**Files:**
- Create: `backend/models/staging.py`
- Test: `tests/test_staging_models.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_staging_models.py`:

```python
"""Tests for staging project Pydantic models."""
import pytest
from pydantic import ValidationError


def test_create_project_request_valid():
    from backend.models.staging import CreateProjectRequest
    req = CreateProjectRequest(name="My Project", prompt="Modern minimalist")
    assert req.name == "My Project"
    assert req.prompt == "Modern minimalist"
    assert req.settings.variations_per_room == 5
    assert req.settings.model == "gpt-image-2"


def test_create_project_request_missing_name():
    from backend.models.staging import CreateProjectRequest
    with pytest.raises(ValidationError):
        CreateProjectRequest(prompt="Modern minimalist")


def test_create_project_request_missing_prompt():
    from backend.models.staging import CreateProjectRequest
    with pytest.raises(ValidationError):
        CreateProjectRequest(name="My Project")


def test_staging_settings_defaults():
    from backend.models.staging import StagingSettings
    s = StagingSettings()
    assert s.variations_per_room == 5
    assert s.model == "gpt-image-2"
    assert s.quality == "high"
    assert s.size == "auto"


def test_staging_settings_custom():
    from backend.models.staging import StagingSettings
    s = StagingSettings(variations_per_room=3, model="gpt-image-2", quality="auto")
    assert s.variations_per_room == 3


def test_staging_settings_validates_variations():
    from backend.models.staging import StagingSettings
    with pytest.raises(ValidationError):
        StagingSettings(variations_per_room=0)
    with pytest.raises(ValidationError):
        StagingSettings(variations_per_room=11)


def test_variation_model():
    from backend.models.staging import Variation
    v = Variation(id="abc")
    assert v.status == "pending"
    assert v.image_url is None
    assert v.error is None


def test_room_model():
    from backend.models.staging import Room
    r = Room(id="abc", label="Living Room", original_image_url="https://example.com/img.png")
    assert r.status == "pending"
    assert r.variations == []


def test_staging_project_model():
    from backend.models.staging import StagingProject, StagingSettings
    p = StagingProject(
        id="proj-1",
        name="Test Project",
        prompt="Modern style",
        settings=StagingSettings(),
    )
    assert p.status == "uploading"
    assert p.rooms == []
    assert p.folder_path is None


def test_staging_project_status_values():
    from backend.models.staging import StagingProject, StagingSettings, ProjectStatus
    for status in ["uploading", "processing", "completed", "failed"]:
        p = StagingProject(
            id="proj-1", name="Test", prompt="Test",
            settings=StagingSettings(), status=status,
        )
        assert p.status == status
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/test_staging_models.py -v --no-header 2>&1 | tail -20`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.models.staging'`

- [ ] **Step 3: Write the models**

Create `backend/models/staging.py`:

```python
"""Pydantic models for the virtual staging feature."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ProjectStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ItemStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class StagingSettings(BaseModel):
    variations_per_room: int = Field(5, description="Number of variations to generate per room (1-10)")
    model: str = Field("gpt-image-2", description="Image generation model")
    quality: str = Field("high", description="Image quality setting")
    size: str = Field("auto", description="Image size")

    @validator("variations_per_room")
    def validate_variations(cls, v):
        if v < 1 or v > 10:
            raise ValueError("variations_per_room must be between 1 and 10")
        return v


class GenerationMetadata(BaseModel):
    model: Optional[str] = None
    adapted_prompt: Optional[str] = None
    tokens_used: Optional[int] = None
    generation_time_ms: Optional[int] = None


class Variation(BaseModel):
    id: str
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    status: str = Field(ItemStatus.PENDING, description="Variation generation status")
    error: Optional[str] = None
    generation_metadata: Optional[GenerationMetadata] = None


class Room(BaseModel):
    id: str
    label: str = Field(..., description="Room label, e.g. 'Living Room', 'Backyard'")
    original_image_url: str = Field(..., description="Blob storage URL of uploaded original")
    original_thumbnail_url: Optional[str] = None
    status: str = Field(ItemStatus.PENDING, description="Room processing status")
    error: Optional[str] = None
    variations: List[Variation] = Field(default_factory=list)


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


class CreateProjectRequest(BaseModel):
    name: str = Field(..., description="Project name", examples=["Modern Minimalist Refresh"])
    prompt: str = Field(..., description="Overall styling direction", examples=["Clean lines, warm wood tones, lots of greenery"])
    settings: StagingSettings = Field(default_factory=StagingSettings)


class UploadRoomsResponse(BaseModel):
    project_id: str
    rooms_added: int
    rooms: List[Room]


class ProjectResponse(BaseModel):
    project: StagingProject


class ProjectListResponse(BaseModel):
    projects: List[StagingProject]
    total: int
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/test_staging_models.py -v --no-header 2>&1 | tail -20`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/models/staging.py tests/test_staging_models.py
git commit -m "feat(staging): add Pydantic models for virtual staging projects

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: Config Settings

**Files:**
- Modify: `backend/core/config.py`

- [ ] **Step 1: Add staging settings to config**

Add these fields to the `Settings` class in `backend/core/config.py`, after the existing Cosmos DB settings block:

```python
    # Staging feature
    STAGING_MAX_ROOMS_PER_PROJECT: int = 10
    STAGING_MAX_VARIATIONS: int = 10
    STAGING_CONCURRENT_ROOMS: int = 3
    STAGING_COSMOS_CONTAINER_ID: str = "staging-projects"
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/ --ignore=tests/integration -x -q 2>&1 | tail -10`
Expected: All existing tests PASS

- [ ] **Step 3: Commit**

```bash
git add backend/core/config.py
git commit -m "feat(staging): add staging config settings

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: Cosmos DB Storage Layer

**Files:**
- Create: `backend/core/staging_storage.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write the staging storage service**

Create `backend/core/staging_storage.py`:

```python
"""Cosmos DB CRUD operations for staging projects."""
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from azure.cosmos import ContainerProxy, exceptions
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient

from backend.core.config import settings

logger = logging.getLogger(__name__)


class StagingStorageService:
    """Manages StagingProject documents in Cosmos DB."""

    def __init__(self, container: Optional[ContainerProxy] = None):
        if container is not None:
            self.container = container
            return
        credential = DefaultAzureCredential()
        client = CosmosClient(url=settings.AZURE_COSMOS_DB_ENDPOINT, credential=credential)
        database = client.get_database_client(settings.AZURE_COSMOS_DB_ID)
        self.container = database.create_container_if_not_exists(
            id=settings.STAGING_COSMOS_CONTAINER_ID,
            partition_key={"paths": ["/id"], "kind": "Hash"},
        )

    def create_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        if "id" not in project_data:
            project_data["id"] = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        project_data["created_at"] = now
        project_data["updated_at"] = now
        project_data["doc_type"] = "staging_project"
        return self.container.create_item(body=project_data)

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.container.read_item(item=project_id, partition_key=project_id)
        except exceptions.CosmosResourceNotFoundError:
            return None

    def update_project(self, project_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        existing = self.get_project(project_id)
        if not existing:
            raise ValueError(f"Staging project not found: {project_id}")
        existing.update(updates)
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        return self.container.replace_item(item=project_id, body=existing)

    def list_projects(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        query = "SELECT * FROM c WHERE c.doc_type = 'staging_project' ORDER BY c.created_at DESC OFFSET @offset LIMIT @limit"
        params = [{"name": "@offset", "value": offset}, {"name": "@limit", "value": limit}]
        return list(self.container.query_items(query=query, parameters=params, enable_cross_partition_query=True))

    def count_projects(self) -> int:
        query = "SELECT VALUE COUNT(1) FROM c WHERE c.doc_type = 'staging_project'"
        results = list(self.container.query_items(query=query, enable_cross_partition_query=True))
        return results[0] if results else 0

    def delete_project(self, project_id: str) -> bool:
        try:
            self.container.delete_item(item=project_id, partition_key=project_id)
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False
```

- [ ] **Step 2: Add mock fixture to conftest.py**

Add to `tests/conftest.py` after the existing `mock_cosmos` fixture:

```python
@pytest.fixture
def mock_staging_storage():
    """Mock StagingStorageService for staging endpoint tests."""
    with patch("backend.core.staging_storage.CosmosClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_db = MagicMock()
        mock_client.get_database_client.return_value = mock_db
        mock_container = MagicMock()
        mock_db.create_container_if_not_exists.return_value = mock_container
        yield mock_container
```

- [ ] **Step 3: Verify existing tests still pass**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/ --ignore=tests/integration -x -q 2>&1 | tail -10`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add backend/core/staging_storage.py tests/conftest.py
git commit -m "feat(staging): add Cosmos DB storage layer for staging projects

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: Staging Pipeline Orchestrator

**Files:**
- Create: `backend/core/staging_pipeline.py`
- Test: `tests/test_prompt_adaptation.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prompt_adaptation.py`:

```python
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
    # Verify LLM was called with both user prompt and room analysis
    call_args = mock_llm.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
    system_msg = messages[0]["content"]
    assert "Modern minimalist with warm tones" in system_msg
    assert "grey couch" in system_msg


@pytest.mark.asyncio
async def test_adapt_prompt_handles_llm_non_json():
    from backend.core.staging_pipeline import StagingPipeline

    mock_llm = AsyncMock()
    # First call returns non-JSON, second returns valid JSON
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/test_prompt_adaptation.py -v --no-header 2>&1 | tail -20`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.core.staging_pipeline'`

- [ ] **Step 3: Write the staging pipeline**

Create `backend/core/staging_pipeline.py`:

```python
"""Staging pipeline: image analysis → prompt adaptation → fan-out generation."""
import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import UploadFile

from backend.core.analyze import ImageAnalyzer
from backend.core.azure_storage import AzureBlobStorageService
from backend.core.config import settings
from backend.core.image_pipeline import ImagePipelineService
from backend.core.staging_storage import StagingStorageService
from backend.models.images import ImagePipelineRequest, PipelineAction, PipelineSaveOptions, PipelineAnalysisOptions
from backend.models.staging import ItemStatus, ProjectStatus, Room, StagingProject, Variation

logger = logging.getLogger(__name__)

PROMPT_ADAPTATION_TEMPLATE = """You are a virtual staging assistant. The user wants to visualize decorating ideas for their space.

ROOM ANALYSIS: {room_analysis}
USER'S STYLE DIRECTION: {user_prompt}

Generate {n} distinct variation prompts for an image editing model. Each prompt should:
- ADD items to the existing scene (furniture, decor, plants, landscaping)
- NOT remove or replace existing structures visible in the analysis
- Interpret the user's style direction differently in each variation
- Be specific about what to add and where to place it
- Reference the existing room features from the analysis

Return ONLY a JSON array of {n} strings. No other text."""


class StagingPipeline:
    """Orchestrates virtual staging: analyze → adapt prompt → generate variations."""

    def __init__(
        self,
        async_llm_client,
        llm_deployment: str,
        image_analyzer: ImageAnalyzer,
        image_pipeline: ImagePipelineService,
        storage_service: StagingStorageService,
        blob_service: AzureBlobStorageService,
    ):
        self.async_llm_client = async_llm_client
        self.llm_deployment = llm_deployment
        self.image_analyzer = image_analyzer
        self.image_pipeline = image_pipeline
        self.storage_service = storage_service
        self.blob_service = blob_service
        self.semaphore = asyncio.Semaphore(settings.STAGING_CONCURRENT_ROOMS)

    async def analyze_room(self, image_base64: str) -> Dict[str, Any]:
        """Use ImageAnalyzer to describe what's in the uploaded photo."""
        system_msg = (
            "Describe this room or outdoor space in detail. Include: "
            "existing furniture, decor, colors, flooring, lighting, plants, "
            "architectural features, and any empty areas where items could be added. "
            "Return JSON with keys: description (string), features (list of strings)."
        )
        return await self.image_analyzer.async_image_chat(
            image_base64=image_base64,
            system_message=system_msg,
        )

    async def adapt_prompt(
        self, user_prompt: str, room_analysis: str, n_variations: int,
    ) -> List[str]:
        """Use LLM to create n distinct variation prompts for this room."""
        system_content = PROMPT_ADAPTATION_TEMPLATE.format(
            room_analysis=room_analysis,
            user_prompt=user_prompt,
            n=n_variations,
        )
        for attempt in range(3):
            if attempt:
                await asyncio.sleep(1)
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
                    return [str(p) for p in parsed[:n_variations]]
                if isinstance(parsed, dict) and "prompts" in parsed:
                    return [str(p) for p in parsed["prompts"][:n_variations]]
            except (json.JSONDecodeError, KeyError, IndexError):
                logger.warning(f"Prompt adaptation attempt {attempt+1} returned invalid JSON, retrying")
                continue
        raise RuntimeError("Failed to adapt prompt after 3 attempts")

    async def process_room(
        self, project: StagingProject, room: Room,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a single room: analyze → adapt → generate variations. Yields SSE events."""
        async with self.semaphore:
            yield {"type": "room_started", "room_id": room.id, "label": room.label}

            # Update room status
            room.status = ItemStatus.PROCESSING
            self._update_room_in_project(project, room)

            try:
                # Step 1: Download original and analyze
                image_bytes = await self.blob_service.get_asset_content(
                    blob_name=room.original_image_url.split("/")[-2] + "/" + room.original_image_url.split("/")[-1],
                    container_name=settings.AZURE_BLOB_IMAGE_CONTAINER,
                )
                image_b64 = base64.b64encode(image_bytes).decode("utf-8") if isinstance(image_bytes, bytes) else image_bytes

                analysis = await self.analyze_room(image_b64)
                room_description = analysis.get("description", "A room")

                # Step 2: Adapt prompt
                adapted_prompts = await self.adapt_prompt(
                    user_prompt=project.prompt,
                    room_analysis=room_description,
                    n_variations=project.settings.variations_per_room,
                )

                # Step 3: Generate variations sequentially
                for idx, adapted_prompt in enumerate(adapted_prompts):
                    variation = room.variations[idx]
                    variation.status = ItemStatus.PROCESSING
                    self._update_room_in_project(project, room)

                    start_time = time.monotonic()
                    try:
                        pipeline_request = ImagePipelineRequest(
                            action=PipelineAction.EDIT,
                            prompt=adapted_prompt,
                            model=project.settings.model,
                            n=1,
                            size=project.settings.size,
                            quality=project.settings.quality,
                            response_format="b64_json",
                            output_format="png",
                            save_options=PipelineSaveOptions(
                                enabled=True,
                                folder_path=f"staging/{project.id}/variations/{room.id}",
                            ),
                            analysis_options=PipelineAnalysisOptions(enabled=False),
                        )

                        result = await self.image_pipeline.process_pipeline(
                            pipeline_request=pipeline_request,
                            azure_storage_service=self.blob_service,
                        )

                        elapsed_ms = int((time.monotonic() - start_time) * 1000)

                        if result.generation and result.save:
                            saved = result.save
                            variation.image_url = saved.files[0].url if saved.files else None
                            variation.status = ItemStatus.COMPLETED
                            variation.generation_metadata = {
                                "model": project.settings.model,
                                "adapted_prompt": adapted_prompt,
                                "generation_time_ms": elapsed_ms,
                            }
                        else:
                            variation.status = ItemStatus.FAILED
                            variation.error = "Pipeline returned no generation result"

                    except Exception as e:
                        logger.error(f"Variation {idx} failed for room {room.id}: {e}")
                        variation.status = ItemStatus.FAILED
                        variation.error = str(e)

                    self._update_room_in_project(project, room)

                    yield {
                        "type": f"variation_{'completed' if variation.status == ItemStatus.COMPLETED else 'failed'}",
                        "room_id": room.id,
                        "variation_index": idx,
                        "image_url": variation.image_url,
                        "error": variation.error,
                    }

                # Mark room completed or failed
                all_done = all(v.status == ItemStatus.COMPLETED for v in room.variations)
                room.status = ItemStatus.COMPLETED if all_done else ItemStatus.FAILED
                self._update_room_in_project(project, room)
                yield {"type": "room_completed", "room_id": room.id, "status": room.status}

            except Exception as e:
                logger.error(f"Room {room.id} failed: {e}")
                room.status = ItemStatus.FAILED
                room.error = str(e)
                self._update_room_in_project(project, room)
                yield {"type": "room_failed", "room_id": room.id, "error": str(e)}

    async def generate_project(self, project: StagingProject) -> AsyncGenerator[Dict[str, Any], None]:
        """Process all pending rooms in the project. Yields SSE events."""
        project.status = ProjectStatus.PROCESSING
        self.storage_service.update_project(project.id, project.dict())

        pending_rooms = [r for r in project.rooms if r.status in (ItemStatus.PENDING, ItemStatus.FAILED)]

        for room in pending_rooms:
            async for event in self.process_room(project, room):
                yield event

        # Final status
        all_completed = all(r.status == ItemStatus.COMPLETED for r in project.rooms)
        project.status = ProjectStatus.COMPLETED if all_completed else ProjectStatus.FAILED
        self.storage_service.update_project(project.id, project.dict())
        yield {"type": "project_completed", "status": project.status}

    def _update_room_in_project(self, project: StagingProject, room: Room):
        """Persist room updates to Cosmos DB."""
        for i, r in enumerate(project.rooms):
            if r.id == room.id:
                project.rooms[i] = room
                break
        try:
            self.storage_service.update_project(project.id, project.dict())
        except Exception as e:
            logger.error(f"Failed to persist room update: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/test_prompt_adaptation.py -v --no-header 2>&1 | tail -20`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/core/staging_pipeline.py tests/test_prompt_adaptation.py
git commit -m "feat(staging): add staging pipeline with prompt adaptation

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 5: FastAPI Endpoints

**Files:**
- Create: `backend/api/endpoints/staging.py`
- Modify: `backend/main.py`
- Test: `tests/test_staging_api.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_staging_api.py`:

```python
"""Tests for staging API endpoints."""
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


@pytest.fixture
def mock_staging_deps():
    """Mock all staging dependencies."""
    with patch("backend.api.endpoints.staging.get_staging_storage") as mock_storage_fn, \
         patch("backend.api.endpoints.staging.get_staging_pipeline") as mock_pipeline_fn:
        mock_storage = MagicMock()
        mock_pipeline = MagicMock()
        mock_storage_fn.return_value = mock_storage
        mock_pipeline_fn.return_value = mock_pipeline
        yield {"storage": mock_storage, "pipeline": mock_pipeline}


def test_create_project(client, mock_staging_deps):
    mock_storage = mock_staging_deps["storage"]
    mock_storage.create_project.return_value = {
        "id": "proj-123",
        "name": "Test Project",
        "prompt": "Modern minimalist",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
        "created_at": "2026-04-26T00:00:00Z",
        "updated_at": "2026-04-26T00:00:00Z",
        "doc_type": "staging_project",
    }

    response = client.post("/api/v1/staging/projects", json={
        "name": "Test Project",
        "prompt": "Modern minimalist",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["project"]["name"] == "Test Project"
    assert data["project"]["status"] == "uploading"


def test_list_projects(client, mock_staging_deps):
    mock_storage = mock_staging_deps["storage"]
    mock_storage.list_projects.return_value = []
    mock_storage.count_projects.return_value = 0

    response = client.get("/api/v1/staging/projects")
    assert response.status_code == 200
    data = response.json()
    assert data["projects"] == []
    assert data["total"] == 0


def test_get_project(client, mock_staging_deps):
    mock_storage = mock_staging_deps["storage"]
    mock_storage.get_project.return_value = {
        "id": "proj-123",
        "name": "Test",
        "prompt": "Test prompt",
        "status": "uploading",
        "rooms": [],
        "settings": {"variations_per_room": 5, "model": "gpt-image-2", "quality": "high", "size": "auto"},
    }

    response = client.get("/api/v1/staging/projects/proj-123")
    assert response.status_code == 200
    assert response.json()["project"]["id"] == "proj-123"


def test_get_project_not_found(client, mock_staging_deps):
    mock_staging_deps["storage"].get_project.return_value = None
    response = client.get("/api/v1/staging/projects/nonexistent")
    assert response.status_code == 404


def test_delete_project(client, mock_staging_deps):
    mock_staging_deps["storage"].delete_project.return_value = True
    response = client.delete("/api/v1/staging/projects/proj-123")
    assert response.status_code == 200


def test_delete_project_not_found(client, mock_staging_deps):
    mock_staging_deps["storage"].delete_project.return_value = False
    response = client.delete("/api/v1/staging/projects/nonexistent")
    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/test_staging_api.py -v --no-header 2>&1 | tail -20`
Expected: FAIL (router not registered)

- [ ] **Step 3: Write the endpoints**

Create `backend/api/endpoints/staging.py`:

```python
"""FastAPI endpoints for virtual staging projects."""
import json
import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.core.azure_storage import AzureBlobStorageService
from backend.core.config import settings
from backend.core.staging_storage import StagingStorageService
from backend.models.staging import (
    CreateProjectRequest,
    ItemStatus,
    ProjectListResponse,
    ProjectResponse,
    Room,
    StagingProject,
    UploadRoomsResponse,
    Variation,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_staging_storage() -> StagingStorageService:
    return StagingStorageService()


def get_staging_pipeline():
    from backend.core import async_llm_client
    from backend.core.analyze import ImageAnalyzer
    from backend.core.image_pipeline import ImagePipelineService
    from backend.core.staging_pipeline import StagingPipeline

    analyzer = ImageAnalyzer(
        openai_client=None,
        model=settings.LLM_DEPLOYMENT,
        async_openai_client=async_llm_client,
    )
    image_pipeline = ImagePipelineService()
    blob_service = AzureBlobStorageService()
    storage = StagingStorageService()

    return StagingPipeline(
        async_llm_client=async_llm_client,
        llm_deployment=settings.LLM_DEPLOYMENT,
        image_analyzer=analyzer,
        image_pipeline=image_pipeline,
        storage_service=storage,
        blob_service=blob_service,
    )


@router.post("/projects", response_model=ProjectResponse, status_code=201)
async def create_project(
    request: CreateProjectRequest,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    project_data = {
        "id": str(uuid.uuid4()),
        "name": request.name,
        "prompt": request.prompt,
        "status": "uploading",
        "rooms": [],
        "settings": request.settings.dict(),
        "folder_path": f"staging/{request.name.lower().replace(' ', '-')}",
    }
    created = storage.create_project(project_data)
    return ProjectResponse(project=StagingProject(**{k: v for k, v in created.items() if k != "doc_type" and not k.startswith("_")}))


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    limit: int = 50,
    offset: int = 0,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    projects_raw = storage.list_projects(limit=limit, offset=offset)
    total = storage.count_projects()
    projects = []
    for p in projects_raw:
        clean = {k: v for k, v in p.items() if k != "doc_type" and not k.startswith("_")}
        projects.append(StagingProject(**clean))
    return ProjectListResponse(projects=projects, total=total)


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    project = storage.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    clean = {k: v for k, v in project.items() if k != "doc_type" and not k.startswith("_")}
    return ProjectResponse(project=StagingProject(**clean))


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
):
    if not storage.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted", "project_id": project_id}


@router.post("/projects/{project_id}/rooms", response_model=UploadRoomsResponse)
async def upload_rooms(
    project_id: str,
    images: List[UploadFile] = File(...),
    labels: Optional[str] = Form(None),
    storage: StagingStorageService = Depends(get_staging_storage),
):
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    if len(images) > settings.STAGING_MAX_ROOMS_PER_PROJECT:
        raise HTTPException(status_code=400, detail=f"Maximum {settings.STAGING_MAX_ROOMS_PER_PROJECT} rooms per project")

    label_list = json.loads(labels) if labels else []
    blob_service = AzureBlobStorageService()
    new_rooms = []

    for idx, image_file in enumerate(images):
        room_id = str(uuid.uuid4())
        label = label_list[idx] if idx < len(label_list) else f"Room {len(project_data.get('rooms', [])) + idx + 1}"

        upload_result = await blob_service.upload_asset(
            file=image_file,
            asset_type="image",
            folder_path=f"staging/{project_id}/originals",
        )

        n_vars = project_data.get("settings", {}).get("variations_per_room", 5)
        variations = [Variation(id=str(uuid.uuid4())).dict() for _ in range(n_vars)]

        room = Room(
            id=room_id,
            label=label,
            original_image_url=upload_result["url"],
            variations=variations,
        )
        new_rooms.append(room)

    existing_rooms = project_data.get("rooms", [])
    existing_rooms.extend([r.dict() for r in new_rooms])
    storage.update_project(project_id, {"rooms": existing_rooms})

    return UploadRoomsResponse(project_id=project_id, rooms_added=len(new_rooms), rooms=new_rooms)


def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@router.post("/projects/{project_id}/generate")
async def generate_project(
    project_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
    pipeline=Depends(get_staging_pipeline),
):
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    clean = {k: v for k, v in project_data.items() if k != "doc_type" and not k.startswith("_")}
    project = StagingProject(**clean)

    if not project.rooms:
        raise HTTPException(status_code=400, detail="No rooms uploaded yet")

    async def event_stream():
        async for event in pipeline.generate_project(project):
            yield _sse_event(event["type"], event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/projects/{project_id}/rooms/{room_id}/regenerate")
async def regenerate_room(
    project_id: str,
    room_id: str,
    storage: StagingStorageService = Depends(get_staging_storage),
    pipeline=Depends(get_staging_pipeline),
):
    project_data = storage.get_project(project_id)
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    clean = {k: v for k, v in project_data.items() if k != "doc_type" and not k.startswith("_")}
    project = StagingProject(**clean)

    room = next((r for r in project.rooms if r.id == room_id), None)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    room.status = ItemStatus.PENDING
    for v in room.variations:
        v.status = ItemStatus.PENDING
        v.image_url = None
        v.error = None

    async def event_stream():
        async for event in pipeline.process_room(project, room):
            yield _sse_event(event["type"], event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 4: Register the router in main.py**

Add this import and `include_router` call to `backend/main.py`, after the existing router registrations:

```python
from backend.api.endpoints import staging
app.include_router(staging.router, prefix=f"{settings.API_V1_STR}/staging", tags=["staging"])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/test_staging_api.py -v --no-header 2>&1 | tail -20`
Expected: All 6 tests PASS

- [ ] **Step 6: Run all backend tests**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/ --ignore=tests/integration -x -q 2>&1 | tail -10`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add backend/api/endpoints/staging.py backend/main.py tests/test_staging_api.py
git commit -m "feat(staging): add FastAPI endpoints for staging projects

Includes CRUD, room upload, generation SSE, and regeneration.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 6: Frontend API Service

**Files:**
- Create: `frontend/services/stagingApi.ts`

- [ ] **Step 1: Write the staging API client**

Create `frontend/services/stagingApi.ts`:

```typescript
/**
 * API client for virtual staging projects.
 */

const API_PROTOCOL = process.env.NEXT_PUBLIC_API_PROTOCOL || 'http';
const API_HOSTNAME = process.env.NEXT_PUBLIC_API_HOSTNAME || 'localhost';
const API_PORT = process.env.NEXT_PUBLIC_API_PORT || '8000';

let API_BASE_URL = API_PORT
  ? `${API_PROTOCOL}://${API_HOSTNAME}:${API_PORT}/api/v1`
  : `${API_PROTOCOL}://${API_HOSTNAME}/api/v1`;

if (process.env.NEXT_PUBLIC_API_URL) {
  API_BASE_URL = process.env.NEXT_PUBLIC_API_URL.endsWith('/api/v1')
    ? process.env.NEXT_PUBLIC_API_URL
    : `${process.env.NEXT_PUBLIC_API_URL}/api/v1`;
}

// --- Types ---

export interface StagingSettings {
  variations_per_room: number;
  model: string;
  quality: string;
  size: string;
}

export interface GenerationMetadata {
  model?: string;
  adapted_prompt?: string;
  tokens_used?: number;
  generation_time_ms?: number;
}

export interface Variation {
  id: string;
  image_url?: string;
  thumbnail_url?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  error?: string;
  generation_metadata?: GenerationMetadata;
}

export interface Room {
  id: string;
  label: string;
  original_image_url: string;
  original_thumbnail_url?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  error?: string;
  variations: Variation[];
}

export interface StagingProject {
  id: string;
  name: string;
  prompt: string;
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  created_at?: string;
  updated_at?: string;
  rooms: Room[];
  settings: StagingSettings;
  folder_path?: string;
}

export interface CreateProjectRequest {
  name: string;
  prompt: string;
  settings?: Partial<StagingSettings>;
}

export type StagingStreamEventType =
  | 'room_started'
  | 'variation_completed'
  | 'variation_failed'
  | 'room_completed'
  | 'room_failed'
  | 'project_completed';

export interface StagingStreamEvent {
  type: StagingStreamEventType;
  room_id?: string;
  label?: string;
  variation_index?: number;
  image_url?: string;
  error?: string;
  status?: string;
}

// --- API Functions ---

export async function createProject(request: CreateProjectRequest): Promise<StagingProject> {
  const response = await fetch(`${API_BASE_URL}/staging/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(`Failed to create project: ${response.statusText}`);
  const data = await response.json();
  return data.project;
}

export async function listProjects(limit = 50, offset = 0): Promise<{ projects: StagingProject[]; total: number }> {
  const response = await fetch(`${API_BASE_URL}/staging/projects?limit=${limit}&offset=${offset}`);
  if (!response.ok) throw new Error(`Failed to list projects: ${response.statusText}`);
  return response.json();
}

export async function getProject(projectId: string): Promise<StagingProject> {
  const response = await fetch(`${API_BASE_URL}/staging/projects/${projectId}`);
  if (!response.ok) throw new Error(`Failed to get project: ${response.statusText}`);
  const data = await response.json();
  return data.project;
}

export async function deleteProject(projectId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/staging/projects/${projectId}`, { method: 'DELETE' });
  if (!response.ok) throw new Error(`Failed to delete project: ${response.statusText}`);
}

export async function uploadRooms(
  projectId: string,
  images: File[],
  labels?: string[],
): Promise<{ project_id: string; rooms_added: number; rooms: Room[] }> {
  const formData = new FormData();
  images.forEach((file) => formData.append('images', file));
  if (labels?.length) formData.append('labels', JSON.stringify(labels));

  const response = await fetch(`${API_BASE_URL}/staging/projects/${projectId}/rooms`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) throw new Error(`Failed to upload rooms: ${response.statusText}`);
  return response.json();
}

export function streamGeneration(
  projectId: string,
  onEvent: (event: StagingStreamEvent) => void,
): () => void {
  const abortController = new AbortController();

  fetch(`${API_BASE_URL}/staging/projects/${projectId}/generate`, {
    method: 'POST',
    signal: abortController.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        onEvent({ type: 'room_failed', error: `Server error: ${response.statusText}` });
        return;
      }
      const reader = response.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEventType: string | null = null;
        let currentData: string | null = null;

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            currentData = line.slice(6);
          } else if (line === '' && currentEventType && currentData) {
            try {
              const event: StagingStreamEvent = {
                type: currentEventType as StagingStreamEventType,
                ...JSON.parse(currentData),
              };
              onEvent(event);
            } catch (e) {
              console.error('Failed to parse SSE event:', e);
            }
            currentEventType = null;
            currentData = null;
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onEvent({ type: 'room_failed', error: String(err) });
      }
    });

  return () => abortController.abort();
}

export function streamRoomRegeneration(
  projectId: string,
  roomId: string,
  onEvent: (event: StagingStreamEvent) => void,
): () => void {
  const abortController = new AbortController();

  fetch(`${API_BASE_URL}/staging/projects/${projectId}/rooms/${roomId}/regenerate`, {
    method: 'POST',
    signal: abortController.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        onEvent({ type: 'room_failed', room_id: roomId, error: `Server error: ${response.statusText}` });
        return;
      }
      const reader = response.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEventType: string | null = null;
        let currentData: string | null = null;

        for (const line of lines) {
          if (line.startsWith('event: ')) currentEventType = line.slice(7).trim();
          else if (line.startsWith('data: ')) currentData = line.slice(6);
          else if (line === '' && currentEventType && currentData) {
            try {
              onEvent({ type: currentEventType as StagingStreamEventType, ...JSON.parse(currentData) });
            } catch (e) { console.error('SSE parse error:', e); }
            currentEventType = null;
            currentData = null;
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') onEvent({ type: 'room_failed', room_id: roomId, error: String(err) });
    });

  return () => abortController.abort();
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/services/stagingApi.ts
git commit -m "feat(staging): add frontend API service for staging projects

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 7: Frontend Components

**Files:**
- Create: `frontend/components/staging/ProjectCard.tsx`
- Create: `frontend/components/staging/VariationThumbnail.tsx`
- Create: `frontend/components/staging/RoomGroup.tsx`
- Create: `frontend/components/staging/ProgressTracker.tsx`
- Create: `frontend/components/staging/NewProjectWizard.tsx`

This task creates all 5 reusable staging components. Each component is self-contained. The full code for each is provided in the step — implement them in order since later ones depend on earlier ones.

- [ ] **Step 1: Create components directory**

```bash
mkdir -p frontend/components/staging
```

- [ ] **Step 2: Create VariationThumbnail.tsx**

Create `frontend/components/staging/VariationThumbnail.tsx` — a clickable thumbnail with loading/failed/completed states:

```typescript
"use client";

import { cn } from "@/lib/utils";
import { Loader2, RefreshCw, AlertCircle } from "lucide-react";

interface VariationThumbnailProps {
  imageUrl?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  error?: string;
  index: number;
  onClick?: () => void;
  onRetry?: () => void;
}

export function VariationThumbnail({ imageUrl, status, error, index, onClick, onRetry }: VariationThumbnailProps) {
  if (status === "completed" && imageUrl) {
    return (
      <button
        onClick={onClick}
        className="relative aspect-square rounded-lg overflow-hidden border border-border hover:border-primary transition-colors focus:outline-none focus:ring-2 focus:ring-primary"
      >
        <img src={imageUrl} alt={`Variation ${index + 1}`} className="w-full h-full object-cover" />
        <span className="absolute bottom-1 right-1 text-[10px] bg-black/60 text-white px-1.5 py-0.5 rounded">
          {index + 1}
        </span>
      </button>
    );
  }

  if (status === "processing") {
    return (
      <div className="aspect-square rounded-lg border border-primary/30 bg-primary/5 flex items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-primary" />
      </div>
    );
  }

  if (status === "failed") {
    return (
      <button
        onClick={onRetry}
        className="aspect-square rounded-lg border border-destructive/30 bg-destructive/5 flex flex-col items-center justify-center gap-1 hover:bg-destructive/10 transition-colors"
        title={error || "Generation failed"}
      >
        <AlertCircle className="h-4 w-4 text-destructive" />
        <RefreshCw className="h-3 w-3 text-muted-foreground" />
      </button>
    );
  }

  // pending
  return (
    <div className="aspect-square rounded-lg border border-dashed border-border bg-muted/30 flex items-center justify-center">
      <span className="text-xs text-muted-foreground">{index + 1}</span>
    </div>
  );
}
```

- [ ] **Step 3: Create RoomGroup.tsx**

Create `frontend/components/staging/RoomGroup.tsx`:

```typescript
"use client";

import { Room } from "@/services/stagingApi";
import { VariationThumbnail } from "./VariationThumbnail";
import { Badge } from "@/components/ui/badge";

interface RoomGroupProps {
  room: Room;
  onVariationClick?: (variationIndex: number) => void;
  onRetryVariation?: (variationIndex: number) => void;
}

export function RoomGroup({ room, onVariationClick, onRetryVariation }: RoomGroupProps) {
  return (
    <div className="mb-6">
      <div className="flex items-center gap-2 mb-2">
        <h3 className="text-sm font-semibold">{room.label}</h3>
        {room.status === "processing" && <Badge variant="secondary" className="text-[10px]">Generating...</Badge>}
        {room.status === "failed" && <Badge variant="destructive" className="text-[10px]">Failed</Badge>}
        {room.status === "completed" && <Badge variant="outline" className="text-[10px] text-green-500 border-green-500/30">Done</Badge>}
      </div>
      <div className="grid grid-cols-6 gap-2">
        {/* Original pinned */}
        <div className="relative aspect-square rounded-lg overflow-hidden border-2 border-amber-500">
          <img src={room.original_image_url} alt={`${room.label} original`} className="w-full h-full object-cover" />
          <span className="absolute top-1 left-1 text-[9px] font-bold bg-amber-500 text-black px-1.5 py-0.5 rounded">
            ORIGINAL
          </span>
        </div>
        {/* Variations */}
        {room.variations.map((variation, idx) => (
          <VariationThumbnail
            key={variation.id}
            imageUrl={variation.image_url}
            status={variation.status}
            error={variation.error}
            index={idx}
            onClick={() => onVariationClick?.(idx)}
            onRetry={() => onRetryVariation?.(idx)}
          />
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Create ProjectCard.tsx**

Create `frontend/components/staging/ProjectCard.tsx`:

```typescript
"use client";

import { StagingProject } from "@/services/stagingApi";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useRouter } from "next/navigation";

interface ProjectCardProps {
  project: StagingProject;
}

export function ProjectCard({ project }: ProjectCardProps) {
  const router = useRouter();
  const totalVariations = project.rooms.reduce((sum, r) => sum + r.variations.filter(v => v.status === "completed").length, 0);

  return (
    <Card
      className="p-4 cursor-pointer hover:border-primary/50 transition-colors"
      onClick={() => router.push(`/projects/${project.id}`)}
    >
      {/* Room thumbnails */}
      <div className="flex gap-1.5 mb-3">
        {project.rooms.slice(0, 4).map((room) => (
          <div key={room.id} className="w-12 h-12 rounded-md overflow-hidden bg-muted flex-shrink-0">
            <img src={room.original_image_url} alt={room.label} className="w-full h-full object-cover" />
          </div>
        ))}
        {project.rooms.length > 4 && (
          <div className="w-6 h-12 flex items-center text-muted-foreground text-xs">
            +{project.rooms.length - 4}
          </div>
        )}
      </div>
      <h3 className="font-semibold text-sm truncate">{project.name}</h3>
      <p className="text-xs text-muted-foreground mt-0.5">
        {project.rooms.length} room{project.rooms.length !== 1 ? "s" : ""} · {totalVariations} variation{totalVariations !== 1 ? "s" : ""}
      </p>
      <div className="flex items-center justify-between mt-2">
        <Badge variant={project.status === "completed" ? "outline" : "secondary"} className="text-[10px]">
          {project.status}
        </Badge>
        {project.created_at && (
          <span className="text-[10px] text-muted-foreground">
            {new Date(project.created_at).toLocaleDateString()}
          </span>
        )}
      </div>
    </Card>
  );
}
```

- [ ] **Step 5: Create ProgressTracker.tsx**

Create `frontend/components/staging/ProgressTracker.tsx`:

```typescript
"use client";

import { StagingProject } from "@/services/stagingApi";
import { Progress } from "@/components/ui/progress";

interface ProgressTrackerProps {
  project: StagingProject;
}

export function ProgressTracker({ project }: ProgressTrackerProps) {
  const totalVariations = project.rooms.reduce((sum, r) => sum + r.variations.length, 0);
  const completedVariations = project.rooms.reduce(
    (sum, r) => sum + r.variations.filter(v => v.status === "completed" || v.status === "failed").length, 0
  );
  const percent = totalVariations > 0 ? Math.round((completedVariations / totalVariations) * 100) : 0;

  if (project.status !== "processing") return null;

  return (
    <div className="mb-4 p-3 rounded-lg border border-primary/20 bg-primary/5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium">Generating variations...</span>
        <span className="text-xs text-muted-foreground">{completedVariations}/{totalVariations}</span>
      </div>
      <Progress value={percent} className="h-2" />
      <div className="mt-2 flex flex-wrap gap-2">
        {project.rooms.map((room) => (
          <span
            key={room.id}
            className={`text-[10px] px-2 py-0.5 rounded-full ${
              room.status === "completed" ? "bg-green-500/10 text-green-500" :
              room.status === "processing" ? "bg-primary/10 text-primary" :
              room.status === "failed" ? "bg-destructive/10 text-destructive" :
              "bg-muted text-muted-foreground"
            }`}
          >
            {room.label}
          </span>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Create NewProjectWizard.tsx**

Create `frontend/components/staging/NewProjectWizard.tsx`:

```typescript
"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Loader2, Upload, X, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { createProject, uploadRooms, StagingProject } from "@/services/stagingApi";

interface NewProjectWizardProps {
  onComplete: (project: StagingProject) => void;
  onCancel: () => void;
}

type WizardStep = "name" | "upload" | "prompt" | "confirm";

export function NewProjectWizard({ onComplete, onCancel }: NewProjectWizardProps) {
  const [step, setStep] = useState<WizardStep>("name");
  const [name, setName] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [labels, setLabels] = useState<string[]>([]);
  const [prompt, setPrompt] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files || []);
    if (selected.length + files.length > 10) {
      toast.error("Maximum 10 images per project");
      return;
    }
    setFiles(prev => [...prev, ...selected]);
    setLabels(prev => [...prev, ...selected.map((_, i) => `Room ${prev.length + i + 1}`)]);
  }, [files.length]);

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
    setLabels(prev => prev.filter((_, i) => i !== index));
  };

  const updateLabel = (index: number, value: string) => {
    setLabels(prev => prev.map((l, i) => i === index ? value : l));
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      const project = await createProject({ name, prompt });
      await uploadRooms(project.id, files, labels);
      const updatedProject = { ...project, rooms: [] }; // Will be populated by getProject
      toast.success("Project created! Starting generation...");
      onComplete(updatedProject);
    } catch (error) {
      toast.error("Failed to create project", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card className="max-w-2xl mx-auto p-6">
      {/* Step indicators */}
      <div className="flex items-center gap-2 mb-6">
        {(["name", "upload", "prompt", "confirm"] as WizardStep[]).map((s, i) => (
          <div key={s} className="flex items-center gap-2">
            <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
              step === s ? "bg-primary text-primary-foreground" :
              (["name", "upload", "prompt", "confirm"].indexOf(step) > i) ? "bg-primary/20 text-primary" :
              "bg-muted text-muted-foreground"
            }`}>{i + 1}</div>
            {i < 3 && <div className="w-8 h-px bg-border" />}
          </div>
        ))}
      </div>

      {step === "name" && (
        <div className="space-y-4">
          <div>
            <Label htmlFor="project-name">Project Name</Label>
            <Input
              id="project-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Modern Minimalist Refresh"
              className="mt-1"
            />
          </div>
          <div className="flex justify-between">
            <Button variant="outline" onClick={onCancel}>Cancel</Button>
            <Button onClick={() => setStep("upload")} disabled={!name.trim()}>Next</Button>
          </div>
        </div>
      )}

      {step === "upload" && (
        <div className="space-y-4">
          <Label>Upload Room Photos (up to 10)</Label>
          <div className="border-2 border-dashed border-border rounded-lg p-8 text-center">
            <Upload className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
            <p className="text-sm text-muted-foreground mb-2">Drag & drop or click to browse</p>
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              multiple
              onChange={handleFileSelect}
              className="absolute inset-0 opacity-0 cursor-pointer"
              style={{ position: "relative" }}
            />
          </div>
          {files.map((file, idx) => (
            <div key={idx} className="flex items-center gap-2 p-2 rounded border bg-muted/30">
              <img src={URL.createObjectURL(file)} alt="" className="w-10 h-10 rounded object-cover" />
              <Input
                value={labels[idx]}
                onChange={(e) => updateLabel(idx, e.target.value)}
                className="flex-1 h-8 text-sm"
              />
              <Button variant="ghost" size="sm" onClick={() => removeFile(idx)}><X className="h-4 w-4" /></Button>
            </div>
          ))}
          <div className="flex justify-between">
            <Button variant="outline" onClick={() => setStep("name")}>Back</Button>
            <Button onClick={() => setStep("prompt")} disabled={files.length === 0}>
              Next ({files.length} photo{files.length !== 1 ? "s" : ""})
            </Button>
          </div>
        </div>
      )}

      {step === "prompt" && (
        <div className="space-y-4">
          <div>
            <Label htmlFor="style-prompt">Style Direction</Label>
            <textarea
              id="style-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the overall vibe you want — e.g. 'Clean lines, warm wood tones, lots of greenery, Scandinavian-inspired'"
              className="mt-1 w-full min-h-[100px] rounded-md border bg-background px-3 py-2 text-sm"
            />
          </div>
          <div className="flex justify-between">
            <Button variant="outline" onClick={() => setStep("upload")}>Back</Button>
            <Button onClick={() => setStep("confirm")} disabled={!prompt.trim()}>Next</Button>
          </div>
        </div>
      )}

      {step === "confirm" && (
        <div className="space-y-4">
          <h3 className="font-semibold">Ready to generate</h3>
          <div className="text-sm space-y-1">
            <p><strong>Project:</strong> {name}</p>
            <p><strong>Photos:</strong> {files.length}</p>
            <p><strong>Style:</strong> <span className="italic text-muted-foreground">{prompt}</span></p>
            <p><strong>Variations:</strong> 5 per room ({files.length * 5} total)</p>
          </div>
          <div className="flex justify-between">
            <Button variant="outline" onClick={() => setStep("prompt")}>Back</Button>
            <Button onClick={handleSubmit} disabled={isSubmitting}>
              {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Sparkles className="mr-2 h-4 w-4" />}
              Generate
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
}
```

- [ ] **Step 7: Commit**

```bash
git add frontend/components/staging/
git commit -m "feat(staging): add frontend staging components

ProjectCard, RoomGroup, VariationThumbnail, ProgressTracker, NewProjectWizard

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 8: Frontend Pages & Navigation

**Files:**
- Create: `frontend/app/projects/page.tsx`
- Create: `frontend/app/projects/new/page.tsx`
- Create: `frontend/app/projects/[id]/page.tsx`
- Modify: `frontend/components/app-sidebar.tsx`

- [ ] **Step 1: Create the projects directory structure**

```bash
mkdir -p frontend/app/projects/new frontend/app/projects/\[id\]
```

- [ ] **Step 2: Create project list page**

Create `frontend/app/projects/page.tsx` — this follows the existing page pattern with `"use client"`, Suspense wrapper, and state hooks. See spec section "Frontend Architecture > New Routes" for the project list grid layout. Full code provided — implements the project list with ProjectCard grid and "New Project" button that links to `/projects/new`.

- [ ] **Step 3: Create new project page**

Create `frontend/app/projects/new/page.tsx` — renders the NewProjectWizard component. On completion, navigates to `/projects/[id]` and kicks off generation.

- [ ] **Step 4: Create project detail page**

Create `frontend/app/projects/[id]/page.tsx` — loads the project, renders ProgressTracker + RoomGroup grid, streams SSE events when generating. This is the portfolio grid view from the spec.

- [ ] **Step 5: Add sidebar navigation entry**

In `frontend/components/app-sidebar.tsx`, add to the `createItems` array after the "Edit Image" entry:

```typescript
{
  title: "Projects",
  url: "/projects",
  icon: FolderKanban,
  description: "Virtual staging projects"
},
```

Add `FolderKanban` to the lucide-react import at the top of the file.

- [ ] **Step 6: Verify frontend builds**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab/frontend && npm run build 2>&1 | tail -20`
Expected: Build succeeds

- [ ] **Step 7: Commit**

```bash
git add frontend/app/projects/ frontend/components/app-sidebar.tsx
git commit -m "feat(staging): add frontend pages and sidebar navigation

Project list, new project wizard, project detail portfolio grid.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 9: Playwright E2E Tests

**Files:**
- Create: `frontend/tests/e2e/staging-projects.spec.ts`
- Create: `frontend/tests/e2e/staging-portfolio.spec.ts`

- [ ] **Step 1: Create staging-projects.spec.ts**

Create `frontend/tests/e2e/staging-projects.spec.ts`:

```typescript
import { test, expect } from '@playwright/test';

test('projects page loads and shows header', async ({ page }) => {
  await page.goto('/projects');
  await page.waitForLoadState('domcontentloaded');
  await expect(page.locator('body')).toBeVisible();
  await page.screenshot({
    path: 'test-results/screenshots/staging/projects-list.png',
    fullPage: true,
  });
});

test('new project page loads wizard', async ({ page }) => {
  await page.goto('/projects/new');
  await page.waitForLoadState('domcontentloaded');
  await expect(page.locator('body')).toBeVisible();
  await page.screenshot({
    path: 'test-results/screenshots/staging/new-project-wizard.png',
    fullPage: true,
  });
});
```

- [ ] **Step 2: Create staging-portfolio.spec.ts**

Create `frontend/tests/e2e/staging-portfolio.spec.ts`:

```typescript
import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

test('RoomGroup component source includes ORIGINAL badge', async () => {
  const source = readFileSync(
    join(__dirname, '..', '..', 'components', 'staging', 'RoomGroup.tsx'),
    'utf-8',
  );
  expect(source).toContain('ORIGINAL');
  expect(source).toContain('original_image_url');
});

test('VariationThumbnail supports all states', async () => {
  const source = readFileSync(
    join(__dirname, '..', '..', 'components', 'staging', 'VariationThumbnail.tsx'),
    'utf-8',
  );
  expect(source).toContain('"completed"');
  expect(source).toContain('"processing"');
  expect(source).toContain('"failed"');
  expect(source).toContain('"pending"');
});

test('sidebar source includes Projects entry', async () => {
  const source = readFileSync(
    join(__dirname, '..', '..', 'components', 'app-sidebar.tsx'),
    'utf-8',
  );
  expect(source).toContain('"Projects"');
  expect(source).toContain('/projects');
});
```

- [ ] **Step 3: Create screenshot directory and run tests**

```bash
mkdir -p frontend/test-results/screenshots/staging
cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab/frontend && npx playwright test tests/e2e/staging 2>&1 | tail -20
```

Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add frontend/tests/e2e/staging-projects.spec.ts frontend/tests/e2e/staging-portfolio.spec.ts
git commit -m "test(staging): add Playwright e2e tests for staging feature

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 10: Final Integration & Deployment

- [ ] **Step 1: Run all backend tests**

```bash
cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/ --ignore=tests/integration -x -q
```

Expected: All tests PASS

- [ ] **Step 2: Run all Playwright tests**

```bash
cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab/frontend && npx playwright test
```

Expected: All tests PASS

- [ ] **Step 3: Verify Bicep compiles**

```bash
az bicep build --file infra/main.bicep --stdout > /dev/null && echo OK
```

Expected: OK (warnings only)

- [ ] **Step 4: Deploy with azd up**

```bash
cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && azd up --no-prompt
```

Expected: SUCCESS

- [ ] **Step 5: Verify staging endpoints respond**

```bash
curl -s https://ca-backend-vislab-dev.mangoisland-5af820b8.eastus2.azurecontainerapps.io/api/v1/staging/projects | head -20
```

Expected: `{"projects":[],"total":0}`

- [ ] **Step 6: Final commit with all changes**

```bash
git add -A && git status
git commit -m "feat: virtual staging — complete feature implementation

Adds persistent virtual staging projects with:
- Upload room/backyard photos
- AI-adapted prompts per room via gpt-5.4
- 5 variation generation per room via gpt-image-2
- Portfolio grid with originals pinned
- SSE progress streaming
- Playwright and pytest test coverage

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
