# Virtual Staging Feature — Design Spec

## Overview

Add a "Virtual Staging" feature to Visionary Lab that lets users upload photos of rooms and outdoor spaces, provide a single styling prompt, and receive multiple AI-generated variations showing different decoration, furniture, and landscaping ideas. Originals are preserved — the AI adds to the scene rather than replacing it.

## Core Concept

Upload room/backyard photos → write one prompt describing the overall vibe → AI adapts that prompt per room and generates 5 variations each → browse results in a portfolio grid with originals pinned → projects persist for later revisiting.

## Architecture: Hybrid Approach

New `StagingProject` Cosmos DB model for persistence and project management, but delegates actual image generation to the existing `image_pipeline.process_pipeline()` with EDIT action. This reuses the battle-tested pipeline while adding a clean project layer on top.

## Data Model

### StagingProject (Cosmos DB)

```json
{
  "id": "uuid",
  "name": "string",
  "prompt": "string — the user's overall styling direction",
  "status": "uploading | processing | completed | failed",
  "created_at": "datetime",
  "updated_at": "datetime",
  "rooms": [
    {
      "id": "uuid",
      "label": "string — e.g. Living Room, Backyard (auto-detected or user-set)",
      "original_image_url": "string — blob storage URL",
      "original_thumbnail_url": "string",
      "status": "pending | processing | completed | failed",
      "error": "string | null — failure reason if status is failed",
      "variations": [
        {
          "id": "uuid",
          "image_url": "string — blob storage URL",
          "thumbnail_url": "string",
          "status": "pending | processing | completed | failed",
          "error": "string | null",
          "generation_metadata": {
            "model": "string — e.g. gpt-image-2",
            "adapted_prompt": "string — LLM-adapted prompt for this room",
            "tokens_used": "number",
            "generation_time_ms": "number"
          }
        }
      ]
    }
  ],
  "settings": {
    "variations_per_room": 5,
    "model": "gpt-image-2",
    "quality": "high",
    "size": "auto"
  },
  "folder_path": "string — gallery folder for saved images"
}
```

### Storage

- Original uploaded images stored in Azure Blob Storage under `staging/{project_id}/originals/`
- Generated variations stored under `staging/{project_id}/variations/{room_id}/`
- Project metadata in Cosmos DB (same database as existing gallery metadata), in a `staging-projects` container with partition key `/id` (each project is its own partition)

## API Endpoints

### Project CRUD

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/staging/projects` | Create project (name, prompt, settings) |
| `GET` | `/api/v1/staging/projects` | List user's projects |
| `GET` | `/api/v1/staging/projects/{id}` | Get project with all rooms and variations |
| `DELETE` | `/api/v1/staging/projects/{id}` | Delete project and all associated blobs |

### Room Management

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/staging/projects/{id}/rooms` | Upload room photos (multipart, up to 10 images) |
| `DELETE` | `/api/v1/staging/projects/{id}/rooms/{room_id}` | Remove a room from the project |

### Generation

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/staging/projects/{id}/generate` | Start generation for all pending rooms (returns immediately, SSE for progress) |
| `GET` | `/api/v1/staging/projects/{id}/progress` | SSE stream for generation progress |
| `POST` | `/api/v1/staging/projects/{id}/rooms/{room_id}/regenerate` | Re-run generation for a specific room |

### Request/Response Examples

**Create project:**
```json
POST /api/v1/staging/projects
{
  "name": "Modern Minimalist Refresh",
  "prompt": "Clean lines, warm wood tones, lots of greenery, Scandinavian-inspired",
  "settings": {
    "variations_per_room": 5,
    "model": "gpt-image-2",
    "quality": "high"
  }
}
```

**Upload rooms:**
```
POST /api/v1/staging/projects/{id}/rooms
Content-Type: multipart/form-data
  - images[]: File[] (up to 10 JPG/PNG/WebP, max 25MB each)
  - labels[]: string[] (optional — "Living Room", "Backyard", etc.)
```

**SSE progress events:**
```json
{"type": "room_started", "room_id": "abc", "label": "Living Room"}
{"type": "variation_completed", "room_id": "abc", "variation_index": 0, "image_url": "https://..."}
{"type": "variation_failed", "room_id": "abc", "variation_index": 2, "error": "Content filter triggered"}
{"type": "room_completed", "room_id": "abc"}
{"type": "project_completed"}
```

## Generation Pipeline

### Prompt Adaptation Flow

For each room, three steps happen:

1. **Image analysis** — Reuses the existing `ImageAnalyzer` class to describe what's in the uploaded photo (furniture, layout, colors, outdoor features). This gives the LLM context about the space.

2. **Prompt adaptation** — A single gpt-5.4 call per room. The system prompt instructs the LLM to combine the user's master prompt with the room analysis, producing 5 distinct variation prompts. Each variation interprets the style differently while respecting the existing room structure.

   System prompt template:
   ```
   You are a virtual staging assistant. The user wants to visualize decorating ideas.

   ROOM ANALYSIS: {image_analysis_output}
   USER'S STYLE DIRECTION: {user_prompt}

   Generate {n} distinct variation prompts for an image editing model. Each prompt should:
   - ADD items to the existing scene (furniture, decor, plants, landscaping)
   - NOT remove or replace existing structures
   - Interpret the user's style direction differently in each variation
   - Be specific about what to add and where to place it
   - Reference the existing room features from the analysis

   Return as JSON array of strings.
   ```

3. **Image generation** — Each adapted prompt is sent to the existing `image_pipeline.process_pipeline()` with:
   - `action: "EDIT"`
   - `source_images: [original_room_photo]`
   - `prompt: adapted_prompt_for_this_variation`
   - `model: project.settings.model` (gpt-image-2)

### Concurrency Model

- Rooms are processed in parallel using `asyncio.gather` (up to 3 concurrent rooms to respect rate limits)
- Variations within a room are sequential to avoid rate-limit bursts
- A semaphore controls max concurrent API calls

### Progress Reporting

The `/generate` endpoint returns `202 Accepted` immediately. Progress is reported via SSE on the `/progress` endpoint. The Cosmos DB document is updated after each variation completes, so polling the project endpoint also works as a fallback.

## Frontend Architecture

### New Routes

| Route | Page | Purpose |
|-------|------|---------|
| `/projects` | `app/projects/page.tsx` | Project list — grid of project cards |
| `/projects/new` | `app/projects/new/page.tsx` | New project wizard (name → upload → prompt → generate) |
| `/projects/[id]` | `app/projects/[id]/page.tsx` | Project detail — portfolio grid |

### New Components (`frontend/components/staging/`)

| Component | Purpose |
|-----------|---------|
| `ProjectCard.tsx` | Card for project list — shows room thumbnails, name, counts |
| `RoomGroup.tsx` | A row in the portfolio grid — original pinned left + variation thumbnails |
| `VariationThumbnail.tsx` | Clickable thumbnail with loading/failed/completed states |
| `NewProjectWizard.tsx` | Multi-step form: name → upload images → write prompt → confirm |
| `ProgressTracker.tsx` | SSE-driven progress indicator showing per-room generation status |

### Navigation

New "Projects" entry in the existing sidebar (`app-sidebar.tsx`), positioned after "Edit Image" and before "Video". Badge shows "NEW" initially.

### Portfolio Grid Layout

- Each room is a horizontal row
- First item in each row: original photo with gold "ORIGINAL" badge border
- Remaining items: variation thumbnails
- In-progress variations show a spinner; pending ones show dashed placeholder
- Failed variations show a retry button
- Click any image to open full-size in a lightbox (reuse existing `ImageDetailView`)

### New Project Wizard Flow

1. **Name step** — Text input for project name
2. **Upload step** — Multi-image uploader (drag & drop zone, up to 10 images). Each image gets an auto-detected or editable label.
3. **Prompt step** — Textarea for the master styling prompt. Optional: "Enhance prompt" button using existing `/images/prompt/enhance` endpoint.
4. **Confirm & Generate** — Summary of rooms + prompt + settings → "Generate" button kicks off the pipeline.

## Error Handling

| Scenario | Handling |
|----------|----------|
| One variation fails (content filter, timeout) | Mark variation `"failed"` with reason. UI shows retry button. Other variations continue. |
| Entire room fails (all variations fail) | Mark room `"failed"`. Show "Retry Room" button. Other rooms unaffected. |
| Upload too large | Client-side validation: 25MB per image, 10 images max. Backend validates too with 413 response. |
| Rate limiting from Azure | Exponential backoff with 3 retries per variation. If still failing, mark as `"failed"` with "rate limited" reason. |
| Generation interrupted (server restart) | On backend startup, scan for projects with `status: "processing"` and resume incomplete rooms. |
| SSE connection drops | Frontend reconnects automatically with `EventSource`. Polls project endpoint as fallback. |
| Invalid image format | Reject at upload time with clear error message. Accept JPG, PNG, WebP only. |

## Testing Strategy

### Backend (pytest)

| Test file | Coverage |
|-----------|----------|
| `tests/test_staging_models.py` | Pydantic model validation — StagingProject, Room, Variation schemas |
| `tests/test_staging_api.py` | Endpoint tests — create project, upload rooms, trigger generation, list/get, delete |
| `tests/test_prompt_adaptation.py` | Mock LLM calls, verify adapted prompts incorporate room analysis + user style direction |

All tests mock the LLM and image pipeline — no live AI calls in tests.

### Frontend (Playwright)

| Test file | Coverage |
|-----------|----------|
| `tests/e2e/staging-projects.spec.ts` | Navigate to /projects, verify page loads, create new project wizard flow |
| `tests/e2e/staging-portfolio.spec.ts` | Mock API responses, verify portfolio grid renders rooms with originals pinned, variation thumbnails appear |

Screenshots stored in `frontend/test-results/screenshots/staging/`.

## Files to Create/Modify

### New Files

**Backend:**
- `backend/models/staging.py` — Pydantic models (StagingProject, Room, Variation, CreateProjectRequest, etc.)
- `backend/api/endpoints/staging.py` — FastAPI router with all staging endpoints
- `backend/core/staging_pipeline.py` — Orchestrator: prompt adaptation + fan-out to image pipeline
- `backend/core/staging_storage.py` — Cosmos DB CRUD for staging projects

**Frontend:**
- `frontend/app/projects/page.tsx` — Project list page
- `frontend/app/projects/new/page.tsx` — New project wizard
- `frontend/app/projects/[id]/page.tsx` — Project detail (portfolio grid)
- `frontend/components/staging/ProjectCard.tsx`
- `frontend/components/staging/RoomGroup.tsx`
- `frontend/components/staging/VariationThumbnail.tsx`
- `frontend/components/staging/NewProjectWizard.tsx`
- `frontend/components/staging/ProgressTracker.tsx`
- `frontend/services/stagingApi.ts` — API client functions for staging endpoints

**Tests:**
- `tests/test_staging_models.py`
- `tests/test_staging_api.py`
- `tests/test_prompt_adaptation.py`
- `frontend/tests/e2e/staging-projects.spec.ts`
- `frontend/tests/e2e/staging-portfolio.spec.ts`

### Modified Files

- `frontend/components/app-sidebar.tsx` — Add "Projects" navigation entry
- `backend/api/router.py` (or equivalent) — Register staging router
- `backend/core/config.py` — Add any staging-specific settings (e.g., max rooms per project)
