# Edit Existing Project — Design Spec

**Date:** 2026-04-29
**Status:** Draft (pending user review)

## Problem

Today, virtual staging projects in Visionary Lab can be created through a 5-step wizard but cannot be edited afterwards. Once a project exists, the only mutating actions are `regenerate`, `reset`, and `delete`. Users have no way to refine the prompt, adjust the design brief, change generation settings, swap room images, or revisit the AI design chat after the project is created.

This spec adds a comprehensive edit experience to the project detail page, covering every field on a project, plus an explicit version history with revert.

## Goals

- All project fields editable after creation: name, prompt, design brief (plant palette, placement guide, preserve elements, per-image notes), generation settings, room labels, room images (add/remove), and the AI design chat (resumable).
- Existing variations are preserved across edits — they only clear when the user explicitly chooses to regenerate.
- Explicit, user-driven version snapshots ("Save as Version") with the ability to revert. Reverts are themselves reversible (auto-snapshot before revert).
- Refactor wizard components into shared, reusable building blocks so the edit experience and the creation wizard share the same brief/chat editors.

## Non-Goals

- Automatic snapshots on every save (explicit-only).
- Snapshotting variation images (snapshots store project state only — images stay in blob storage uniformly across versions).
- Multi-user concurrent editing (last-write-wins is acceptable; this is a single-user app).
- Renaming variations or editing image metadata.

## Approach

**Approach C — Extract shared core + thin wrappers.**

Editing logic from the wizard's brief editor, design chat, and settings sections is extracted into shared headless hooks and presentational components. Both the creation wizard and the new edit-page tabs are thin wrappers around those shared pieces. This produces clean reuse without the wizard and edit experiences becoming coupled to each other.

The version history feature is a separate concern with its own hook (`useProjectVersions`) and Cosmos document type (`staging_project_version`).

## User-Facing Design

### Page Layout & Navigation

The project detail page (`/frontend/app/projects/[id]/page.tsx`) is restructured into three zones:

**Header zone (always visible)**
- Project name as inline-editable text. Click to edit; blur or Enter saves; Escape cancels. Pencil icon appears on hover as affordance.
- Status badge unchanged.
- Existing action bar (Generate, Reset, Delete) preserved on the right.

**Tab bar (below header)**
URL query param `?tab=` controls the active tab. Default is `gallery` (no param).

| Tab | Purpose |
|-----|---------|
| **Gallery** *(default)* | Current project view — rooms, variations, lightbox, regenerate. Adds an "Add Images" button and per-room "Remove" action. |
| **Design Brief** | Full editing of prompt, plant palette, placement guide, preserve elements, per-image notes, plus the AI design chat (collapsible panel at top). |
| **Settings** | Generation settings (variations per room, model, quality, size). |
| **History** | List of saved versions with metadata (timestamp, label, note). Each row exposes Preview and Revert. |

Browser back/forward and deep-linking via the `?tab=` param both work.

### Regenerate Prompt

After saving brief or settings changes, a non-blocking inline alert appears:

> "Brief updated — regenerate variations to apply?" with **Regenerate** and **Dismiss** buttons.

Existing variations are untouched until the user clicks Regenerate. Dismiss clears the alert; the alert reappears on the next save.

### Editing While Generating

When `project.status === 'processing'`, all edit forms are disabled (read-only) and a banner explains: "Generation in progress — wait for it to finish or click Reset to edit." This prevents inconsistent state mid-stream. (There is no Pause action; the existing Reset button is the user's escape hatch.)

### Version History UX

- Each version row shows: timestamp, optional label, optional note, and a diff summary indicating which top-level fields differ from the **current** project state (e.g., "Brief changed · Settings same · Name same"). The summary is computed client-side by shallow-comparing snapshot fields to the current project fields after the version list and project both load.
- "Preview" opens a modal showing the full snapshot read-only.
- "Revert" prompts confirmation, then:
  1. Auto-creates a snapshot of the current state labeled "Auto-saved before revert" (visible in the version list like any other snapshot).
  2. Applies the target snapshot's fields to the project.
  3. Variations are kept; the regenerate prompt appears.
- "Delete" removes a snapshot (with confirmation).

## Backend Design

### New API Endpoints

All under `/backend/api/endpoints/staging.py`.

**Project field updates**
- `PATCH /api/v1/staging/projects/{project_id}` — partial update of name, prompt, settings. Body: `UpdateProjectRequest` (all fields optional).

**Room management**
- `DELETE /api/v1/staging/projects/{project_id}/rooms/{room_id}` — removes a room and its variations; cleans up associated blobs in `staging/{project_id}/originals/` and `staging/{project_id}/variations/`.
- `PATCH /api/v1/staging/projects/{project_id}/rooms/{room_id}` — updates room label.
- *(Reuses existing `POST /projects/{id}/rooms` for adding new images.)*

**Brief**
- *(Reuses existing `PUT /projects/{id}/brief`.)*

**Versioning**
- `POST /api/v1/staging/projects/{project_id}/versions` — creates a snapshot. Body: `CreateVersionRequest` with optional `label` and `note`.
- `GET /api/v1/staging/projects/{project_id}/versions` — lists versions, newest first.
- `GET /api/v1/staging/projects/{project_id}/versions/{version_id}` — full version details.
- `POST /api/v1/staging/projects/{project_id}/versions/{version_id}/revert` — applies a snapshot. Auto-creates a "before revert" snapshot first.
- `DELETE /api/v1/staging/projects/{project_id}/versions/{version_id}` — deletes a snapshot.

### New Pydantic Models (`/backend/models/staging.py`)

```python
class UpdateProjectRequest(BaseModel):
    name: str | None = None
    prompt: str | None = None
    settings: StagingSettings | None = None

class UpdateRoomRequest(BaseModel):
    label: str

class CreateVersionRequest(BaseModel):
    label: str | None = None
    note: str | None = None

class VersionSnapshot(BaseModel):
    name: str
    prompt: str
    settings: StagingSettings
    design_brief: DesignBrief | None
    room_labels: dict[str, str]            # room_id -> label
    conversation_history: list[ChatMessage] | None

class StagingProjectVersion(BaseModel):
    id: UUID
    project_id: UUID
    label: str | None
    note: str | None
    created_at: datetime
    snapshot: VersionSnapshot
```

Add to existing `StagingProject` model:
- `conversation_history: list[ChatMessage] | None = None` — explicit field so the design chat can resume after a page reload.

### Storage

**Cosmos DB document type:** new `doc_type="staging_project_version"`. Documents stored in the same container as projects, with partition key `/project_id` so a project's versions are colocated for efficient list queries.

**`StagingStorageService` additions** (`/backend/core/staging_storage.py`):
- `create_version(project_id, snapshot, label, note)`
- `list_versions(project_id, limit, offset)` — DESC by `created_at`
- `get_version(project_id, version_id)`
- `delete_version(project_id, version_id)`

Revert is composed in the endpoint handler: snapshot current state, then call `update_project` with the target snapshot fields.

**Storage cost:** snapshots are small (a few KB JSON each). No image data is duplicated. With explicit-only snapshots, version count per project stays modest (likely < 20). No automatic cleanup policy at launch; can add a max-versions cap later if needed.

## Frontend Design

### Shared Headless Hooks (`/frontend/components/staging/shared/`)

- `useBriefEditor(initialBrief)` — manages brief state, dirty tracking, validation, save handler. Used by the wizard step 4 and by `ProjectBriefTab`.
- `useDesignChat(projectId, initialHistory)` — manages chat state, focused image, AI calls, "ready" intent detection.
- `useProjectSettings(initialSettings)` — manages settings form state and dirty tracking.
- `useProjectVersions(projectId)` — fetch/create/list/revert versions; loading and error states.

### Shared Presentational Components

Extracted from existing wizard code:
- `<PlantPaletteTable />` — already exists; promoted to shared.
- `<DesignBriefForm />` — extracted from `DesignBriefEditor`; pure form, no flow logic.
- `<DesignChatPanel />` — extracted from `DesignChat`; pure chat UI, no flow logic.
- `<GenerationSettingsForm />` — extracted from `GenerationSummary`'s settings section.

### Edit-Page Components (`/frontend/components/staging/edit/`)

- `<ProjectTabs />` — tab bar with URL query param sync.
- `<EditableProjectName />` — inline-editable header text.
- `<ProjectGalleryTab />` — wraps existing detail body; adds "Add Images" button and per-room "Remove" action.
- `<ProjectBriefTab />` — composes `<DesignChatPanel />` (collapsible) + `<DesignBriefForm />`.
- `<ProjectSettingsTab />` — composes `<GenerationSettingsForm />`.
- `<ProjectHistoryTab />` — version list, preview modal, revert flow.
- `<RegeneratePrompt />` — inline alert shown after brief/settings save.

### Wizard Refactor (`<NewProjectWizard />`)

Steps 3 and 4 are reworked to use `<DesignChatPanel />` and `<DesignBriefForm />` via the shared hooks. No behavior change — the wizard remains a guided 5-step flow; only the implementation delegates to shared pieces.

### Service Layer (`/frontend/services/stagingApi.ts`)

New functions:
- `updateProject(projectId, patch)`
- `removeRoom(projectId, roomId)`
- `updateRoom(projectId, roomId, patch)`
- `listVersions(projectId)`
- `createVersion(projectId, label?, note?)`
- `getVersion(projectId, versionId)`
- `revertVersion(projectId, versionId)`
- `deleteVersion(projectId, versionId)`

### State Management

The edit page uses local React state per tab plus a top-level `project` state in `[id]/page.tsx`. After any successful save, the project is refetched to keep all tabs in sync. No new global context is introduced.

## Migration

No destructive migration needed. Existing projects:
- Lack `conversation_history` → field is optional, defaults to `None`. The chat panel shows the empty-state UI.
- Lack any version snapshots → version queries return an empty list. The History tab shows "No versions saved yet" with a "Save first version" CTA.

The existing project list and detail pages keep working unchanged for any project that hasn't been edited yet.

## Edge Cases

- **Editing during generation:** edit forms disabled while `status === 'processing'`; banner explains. Generate/Reset still work.
- **Revert during processing:** blocked with "Cancel current generation first" prompt.
- **Concurrent edits across browser tabs:** last-write-wins; project is refetched after every save so the second tab sees the new state on its next save attempt.
- **Removing the only image:** blocked client-side with "A project needs at least one image."
- **Versions referencing deleted rooms:** snapshot stores room IDs and labels. Revert restores label for rooms that still exist; removed rooms are not resurrected.
- **Empty brief on revert:** if the snapshot predates brief generation, the Brief tab falls back to the chat-then-generate flow.
- **Project name validation:** non-empty, trimmed, max 200 chars (matches existing wizard validation).

## Testing Strategy

### Backend (`/tests/`)

Unit tests for new endpoints:
- `PATCH /projects/{id}` — name only, prompt only, settings only, all fields, invalid project ID, validation errors.
- `DELETE /projects/{id}/rooms/{room_id}` — verifies blob cleanup, room not found, last room blocked.
- `PATCH /projects/{id}/rooms/{room_id}` — label update, validation.
- Version endpoints — create with/without label, list (empty + populated), get, revert (verifies auto-snapshot of current state happens first; verifies project fields applied), delete.

Storage layer tests for `create_version`, `list_versions`, `get_version`, `delete_version` — mocked Cosmos client.

### Frontend (Playwright E2E)

New file `tests/e2e/edit-project.spec.ts`:
- Inline name edit (click → edit → save; blur saves; Escape cancels).
- Tab navigation persists in URL; browser back/forward works.
- Brief tab: edit → save → regenerate prompt appears → dismiss → variations preserved.
- Brief tab: edit → save → regenerate → SSE stream progresses.
- Settings tab: edit → save → regenerate prompt appears.
- Add image to existing project → new room appears in Gallery.
- Remove room → variations and blobs cleaned up.
- Save version → appears in History tab → revert restores fields → variations untouched.
- Resume design chat — opens with prior history visible.

Per `AGENTS.md`, reports saved to `tests/playwright/<YYYY-MM-DD-HHMMSS>/` with screenshots and multi-browser coverage.

### Build & Lint

- `cd frontend && npm run build`
- `cd frontend && npx next lint`
- `uv run pytest tests/ --ignore=tests/integration -v`

## Open Questions

None at design approval. All scope decisions locked in during brainstorming.

## Out of Scope (Captured for Future Work)

- Automatic snapshots on a schedule or on every save.
- Image-level edits (cropping, replacing the original of a single room without re-uploading).
- Multi-user real-time co-editing.
- Diff view between two arbitrary versions in the History tab (currently each version is previewed standalone).
- Storage cleanup policy / max version cap.
