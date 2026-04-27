# Activity Log Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a right-side slide-out panel that shows live console-style generation events scoped per creation page, with auto-open on generation start and a toggle to dismiss/reopen.

**Architecture:** React Context (`ActivityLogProvider`) wraps the app in `layout.tsx`. Pages call `useActivityLog().log()` to push entries. A shadcn `Sheet` (side="right") rendered once in the layout displays entries. The panel auto-opens on first log entry and auto-scrolls to latest.

**Tech Stack:** React Context, shadcn/ui Sheet + ScrollArea + Badge, lucide-react icons, Tailwind CSS, existing SSE streaming infrastructure.

---

## File Structure

| File | Purpose | Action |
|------|---------|--------|
| `frontend/context/activity-log-context.tsx` | Provider, hook, LogEntry type, state management | Create |
| `frontend/components/activity-log/LogEntry.tsx` | Single log entry component — timestamp, icon, message, detail | Create |
| `frontend/components/activity-log/ActivityLogPanel.tsx` | Sheet-based panel with scrollable console-style log | Create |
| `frontend/components/activity-log/ActivityLogToggle.tsx` | Header toggle button with event count badge | Create |
| `frontend/app/layout.tsx` | Wrap children with provider; add toggle to header; render panel | Modify |
| `frontend/app/projects/[id]/page.tsx` | Push staging SSE events to activity log | Modify |
| `backend/core/staging_pipeline.py` | Add elapsed_ms, tokens_used, model to SSE events | Modify |
| `frontend/app/edit-image/components/EditorContainer.tsx` | Push edit events to activity log | Modify |
| `frontend/components/ImageCreationContainer.tsx` | Push generate events to activity log | Modify |
| `frontend/context/video-queue-context.tsx` | Push video job events to activity log | Modify |
| `tests/test_staging_pipeline.py` | Test enhanced SSE events | Modify |

---

### Task 1: Create ActivityLogProvider Context

**Files:**
- Create: `frontend/context/activity-log-context.tsx`

- [ ] **Step 1: Create the context file**

```typescript
// frontend/context/activity-log-context.tsx
"use client";

import { createContext, useContext, useRef, useState, useCallback, type ReactNode } from "react";

export interface LogEntry {
  id: string;
  timestamp: Date;
  level: "info" | "success" | "error" | "warn";
  message: string;
  detail?: string;
  icon?: string;
}

interface ActivityLogContextValue {
  entries: LogEntry[];
  log: (entry: Omit<LogEntry, "id" | "timestamp">) => void;
  clear: () => void;
  isOpen: boolean;
  setOpen: (open: boolean) => void;
  hasActivity: boolean;
}

const MAX_ENTRIES = 500;

const ActivityLogContext = createContext<ActivityLogContextValue | null>(null);

export function ActivityLogProvider({ children }: { children: ReactNode }) {
  const entriesRef = useRef<LogEntry[]>([]);
  const [revision, setRevision] = useState(0);
  const [isOpen, setIsOpen] = useState(false);

  const log = useCallback((entry: Omit<LogEntry, "id" | "timestamp">) => {
    const newEntry: LogEntry = {
      ...entry,
      id: crypto.randomUUID(),
      timestamp: new Date(),
    };
    entriesRef.current = [...entriesRef.current, newEntry].slice(-MAX_ENTRIES);

    // Auto-open on first entry
    if (entriesRef.current.length === 1) {
      setIsOpen(true);
    }

    setRevision((r) => r + 1);
  }, []);

  const clear = useCallback(() => {
    entriesRef.current = [];
    setRevision((r) => r + 1);
  }, []);

  const value: ActivityLogContextValue = {
    entries: entriesRef.current,
    log,
    clear,
    isOpen,
    setOpen: setIsOpen,
    hasActivity: entriesRef.current.length > 0,
  };

  return (
    <ActivityLogContext.Provider value={value}>
      {children}
    </ActivityLogContext.Provider>
  );
}

export function useActivityLog(): ActivityLogContextValue {
  const context = useContext(ActivityLogContext);
  if (!context) {
    // No-op fallback if used outside provider (should not happen)
    return {
      entries: [],
      log: () => {},
      clear: () => {},
      isOpen: false,
      setOpen: () => {},
      hasActivity: false,
    };
  }
  return context;
}
```

- [ ] **Step 2: Verify the file compiles**

Run: `cd frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors related to `activity-log-context.tsx`

- [ ] **Step 3: Commit**

```bash
git add frontend/context/activity-log-context.tsx
git commit -m "feat: add ActivityLogProvider context for live generation logging

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Create LogEntry Component

**Files:**
- Create: `frontend/components/activity-log/LogEntry.tsx`

- [ ] **Step 1: Create the LogEntry component**

```typescript
// frontend/components/activity-log/LogEntry.tsx
"use client";

import { cn } from "@/utils/utils";
import type { LogEntry as LogEntryType } from "@/context/activity-log-context";

interface LogEntryProps {
  entry: LogEntryType;
}

const levelColors: Record<LogEntryType["level"], string> = {
  info: "text-blue-400",
  success: "text-green-400",
  error: "text-red-400",
  warn: "text-amber-400",
};

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function LogEntryRow({ entry }: LogEntryProps) {
  return (
    <div className="px-3 py-1.5 border-b border-white/5 font-mono text-[11px] leading-relaxed">
      <div className="text-muted-foreground/50">{formatTime(entry.timestamp)}</div>
      <div className={cn(levelColors[entry.level])}>
        {entry.icon && <span className="mr-1">{entry.icon}</span>}
        {entry.message}
      </div>
      {entry.detail && (
        <div className="text-muted-foreground/40 text-[10px]">{entry.detail}</div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/components/activity-log/LogEntry.tsx
git commit -m "feat: add LogEntry component for activity log panel

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Create ActivityLogPanel Component

**Files:**
- Create: `frontend/components/activity-log/ActivityLogPanel.tsx`

- [ ] **Step 1: Create the panel component**

```typescript
// frontend/components/activity-log/ActivityLogPanel.tsx
"use client";

import { useRef, useEffect, useState } from "react";
import { Trash2, ArrowDown } from "lucide-react";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useActivityLog } from "@/context/activity-log-context";
import { LogEntryRow } from "./LogEntry";

export function ActivityLogPanel() {
  const { entries, clear, isOpen, setOpen } = useActivityLog();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const prevLengthRef = useRef(entries.length);

  // Auto-scroll to bottom when new entries arrive
  useEffect(() => {
    if (entries.length > prevLengthRef.current && autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
    prevLengthRef.current = entries.length;
  }, [entries.length, autoScroll]);

  // Detect user scroll position to pause/resume auto-scroll
  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const nearBottom = scrollHeight - scrollTop - clientHeight < 50;
    setAutoScroll(nearBottom);
  };

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      setAutoScroll(true);
    }
  };

  return (
    <Sheet open={isOpen} onOpenChange={setOpen}>
      <SheetContent
        side="right"
        className="w-[380px] sm:max-w-[380px] p-0 flex flex-col bg-[#0d1117] border-l border-border/50"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 shrink-0">
          <div className="flex items-center gap-2">
            <SheetTitle className="text-sm font-semibold text-foreground">Activity Log</SheetTitle>
            {entries.length > 0 && (
              <Badge variant="secondary" className="text-[10px] px-1.5 py-0 h-5">
                {entries.length}
              </Badge>
            )}
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-muted-foreground hover:text-foreground"
            onClick={clear}
            title="Clear log"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* Log body */}
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto min-h-0"
        >
          {entries.length === 0 ? (
            <div className="flex items-center justify-center h-full text-muted-foreground/40 text-xs font-mono">
              No activity yet
            </div>
          ) : (
            entries.map((entry) => <LogEntryRow key={entry.id} entry={entry} />)
          )}
        </div>

        {/* "New events" button when auto-scroll is paused */}
        {!autoScroll && entries.length > 0 && (
          <div className="absolute bottom-10 left-1/2 -translate-x-1/2">
            <Button
              size="sm"
              variant="secondary"
              className="text-xs h-7 shadow-lg"
              onClick={scrollToBottom}
            >
              <ArrowDown className="h-3 w-3 mr-1" />
              New events
            </Button>
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-white/10 shrink-0">
          <span className="text-[10px] text-muted-foreground/40 font-mono">
            Auto-scroll: {autoScroll ? "on" : "paused"}
          </span>
          <span className="text-[10px] text-green-400/60 flex items-center gap-1 font-mono">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-green-400" />
            Ready
          </span>
        </div>
      </SheetContent>
    </Sheet>
  );
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/components/activity-log/ActivityLogPanel.tsx
git commit -m "feat: add ActivityLogPanel with auto-scroll and console styling

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Create ActivityLogToggle Component

**Files:**
- Create: `frontend/components/activity-log/ActivityLogToggle.tsx`

- [ ] **Step 1: Create the toggle button component**

```typescript
// frontend/components/activity-log/ActivityLogToggle.tsx
"use client";

import { Terminal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useActivityLog } from "@/context/activity-log-context";

export function ActivityLogToggle() {
  const { isOpen, setOpen, hasActivity, entries } = useActivityLog();

  if (!hasActivity) return null;

  return (
    <Button
      variant="ghost"
      size="icon"
      className="relative h-8 w-8"
      onClick={() => setOpen(!isOpen)}
      title={isOpen ? "Hide activity log" : "Show activity log"}
    >
      <Terminal className="h-4 w-4" />
      {!isOpen && entries.length > 0 && (
        <span className="absolute -top-0.5 -right-0.5 h-2.5 w-2.5 rounded-full bg-blue-500 border-2 border-background" />
      )}
    </Button>
  );
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/components/activity-log/ActivityLogToggle.tsx
git commit -m "feat: add ActivityLogToggle header button with notification dot

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Integrate into Layout

**Files:**
- Modify: `frontend/app/layout.tsx:1-14` (imports), `78-113` (provider + panel + toggle)

- [ ] **Step 1: Add imports to layout.tsx**

Add these imports after line 12 (`import { FolderProvider } from "@/context/folder-context";`):

```typescript
import { ActivityLogProvider } from "@/context/activity-log-context";
import { ActivityLogToggle } from "@/components/activity-log/ActivityLogToggle";
import { ActivityLogPanel } from "@/components/activity-log/ActivityLogPanel";
```

- [ ] **Step 2: Wrap content with ActivityLogProvider**

In `layout.tsx`, change the provider nesting. Replace lines 81-117 (the `FolderProvider` block) with:

```typescript
                    <FolderProvider>
                    <ActivityLogProvider>
                    {/* Main layout with sidebar */}
                    <div className="relative flex min-h-screen h-screen">              
                      {/* Content area with sidebar */}
                      <SidebarProvider
                        style={
                          {
                            "--sidebar-width": "12rem",
                          } as React.CSSProperties
                        }
                        className="flex h-full w-full"
                      >
                        {/* Sidebar for navigation - wrapped in Suspense to fix hydration errors */}
                        <Suspense fallback={
                          <div className="w-[var(--sidebar-width)] shrink-0 border-r h-full" />
                        }>
                          <AppSidebar />
                        </Suspense>
                        <SidebarInset className="flex-1 flex flex-col h-full w-full">
                          <div className="flex h-14 items-center gap-2 border-b shrink-0 px-3">
                            <SidebarTrigger />
                            <Separator orientation="vertical" className="mx-2 h-4" />
                            <div className="ml-auto flex items-center space-x-2">
                              <ActivityLogToggle />
                              <RefreshJobsButton />
                              <VideoQueueClient />
                            </div>
                          </div>
                          <main className="flex-1 overflow-auto w-full transition-all duration-200">
                            <AnimatedLayout>
                              {children}
                            </AnimatedLayout>
                          </main>
                        </SidebarInset>
                      </SidebarProvider>
                    </div>
                    <ActivityLogPanel />
                    <Toaster />
                    </ActivityLogProvider>
                    </FolderProvider>
```

Key changes from the original:
- Added `<ActivityLogProvider>` wrapping inside `<FolderProvider>`
- Added `<ActivityLogToggle />` before `<RefreshJobsButton />` in the header
- Added `<ActivityLogPanel />` before `<Toaster />`

- [ ] **Step 3: Build to verify**

Run: `cd frontend && npm run build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add frontend/app/layout.tsx
git commit -m "feat: integrate ActivityLogProvider, toggle, and panel into app layout

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Enhance Backend SSE Events with Timing and Token Data

**Files:**
- Modify: `backend/core/staging_pipeline.py:199-241`
- Modify: `tests/test_staging_pipeline.py`

- [ ] **Step 1: Write failing test for enhanced SSE event fields**

Add this test to `tests/test_staging_pipeline.py` in the `TestVariationUrlExtraction` class:

```python
    @pytest.mark.asyncio
    async def test_sse_events_include_timing_and_token_data(self):
        """Variation SSE events must include elapsed_ms, tokens_used, and model."""
        from backend.core.staging_pipeline import StagingPipeline

        project = _make_project(n_rooms=1, n_variations=1)
        room = project.rooms[0]

        gen = ImageGenerationResponse(
            success=True,
            message="ok",
            imgen_model_response={
                "data": [{"b64_json": "AAAA"}],
                "usage": {"total_tokens": 1500, "input_tokens": 800, "output_tokens": 700},
            },
            token_usage={"total_tokens": 1500, "input_tokens": 800, "output_tokens": 700},
        )
        save = ImageSaveResponse(
            success=True,
            message="Saved 1 image(s)",
            saved_images=[{"url": "https://example.com/img.png", "blob_name": "img.png"}],
            total_saved=1,
        )
        pipeline_response = ImagePipelineResponse(
            success=True,
            message="Pipeline completed",
            steps=[PipelineStepResult(step="edit", success=True), PipelineStepResult(step="save", success=True)],
            generation=gen,
            save=save,
        )

        mock_pipeline = AsyncMock()
        mock_pipeline.process_pipeline.return_value = pipeline_response

        mock_blob = MagicMock()
        mock_blob.get_asset_content.return_value = (b"\x89PNG\r\n", "image/png")

        mock_storage = MagicMock()
        mock_llm = AsyncMock()
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["Add plants"]'))]
        )
        mock_analyzer = AsyncMock()
        mock_analyzer.async_image_chat.return_value = {"description": "A room", "features": []}

        staging = StagingPipeline(
            async_llm_client=mock_llm,
            llm_deployment="gpt-4o",
            image_analyzer=mock_analyzer,
            image_pipeline=mock_pipeline,
            storage_service=mock_storage,
            blob_service=mock_blob,
        )

        events = []
        async for event in staging.process_room(project, room):
            events.append(event)

        completed_events = [e for e in events if e["type"] == "variation_completed"]
        assert len(completed_events) == 1
        evt = completed_events[0]
        assert "elapsed_ms" in evt
        assert isinstance(evt["elapsed_ms"], int)
        assert "tokens_used" in evt
        assert evt["tokens_used"] == 1500
        assert "model" in evt
        assert evt["model"] == "gpt-image-2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/test_staging_pipeline.py::TestVariationUrlExtraction::test_sse_events_include_timing_and_token_data -v --tb=short`
Expected: FAIL — `elapsed_ms` not in event dict

- [ ] **Step 3: Enhance SSE events in staging_pipeline.py**

Replace lines 235-241 in `backend/core/staging_pipeline.py`:

```python
                    # Extract token usage from generation response
                    token_usage = None
                    if result.generation and result.generation.token_usage:
                        token_usage = result.generation.token_usage

                    self._update_room_in_project(project, room)

                    yield {
                        "type": f"variation_{'completed' if variation.status == ItemStatus.COMPLETED else 'failed'}",
                        "room_id": room.id,
                        "variation_index": idx,
                        "image_url": variation.image_url,
                        "error": variation.error,
                        "elapsed_ms": elapsed_ms,
                        "tokens_used": token_usage.get("total_tokens") if isinstance(token_usage, dict) else None,
                        "model": project.settings.model,
                    }
```

Note: This replaces the existing `self._update_room_in_project(project, room)` and `yield {...}` block. The `_update_room_in_project` call moves above the yield. The `elapsed_ms` variable is already computed on line 204.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/test_staging_pipeline.py -v --tb=short`
Expected: All 4 tests PASS

- [ ] **Step 5: Run full backend tests**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/ --ignore=tests/integration -v --tb=short 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add backend/core/staging_pipeline.py tests/test_staging_pipeline.py
git commit -m "feat: add elapsed_ms, tokens_used, model to staging SSE events

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: Integrate Activity Log into Staging Projects Page

**Files:**
- Modify: `frontend/app/projects/[id]/page.tsx:1-5` (imports), `102-128` (handleStreamEvent)

- [ ] **Step 1: Add import**

Add after the existing imports (line 22):

```typescript
import { useActivityLog } from "@/context/activity-log-context";
```

- [ ] **Step 2: Add hook and clear on mount**

Inside the `ProjectDetailPage` component, after the existing state declarations (after line 34), add:

```typescript
  const activityLog = useActivityLog();

  // Clear activity log when page mounts (scope log to this page)
  useEffect(() => {
    activityLog.clear();
    return () => activityLog.clear();
  }, []);
```

- [ ] **Step 3: Add log calls to handleStreamEvent**

Replace the `handleStreamEvent` callback (lines 102-128) with:

```typescript
  const handleStreamEvent = useCallback((event: StagingStreamEvent) => {
    switch (event.type) {
      case 'room_started':
        activityLog.log({
          level: 'info',
          icon: '▶',
          message: `Starting generation for "${(event as any).label ?? 'room'}"`,
          detail: `Room ${(event as any).room_id?.slice(0, 8)}`,
        });
        debouncedReload();
        break;
      case 'variation_completed':
        activityLog.log({
          level: 'success',
          icon: '✓',
          message: `Variation ${((event as any).variation_index ?? 0) + 1} saved`,
          detail: [
            (event as any).model,
            (event as any).tokens_used ? `${(event as any).tokens_used.toLocaleString()} tokens` : null,
            (event as any).elapsed_ms ? `${((event as any).elapsed_ms / 1000).toFixed(1)}s` : null,
          ].filter(Boolean).join(' · ') || undefined,
        });
        debouncedReload();
        break;
      case 'variation_failed':
        activityLog.log({
          level: 'error',
          icon: '✕',
          message: `Variation ${((event as any).variation_index ?? 0) + 1} failed`,
          detail: (event as any).error || 'Unknown error',
        });
        debouncedReload();
        break;
      case 'room_completed':
        activityLog.log({
          level: 'success',
          icon: '✓',
          message: `Room complete`,
          detail: `Room ${(event as any).room_id?.slice(0, 8)}`,
        });
        debouncedReload();
        break;
      case 'room_failed':
        activityLog.log({
          level: 'error',
          icon: '✕',
          message: `Room failed`,
          detail: (event as any).error || 'Unknown error',
        });
        debouncedReload();
        break;
      case 'room_uploaded':
        debouncedReload();
        break;
      case 'project_completed':
        activityLog.log({
          level: 'success',
          icon: '🎉',
          message: 'Generation complete!',
          detail: `${completedVariations}/${totalVariations} variations succeeded`,
        });
        setIsGenerating(false);
        setGenerationError(null);
        toast.success('Generation completed!');
        loadProject();
        break;
      case 'error':
        activityLog.log({
          level: 'error',
          icon: '✕',
          message: 'Generation error',
          detail: event.error || 'Unknown error',
        });
        setIsGenerating(false);
        setGenerationError(event.error || 'Generation failed');
        toast.error(event.error || 'Generation failed');
        loadProject();
        break;
    }
  }, [debouncedReload, loadProject, activityLog, completedVariations, totalVariations]);
```

- [ ] **Step 4: Add log entry when generation starts**

In `startGeneration` (around line 130), add a log call after `setIsGenerating(true)`:

```typescript
  const startGeneration = useCallback(() => {
    if (isGenerating) return;
    streamCleanupRef.current?.();
    setIsGenerating(true);
    setGenerationError(null);
    activityLog.log({
      level: 'info',
      icon: '▶',
      message: `Starting generation for "${project?.name}"`,
      detail: `${totalVariations} variations queued across ${project?.rooms.length} images`,
    });
    streamCleanupRef.current = streamGeneration(projectId, handleStreamEvent);
  }, [isGenerating, projectId, handleStreamEvent, activityLog, project, totalVariations]);
```

- [ ] **Step 5: Build to verify**

Run: `cd frontend && npm run build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 6: Commit**

```bash
git add frontend/app/projects/\\[id\\]/page.tsx
git commit -m "feat: push staging SSE events to activity log panel

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 8: Integrate Activity Log into Image Edit Page

**Files:**
- Modify: `frontend/app/edit-image/components/EditorContainer.tsx`

- [ ] **Step 1: Add import and hook**

Add import after existing imports:

```typescript
import { useActivityLog } from "@/context/activity-log-context";
```

Inside the component function, add after existing state declarations:

```typescript
  const activityLog = useActivityLog();

  useEffect(() => {
    activityLog.clear();
    return () => activityLog.clear();
  }, []);
```

- [ ] **Step 2: Add log calls to handleSubmit**

In `handleSubmit` (around line 86), add log calls:

After `setIsLoading(true)` (line 87):
```typescript
    activityLog.log({
      level: 'info',
      icon: '🎨',
      message: 'Editing image...',
      detail: `${formData.get('model') ?? 'gpt-image-2'} · ${formData.get('quality') ?? 'auto'} quality · ${formData.get('size') ?? 'auto'}`,
    });
```

After `setResultData({...})` (line 106), before `setEditorState('result')`:
```typescript
      activityLog.log({
        level: 'success',
        icon: '✓',
        message: 'Image generated',
        detail: tokenUsage ? `${tokenUsage.total.toLocaleString()} tokens` : undefined,
      });
```

In the catch block (line 110), before `toast.error(...)`:
```typescript
      activityLog.log({
        level: 'error',
        icon: '✕',
        message: 'Edit failed',
        detail: error instanceof Error ? error.message : 'Unknown error',
      });
```

- [ ] **Step 3: Add log calls to handleSaveImage**

In `handleSaveImage` (line 121), add:

After `try {` (line 129):
```typescript
      activityLog.log({
        level: 'info',
        icon: '💾',
        message: 'Saving to gallery...',
        detail: folder ? `Folder: ${folder}` : 'Root folder',
      });
```

After the `toast.success(...)` (line 143):
```typescript
      activityLog.log({
        level: 'success',
        icon: '✓',
        message: 'Saved to gallery',
        detail: folder ? `Folder: ${folder}` : 'Root folder',
      });
```

- [ ] **Step 4: Build to verify**

Run: `cd frontend && npm run build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add frontend/app/edit-image/components/EditorContainer.tsx
git commit -m "feat: push image edit events to activity log panel

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 9: Integrate Activity Log into New Image Page

**Files:**
- Modify: `frontend/components/ImageCreationContainer.tsx`

- [ ] **Step 1: Add import and hook**

Add import after existing imports:

```typescript
import { useActivityLog } from "@/context/activity-log-context";
```

Inside the component function (after line 98), add:

```typescript
  const activityLog = useActivityLog();

  useEffect(() => {
    activityLog.clear();
    return () => activityLog.clear();
  }, []);
```

- [ ] **Step 2: Add log calls to handleGenerate**

In `handleGenerate` (line 138), add log calls at key points:

After `setIsGenerating(true)` (line 140):
```typescript
      activityLog.log({
        level: 'info',
        icon: '🎨',
        message: newSettings.sourceImages?.length
          ? `Editing ${newSettings.sourceImages.length} image(s)...`
          : `Generating ${newSettings.variations} image(s)...`,
        detail: `${newSettings.model ?? 'gpt-image-2'} · ${newSettings.quality} quality · ${newSettings.imageSize}`,
      });
```

After the edit `toast.success(...)` (line 201):
```typescript
        activityLog.log({
          level: 'success',
          icon: '✓',
          message: `Image editing completed`,
          detail: `${formatImageCount(newSettings.variations)} created`,
        });
```

After the save `toast.success(...)` (line 226):
```typescript
          activityLog.log({
            level: 'success',
            icon: '✓',
            message: `${saveResp.total_saved} images saved${saveResp.analyzed ? ' with AI analysis' : ''}`,
            detail: normalizedFolder ? `Folder: ${normalizedFolder}` : 'Root folder',
          });
```

In the main `catch` block of `handleGenerate` (wherever the generation error is caught), add before the error toast:
```typescript
        activityLog.log({
          level: 'error',
          icon: '✕',
          message: 'Generation failed',
          detail: error instanceof Error ? error.message : 'Unknown error',
        });
```

- [ ] **Step 3: Build to verify**

Run: `cd frontend && npm run build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add frontend/components/ImageCreationContainer.tsx
git commit -m "feat: push new image generation events to activity log panel

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 10: Integrate Activity Log into Video Queue

**Files:**
- Modify: `frontend/context/video-queue-context.tsx`

- [ ] **Step 1: Add import**

Add after existing imports (line 17):

```typescript
import { useActivityLog } from "@/context/activity-log-context";
```

- [ ] **Step 2: Add hook inside VideoQueueProvider**

Inside the `VideoQueueProvider` component function, add the hook:

```typescript
  const activityLog = useActivityLog();
```

- [ ] **Step 3: Add log calls at key video lifecycle points**

Find where video jobs are created (look for `createVideoGenerationJob` or `createVideoGenerationWithAnalysis` calls) and add:

When a video job starts:
```typescript
      activityLog.log({
        level: 'info',
        icon: '🎬',
        message: 'Video generation started',
        detail: `Processing video request`,
      });
```

When a video job completes (look for success toasts or status changes to completed):
```typescript
      activityLog.log({
        level: 'success',
        icon: '✓',
        message: 'Video ready',
        detail: `Video generation completed`,
      });
```

When a video job fails:
```typescript
      activityLog.log({
        level: 'error',
        icon: '✕',
        message: 'Video generation failed',
        detail: error instanceof Error ? error.message : 'Unknown error',
      });
```

Note: The video queue context is complex with streaming. Add log calls alongside existing `toast` calls — wherever there's a `toast.success()`, `toast.error()`, or `toast.loading()` for video operations, add the corresponding `activityLog.log()`.

- [ ] **Step 4: Build to verify**

Run: `cd frontend && npm run build 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add frontend/context/video-queue-context.tsx
git commit -m "feat: push video generation events to activity log panel

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 11: Run Full Test Suite and Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Run backend tests**

Run: `cd /Users/bradcstevens/code/github/bradcstevens/visionary-lab && uv run pytest tests/ --ignore=tests/integration -v --tb=short 2>&1 | tail -10`
Expected: All tests pass (58+)

- [ ] **Step 2: Build frontend**

Run: `cd frontend && npm run build 2>&1 | tail -10`
Expected: Build succeeds with no errors

- [ ] **Step 3: Lint frontend**

Run: `cd frontend && npx next lint 2>&1 | tail -10`
Expected: No new lint errors

- [ ] **Step 4: Verify .gitignore includes .superpowers/**

Run: `grep -q '.superpowers' .gitignore && echo "OK" || echo "MISSING"`
If MISSING, add `.superpowers/` to `.gitignore`.

- [ ] **Step 5: Final commit if any cleanup needed**

```bash
git add -A
git status
# Only commit if there are changes
git diff --cached --quiet || git commit -m "chore: cleanup and verify activity log panel integration

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
