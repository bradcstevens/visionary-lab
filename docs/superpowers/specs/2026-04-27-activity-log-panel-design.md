# Activity Log Panel — Design Spec

## Overview

A slide-out panel on the right side of the app that shows a live, console-style log of events during image and video generation. Each creation page gets its own scoped log. The panel auto-opens when generation begins and can be dismissed and reopened via a toggle button in the header.

## Requirements

- **Page-scoped logs**: Each creation page (`/new-image`, `/edit-image`, `/projects/[id]`, `/new-video`) maintains its own independent log. Navigating away clears the log.
- **Auto-open**: The panel opens automatically when the first log entry is pushed during a generation task.
- **Dismissible + reopenable**: Users can close the panel. A toggle button in the header bar reopens it. The button shows a badge when there are unread entries while the panel is closed.
- **Detailed entries**: Each entry includes timestamp, color-coded action message, and a detail line with model, token counts, and timing.
- **Console aesthetic**: Monospace font, dark background, dense layout — reads like a terminal log.
- **Auto-scroll**: Scrolls to the latest entry. Pauses auto-scroll if the user scrolls up to read history; resumes when they scroll back to bottom.

## Architecture

### Approach: React Context + shadcn Sheet

Follows the existing provider pattern (the app has 5 Context providers). A global `ActivityLogProvider` wraps the app in `layout.tsx`. The panel is a shadcn `Sheet` (side="right") rendered once in the layout. Pages call `useActivityLog()` to push entries.

### New Files

| File | Purpose |
|------|---------|
| `context/activity-log-context.tsx` | Provider, hook, log entry types, state management |
| `components/activity-log/ActivityLogPanel.tsx` | Sheet-based panel with scrollable console-style log |
| `components/activity-log/ActivityLogToggle.tsx` | Header toggle button with event count badge |
| `components/activity-log/LogEntry.tsx` | Single log entry — timestamp, icon, message, detail |

### Modified Files

| File | Change |
|------|--------|
| `app/layout.tsx` | Wrap children with `ActivityLogProvider`; render `ActivityLogPanel` inside `SidebarInset` |
| `app/projects/[id]/page.tsx` | Push staging SSE events (`room_started`, `variation_completed`, etc.) to activity log |
| `app/edit-image/components/EditorContainer.tsx` | Push edit start/complete/error events to activity log |
| `app/new-image/page.tsx` (via `ImageCreationContainer`) | Push generate events to activity log |
| `context/video-queue-context.tsx` | Push video job creation/progress/completion events to activity log |

## Data Model

### LogEntry

```typescript
interface LogEntry {
  id: string;             // crypto.randomUUID()
  timestamp: Date;        // when the event occurred
  level: 'info' | 'success' | 'error' | 'warn';
  message: string;        // primary text, e.g. "Generating variation 2/5 for Front Yard"
  detail?: string;        // secondary line, e.g. "gpt-image-2 · high quality · 1,240 tokens · 14.2s"
  icon?: string;          // emoji prefix: ▶ 🎨 ✓ ✕ 🔍 ⚡
}
```

### ActivityLogContext

```typescript
interface ActivityLogContextValue {
  entries: LogEntry[];
  log: (entry: Omit<LogEntry, 'id' | 'timestamp'>) => void;
  clear: () => void;
  isOpen: boolean;
  setOpen: (open: boolean) => void;
  hasActivity: boolean;   // entries.length > 0
}
```

## Component Design

### ActivityLogProvider (`context/activity-log-context.tsx`)

- Stores entries in a `useRef<LogEntry[]>` to avoid re-rendering the entire tree on every log push.
- Uses a `useState<number>` counter that increments on each `log()` call to trigger re-renders only in the panel.
- `log()` method: pushes entry to the ref array, increments counter, and auto-opens the panel if it's the first entry (entries were empty before this push).
- `clear()` method: empties the array, resets counter. Called by pages on mount via `useEffect` to scope logs to the current page.
- Max entries capped at 500 to prevent memory growth. Oldest entries are dropped when the cap is exceeded.

### ActivityLogPanel (`components/activity-log/ActivityLogPanel.tsx`)

- Uses shadcn `Sheet` with `side="right"` and controlled `open` state from context.
- Width: 380px (via Sheet's className).
- Header: "Activity Log" title, event count badge, clear button (trash icon), close button (X).
- Body: `ScrollArea` containing the list of `LogEntry` components.
- Footer: "Auto-scroll: on/off" indicator, connection status dot.
- Auto-scroll behavior:
  - Maintains a ref to the scroll container.
  - On new entry, if user is within 50px of the bottom, scroll to bottom.
  - If user has scrolled up, pause auto-scroll. Show a "↓ New events" button at the bottom to resume.
- Styling: `bg-[#0d1117]` background, monospace font (`font-mono`), tight line-height.

### ActivityLogToggle (`components/activity-log/ActivityLogToggle.tsx`)

- Renders in the app header (inside `SidebarInset` header div in `layout.tsx`).
- Icon: `Terminal` from lucide-react.
- Shows a small blue dot badge when `hasActivity && !isOpen` (new entries while panel is closed).
- Clicking toggles `setOpen(!isOpen)`.
- Only visible when `hasActivity` is true (no button shown on pages with no generation activity).

### LogEntry (`components/activity-log/LogEntry.tsx`)

- Layout: timestamp on first line (muted color), icon + message on second line (color-coded by level), optional detail on third line (smaller, muted).
- Color mapping:
  - `info` → `text-blue-400` (generating, in-progress actions)
  - `success` → `text-green-400` (completed, saved)
  - `error` → `text-red-400` (failures)
  - `warn` → `text-amber-400` (prompt adaptation, retries, warnings)
- Separator: thin `border-b border-white/5` between entries.

## Integration Points

### Staging Projects (`app/projects/[id]/page.tsx`)

The existing `handleStreamEvent` callback already processes SSE events. Add `useActivityLog()` calls alongside the existing logic:

| SSE Event | Log Entry |
|-----------|-----------|
| `room_started` | `info` — "▶ Starting generation for {room.label}" / "{n} variations queued" |
| `variation_completed` | `success` — "✓ Variation {idx+1}/{total} saved for {room.label}" / token + timing detail |
| `variation_failed` | `error` — "✕ Variation {idx+1}/{total} failed for {room.label}" / error message |
| `room_completed` | `success` — "✓ {room.label} complete" / "{n}/{total} variations succeeded" |
| `room_failed` | `error` — "✕ {room.label} failed" / error detail |
| `project_completed` | `success` — "✓ Generation complete" / summary stats |
| `error` | `error` — "✕ Generation error" / error message |

### Image Edit (`app/edit-image/components/EditorContainer.tsx`)

| Action | Log Entry |
|--------|-----------|
| Form submitted | `info` — "🎨 Editing image..." / "model · quality · size" |
| Response received | `success` — "✓ Image generated" / "tokens · timing" |
| Error caught | `error` — "✕ Edit failed" / error message |
| Save started | `info` — "💾 Saving to gallery..." / folder path |
| Save completed | `success` — "✓ Saved to gallery" / blob URL |

### New Image (`app/new-image/page.tsx` via `ImageCreationContainer`)

| Action | Log Entry |
|--------|-----------|
| Generation started | `info` — "🎨 Generating image..." / "model · n images · quality · size" |
| Response received | `success` — "✓ Image generated" / "tokens · timing" |
| Error caught | `error` — "✕ Generation failed" / error message |

### Video (`context/video-queue-context.tsx`)

| Action | Log Entry |
|--------|-----------|
| Job created | `info` — "🎬 Video generation started" / "model · duration" |
| Job progress | `info` — "🎬 Video processing..." / percentage or status |
| Job completed | `success` — "✓ Video ready" / "duration · size" |
| Job failed | `error` — "✕ Video generation failed" / error message |

## Backend Changes

**The backend SSE events already include sufficient data for detailed logging.** The existing `variation_completed` event includes `image_url`, and `variation_failed` includes `error`. However, the events currently lack timing and token data.

### Enhanced SSE Events (`backend/core/staging_pipeline.py`)

Add `elapsed_ms` and `tokens_used` to variation SSE events:

```python
yield {
    "type": "variation_completed" or "variation_failed",
    "room_id": room.id,
    "variation_index": idx,
    "image_url": variation.image_url,
    "error": variation.error,
    # NEW fields:
    "elapsed_ms": elapsed_ms,
    "tokens_used": token_usage.get("total_tokens") if token_usage else None,
    "model": project.settings.model,
}
```

Extract token usage from the pipeline result's generation response to populate `tokens_used`.

## Error Handling

- If `useActivityLog()` is called outside the provider (should not happen since it's in layout), return a no-op context that silently drops log calls.
- Log entries that fail to render (malformed data) are skipped with a console.warn, not displayed.
- The panel gracefully handles 0 entries (shows "No activity yet" placeholder).

## Testing

### Backend

- Unit test: verify enhanced SSE events include `elapsed_ms`, `tokens_used`, and `model` fields.

### Frontend

- Playwright E2E test: Navigate to a creation page, verify the toggle button appears, verify the panel opens, verify entries render with correct structure (timestamp, message, detail).
- Use mock API responses to simulate SSE events without real generation.

## Out of Scope

- Persisting log entries across page navigations or browser refreshes.
- Filtering or searching within the log.
- Exporting log entries.
- Log entries for non-generation actions (gallery browsing, settings changes, etc.).
