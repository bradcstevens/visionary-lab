"use client"

import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Save, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import type {
  StagingProject,
  UpdateProjectBody,
} from "@/services/stagingApi";
import { ProjectRoomsManager } from "./ProjectRoomsManager";

/**
 * Project Settings side sheet — issue 002 of the projects-page-
 * improvements PRD.
 *
 * Lets the user edit `name`, `prompt`, and `StagingSettings`
 * (variations_per_room, model, quality, size). Saving applies changes
 * to FUTURE generations only — every existing variation and its prompt
 * stays exactly as it was. The notice banner at the top makes this
 * explicit.
 *
 * The sheet only sends the fields that actually changed (dirty
 * tracking). The backend's `PATCH /staging/projects/{id}` interprets
 * absent fields as "don't touch" and partial `settings` objects as
 * key-by-key MERGE — so sending `{settings: {variations_per_room: 3}}`
 * doesn't clobber `model`/`quality`/`size`.
 *
 * Design brief deferral: the PRD lists "design brief" as part of the
 * editable surfaces. The DesignBriefEditor is a heavy tabbed component
 * that needs its own slice to integrate cleanly into the right-side
 * Sheet layout. The backend endpoint already accepts `design_brief`,
 * so a follow-up can wire the editor without backend churn. The
 * existing AI Design Session step in the wizard remains the path for
 * brief edits today.
 *
 * Issue 002 of the project-settings-completeness PRD: the displayed
 * "prompt" is now derived from the design brief's `global_instructions`
 * when one exists (with the same `_is_nonempty_str` gate the backend
 * mirror uses), falling back to the legacy `project.prompt` field
 * otherwise. The save path itself does NOT change — we always send the
 * top-level `prompt` field on edit, and the backend mirror in
 * `_mirror_prompt_and_brief_in_place` (`backend/api/endpoints/staging.py`)
 * propagates the value into `design_brief.global_instructions` when a
 * brief exists. This keeps the client's payload small, dodges the
 * stale-brief clobber risk that "send a full design_brief on every
 * prompt edit" would create, and means Settings, the page header, the
 * Brief tab, the gallery's per-image edit dialog (via
 * `EditPromptDialog.fallbackPrompt`), and any future snapshot/restore
 * path all see one coherent "prompt" with a single source of truth on
 * the server. When no brief exists yet, a small hint below the prompt
 * textarea explains that future edits will be stored on the brief once
 * one is created.
 *
 * Issue 004 of the project-settings-completeness PRD: a new
 * `ProjectRoomsManager` is mounted between the project-level fields
 * and the generation settings. It encapsulates the rooms list and
 * inline rename today; subsequent slices add inline delete (issue 005)
 * and add-photos (issue 006). Room operations persist immediately per
 * action (rather than deferring to the project-level Save button) and
 * notify the parent via the new `onProjectUpdate` callback so the page
 * can resync local state — same pattern as `handleProjectSettingsSave`.
 *
 * Issue 003 of the project-settings-completeness PRD: two changes to
 * the generation-settings half of the sheet.
 *
 *   1. The "Model" field is now rendered as a READ-ONLY label, not an
 *      interactive Select. Users discover their project's model value
 *      here but can't switch it from Settings (model selection lives
 *      with the wizard's create-project step). The display label is
 *      looked up in `MODEL_DISPLAY_LABELS` with the raw model value as
 *      a fallback so a backend-side model addition that hasn't been
 *      mapped on the client yet still renders something readable
 *      instead of a blank field.
 *
 *   2. `computeProjectSettingsDiff` no longer accepts `model` in its
 *      input shape — a STRUCTURAL guarantee (not just a runtime
 *      branch) that the wire payload never carries `settings.model`,
 *      even defensively against a future bug. The pre-issue-003 diff
 *      helper carried `model` only because the form state included it;
 *      now neither does.
 *
 * Issue 007 of the project-settings-completeness PRD: "disable rules
 * during in-flight generation". When `project.status === 'processing'`,
 * the sheet:
 *
 *   - Disables the Save button regardless of `isDirty` (the user can
 *     still type and build a draft, but cannot persist mid-flight).
 *   - Forwards `disabled={isGenerating || isSaving}` to
 *     `ProjectRoomsManager`, which interprets that as "block add and
 *     delete; rename remains enabled" (see the manager's source-file
 *     comment on the narrowed `disabled` semantics).
 *   - Leaves the name input, prompt textarea, and quality / size
 *     dropdowns interactively editable, since local-only edits don't
 *     race the pipeline.
 *   - Renders an inline notice explaining why Save is disabled, so
 *     the user can tell the difference between "no changes" and
 *     "blocked by generation".
 *
 * The page-level overflow menu already gates "Project settings" on
 * the fleet-derived `isAnyInFlight`, but `project.status` is the
 * canonical signal per the issue 007 contract — and a stale-processing
 * project (status='processing' with no live fleet activity) can still
 * reach this sheet via that path. Enforcing the rule HERE makes the
 * gating authoritative regardless of how the sheet is reached.
 *
 * The snapshot reset on open was also moved out of `handleOpenChange`
 * (which only runs for Radix-internal close events — Esc, click
 * outside, X — and the explicit Cancel button) and into a rising-edge
 * `useEffect` keyed on `open` going false → true. The pre-issue-003
 * shape only reset when `next === true` AND Radix actually fired
 * `onOpenChange`, but with a parent-driven open path
 * (setShowSettingsSheet(true) flips the controlled `open` prop without
 * an internal Radix event), the reset never ran on the user's typical
 * "open via overflow menu" flow. Result: after Cancel, the next open
 * showed the user's discarded draft, not the persisted values — the
 * exact "Discard changes" behavior the issue 003 AC pins. The new
 * effect mirrors the EditPromptDialog pattern from
 * `radix-dialog-body-lock-fix` issue 003 (commit 0e75717): a
 * `prevOpenRef` makes the false→true transition the only trigger, so
 * mid-edit `project` refreshes (e.g., a regen completing in the
 * background) cannot stomp the user's draft. The `react-hooks/
 * set-state-in-effect` rule does NOT fire on this guarded shape —
 * empirically verified, same as the EditPromptDialog conversion.
 */

/**
 * Display labels for known model values, used by the read-only model
 * label per issue 003. Unknown values fall back to the raw model
 * string so a backend-side addition still renders something readable
 * instead of a blank field.
 */
const MODEL_DISPLAY_LABELS: Record<string, string> = {
  "gpt-image-2": "GPT Image 2 (default)",
  "gpt-image-1-mini": "GPT Image 1 mini",
  "flux-kontext-pro": "Flux Kontext Pro",
};

const QUALITY_OPTIONS: { value: string; label: string }[] = [
  { value: "low", label: "Low (fastest)" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High (default)" },
  { value: "auto", label: "Auto" },
];

const SIZE_OPTIONS: { value: string; label: string }[] = [
  { value: "auto", label: "Auto (default)" },
  { value: "1024x1024", label: "1024 × 1024 (square)" },
  { value: "1536x1024", label: "1536 × 1024 (landscape)" },
  { value: "1024x1536", label: "1024 × 1536 (portrait)" },
];

export interface ProjectSettingsSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  project: StagingProject;
  onSave: (updates: UpdateProjectBody) => Promise<void>;
  // Issue 004 of project-settings-completeness PRD: room operations
  // (rename today; delete/add in subsequent slices) persist
  // immediately per action via the room-scoped API endpoints, NOT
  // through the project-level Save button. The new ProjectRoomsManager
  // calls this callback after each successful room mutation so the
  // parent page can resync local project state (running its existing
  // SAS-token resolution before the setProject swap, same pattern as
  // handleProjectSettingsSave). Optional so existing callers don't
  // have to wire it on day one — but if absent, room renames will
  // succeed on the server and be invisible in the local UI until the
  // next reload.
  onProjectUpdate?: (project: StagingProject) => void | Promise<void>;
}

/**
 * Mirror of the backend's `_is_nonempty_str` gate
 * (`backend/api/endpoints/staging.py:436-442`): a value is "real" only
 * when it's a string AND has at least one non-whitespace character.
 * Used to decide whether `design_brief.global_instructions` is the
 * canonical prompt or whether to fall back to `project.prompt`. Pure;
 * exported only for direct testing.
 */
export function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

/**
 * Derive the prompt the Settings sheet should display for this project,
 * matching the backend mirror's view of "the canonical prompt".
 *
 * Precedence (issue 002 of project-settings-completeness):
 *   1. `project.design_brief.global_instructions` when it has real
 *      (non-whitespace) content — the user's authored AI Design Session
 *      output is canonical.
 *   2. `project.prompt` otherwise — the legacy field, what the wizard
 *      seeded with `"Draft — pending AI Design Session"` for new
 *      projects.
 *   3. `""` if neither is available.
 *
 * The whitespace-only fallback at step 1 mirrors the backend's
 * `_is_nonempty_str` gate so a project with a brief that contains only
 * whitespace `global_instructions` (which the backend mirror treats as
 * "no real prompt yet") behaves identically on the client and server —
 * the user sees the legacy `project.prompt` rather than visually-empty
 * whitespace they would silently overwrite on first save.
 *
 * Pure; exported for vitest coverage in
 * `__tests__/ProjectSettingsSheet.test.ts`.
 */
export function derivePromptForSettings(project: StagingProject): string {
  const brief = project.design_brief;
  if (brief && isNonEmptyString(brief.global_instructions)) {
    return brief.global_instructions;
  }
  return project.prompt ?? "";
}

/**
 * Compute the diff between current form values and the original project
 * values. Returns only the keys that actually changed. Used both to
 * decide whether the Save button is enabled and to build the PATCH
 * body so the request stays minimal.
 *
 * Issue 003 of the project-settings-completeness PRD: the input shape
 * deliberately OMITS `model`. This is the structural "never include
 * model" guarantee — even if a future bug or programmatic consumer
 * tries to pass a model value through here, it cannot reach the wire
 * payload because the type signature won't accept it. The pre-issue-
 * 003 shape carried `model` because the form state did; now both are
 * read-only display surfaces. The vitest test
 * `the helper's input shape does not accept a model field` uses
 * `@ts-expect-error` to pin this guarantee at the type boundary.
 *
 * Pure helper — exported only for testing.
 */
export function computeProjectSettingsDiff(
  initial: { name: string; prompt: string; variations_per_room: number; quality: string; size: string },
  current: { name: string; prompt: string; variations_per_room: number; quality: string; size: string },
): UpdateProjectBody {
  const diff: UpdateProjectBody = {};
  if (current.name.trim() !== initial.name) diff.name = current.name.trim();
  if (current.prompt.trim() !== initial.prompt) diff.prompt = current.prompt.trim();
  const settingsDiff: Partial<{ variations_per_room: number; quality: string; size: string }> = {};
  if (current.variations_per_room !== initial.variations_per_room) {
    settingsDiff.variations_per_room = current.variations_per_room;
  }
  if (current.quality !== initial.quality) settingsDiff.quality = current.quality;
  if (current.size !== initial.size) settingsDiff.size = current.size;
  if (Object.keys(settingsDiff).length > 0) {
    diff.settings = settingsDiff;
  }
  return diff;
}

export function ProjectSettingsSheet({
  open,
  onOpenChange,
  project,
  onSave,
  onProjectUpdate,
}: ProjectSettingsSheetProps) {
  // Snapshot the project values when the sheet opens. The "initial"
  // values stay frozen for the duration of the sheet's open lifecycle
  // so the diff comparison is stable even if the parent reloads
  // ``project`` mid-edit (e.g., a regen completed in the background).
  // The model is read from this snapshot too (issue 003: it's
  // displayed as a frozen-on-open read-only label, not live, so the
  // sheet stays internally consistent if the project is refreshed
  // mid-edit).
  const [initialValues, setInitialValues] = useState(() =>
    snapshotFromProject(project),
  );

  // Form state — model is intentionally NOT in the form state per
  // issue 003 (read-only display only, lives in initialValues).
  const [name, setName] = useState(initialValues.name);
  const [prompt, setPrompt] = useState(initialValues.prompt);
  const [variationsPerRoom, setVariationsPerRoom] = useState(
    initialValues.variations_per_room,
  );
  const [quality, setQuality] = useState(initialValues.quality);
  const [size, setSize] = useState(initialValues.size);
  const [isSaving, setIsSaving] = useState(false);

  // Issue 007 of project-settings-completeness: when the project is
  // actively generating, destructive / project-mutating Save actions
  // are blocked but local editing (typing in name/prompt textarea,
  // changing dropdowns, renaming a room) is allowed. The user can
  // build a draft they keep but cannot persist mid-generation. The
  // overflow menu on the page already gates "Project settings" on
  // the fleet-derived `isAnyInFlight`, but a stale-processing project
  // (status='processing' with no live fleet activity) can still reach
  // this sheet — and a programmatic consumer that bypasses the menu
  // could too. So we enforce the rule authoritatively HERE on the
  // Save button + the rooms manager's destructive affordances rather
  // than relying solely on the menu gate. `project.status` is the
  // canonical signal per the issue 007 contract.
  const isGenerating = project.status === "processing";

  // Rising-edge snapshot reset — issue 003 of project-settings-
  // completeness. Pre-issue-003 the reset lived in handleOpenChange,
  // which only ran on Radix-internal close events (Esc, click
  // outside, X) and the explicit Cancel button. Parent-driven opens
  // (the user's typical "More actions" → "Project settings" flow,
  // which flips the controlled `open` prop via setShowSettingsSheet)
  // do NOT fire `onOpenChange` and so never re-ran the reset. Result:
  // after Cancel, the next open showed the user's discarded draft,
  // not the persisted values. This effect makes the false→true
  // transition the canonical reset trigger so Cancel + reopen, Esc +
  // reopen, X + reopen, and click-outside + reopen all behave the
  // same. The `prevOpenRef` ensures subsequent renders while
  // `open === true` do NOT stomp the user's in-progress draft (e.g.,
  // a `project` refresh from a background regen). Mirrors the
  // EditPromptDialog pattern from radix-dialog-body-lock-fix issue
  // 003 (commit 0e75717); see that file for the empirical note that
  // `react-hooks/set-state-in-effect` does NOT fire on this guarded
  // shape.
  const prevOpenRef = useRef(false);
  useEffect(() => {
    if (open && !prevOpenRef.current) {
      const snap = snapshotFromProject(project);
      setInitialValues(snap);
      setName(snap.name);
      setPrompt(snap.prompt);
      setVariationsPerRoom(snap.variations_per_room);
      setQuality(snap.quality);
      setSize(snap.size);
      setIsSaving(false);
    }
    prevOpenRef.current = open;
  }, [open, project]);

  const diff = useMemo(() => {
    const current = { name, prompt, variations_per_room: variationsPerRoom, quality, size };
    return computeProjectSettingsDiff(initialValues, current);
  }, [initialValues, name, prompt, variationsPerRoom, quality, size]);
  const hasChanges = Object.keys(diff).length > 0;
  // Validate locally so we don't surface a 422 from the server for
  // obvious mistakes. Empty name/prompt or out-of-range
  // variations_per_room blocks Save with a hint.
  const validationError = (() => {
    if (!name.trim()) return "Project name is required.";
    if (!prompt.trim()) return "Top-level prompt is required.";
    if (variationsPerRoom < 1 || variationsPerRoom > 10) {
      return "Variations per room must be between 1 and 10.";
    }
    return null;
  })();

  const handleSave = async () => {
    if (!hasChanges || validationError || isSaving || isGenerating) return;
    setIsSaving(true);
    try {
      await onSave(diff);
      onOpenChange(false);
    } catch {
      // Parent's onSave is responsible for surfacing the error toast.
      // We keep the sheet open so the user's draft isn't lost.
    } finally {
      setIsSaving(false);
    }
  };

  // Display label for the read-only model field (issue 003). Looks
  // up the human-readable string in MODEL_DISPLAY_LABELS, falling
  // back to the raw model value so a backend-side addition that
  // hasn't been mapped yet still renders something readable instead
  // of a blank field.
  const modelDisplayLabel =
    MODEL_DISPLAY_LABELS[initialValues.model] ?? initialValues.model;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="sm:max-w-lg flex flex-col"
        data-testid="project-settings-sheet"
      >
        <SheetHeader>
          <SheetTitle>Project settings</SheetTitle>
          <SheetDescription>
            Update the name, prompt, or generation settings for this project.
          </SheetDescription>
        </SheetHeader>

        {/* Notice banner — the "future generations only" hint per the
            PRD's Solution → 2 paragraph and User Story 9. */}
        <div
          className="mx-4 rounded-md border border-amber-200 bg-amber-50 dark:border-amber-900/30 dark:bg-amber-950/20 p-3 text-sm text-amber-900 dark:text-amber-200 flex gap-2"
          data-testid="project-settings-future-only-notice"
        >
          <Info className="h-4 w-4 shrink-0 mt-0.5" aria-hidden="true" />
          <span>
            Changes apply to <strong>future generations only</strong>. Your
            existing variations stay exactly as they are.
          </span>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          <div className="space-y-2">
            <Label htmlFor="project-settings-name">Project name</Label>
            <Input
              id="project-settings-name"
              data-testid="project-settings-name-input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={isSaving}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="project-settings-prompt">Top-level prompt</Label>
            <Textarea
              id="project-settings-prompt"
              data-testid="project-settings-prompt-textarea"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
              disabled={isSaving}
            />
            {/* Issue 002 of project-settings-completeness: when no
                design brief exists yet, the canonical prompt lives in
                the legacy project.prompt field — explain that future
                edits will be stored on the brief once one is created.
                The hint matches the asymmetric derivation rule in
                derivePromptForSettings: it is only shown when
                `design_brief` is null/undefined (NOT when the brief
                exists but has whitespace-only global_instructions —
                that case still falls back to project.prompt for
                display, but the user has already started a brief and
                doesn't need the explainer). */}
            {!project.design_brief && (
              <p
                className="text-xs text-muted-foreground"
                data-testid="project-settings-prompt-brief-hint"
              >
                Once a design brief exists, your prompt is stored as part of it.
              </p>
            )}
          </div>

          {/* Issue 004 of project-settings-completeness PRD: rooms
              manager scaffold + inline rename. Mounted between
              "Project details" (name, prompt) and "Generation
              settings" (variations, model, quality, size) per the
              PRD's "mount the new rooms manager between Project
              details and Generation settings" placement rule.

              Issue 007: `disabled` covers add+delete only — rename
              is intentionally always reachable inside the manager.
              We OR `isGenerating` (status === 'processing') with the
              local `isSaving` so destructive room ops are blocked
              both during a project-level save AND during in-flight
              generation. The `onProjectUpdate` is forwarded from the
              parent page and resyncs local project state after a
              server-confirmed room mutation. */}
          <ProjectRoomsManager
            project={project}
            onProjectUpdate={onProjectUpdate ?? (() => {})}
            disabled={isGenerating || isSaving}
          />

          <div className="space-y-2">
            <Label htmlFor="project-settings-variations">
              Variations per room (1–10)
            </Label>
            <Input
              id="project-settings-variations"
              data-testid="project-settings-variations-input"
              type="number"
              min={1}
              max={10}
              value={variationsPerRoom}
              onChange={(e) => {
                const parsed = parseInt(e.target.value, 10);
                setVariationsPerRoom(Number.isFinite(parsed) ? parsed : 0);
              }}
              disabled={isSaving}
            />
          </div>

          {/* Issue 003 of project-settings-completeness PRD: model
              is rendered as a READ-ONLY label, not a Select. The
              user discovers their project's model value here but
              can't switch it from Settings. The display text
              prefers the human-readable form from
              MODEL_DISPLAY_LABELS, falling back to the raw model
              value so an unmapped model still renders something
              readable. ARIA: aria-readonly="true" tells assistive
              tech this is a read-only display, not an interactive
              control. The frozen-on-open snapshot pattern means a
              mid-edit project refresh (e.g., a background regen)
              cannot change the displayed model out from under the
              user. */}
          <div className="space-y-2">
            <Label htmlFor="project-settings-model">Model</Label>
            <div
              id="project-settings-model"
              data-testid="project-settings-model-readonly"
              aria-readonly="true"
              className="flex items-center justify-between rounded-md border border-input bg-muted/40 px-3 py-2 text-sm text-foreground"
            >
              <span>{modelDisplayLabel}</span>
              <span className="text-xs text-muted-foreground">Read-only</span>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="project-settings-quality">Quality</Label>
            <Select value={quality} onValueChange={setQuality} disabled={isSaving}>
              <SelectTrigger
                id="project-settings-quality"
                data-testid="project-settings-quality-select"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {QUALITY_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="project-settings-size">Image size</Label>
            <Select value={size} onValueChange={setSize} disabled={isSaving}>
              <SelectTrigger
                id="project-settings-size"
                data-testid="project-settings-size-select"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SIZE_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {validationError && (
            <p
              className="text-sm text-destructive"
              data-testid="project-settings-validation-error"
            >
              {validationError}
            </p>
          )}

          {/* Issue 007 of project-settings-completeness: surface why
              the Save button is disabled while a generation is in
              flight. Local edits are still allowed; the user keeps
              their draft. */}
          {isGenerating && (
            <p
              className="text-xs text-muted-foreground"
              data-testid="project-settings-generating-notice"
            >
              Save is disabled while this project is generating. Your
              edits are kept as a draft until generation completes.
            </p>
          )}
        </div>

        <SheetFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isSaving}
            data-testid="project-settings-cancel"
          >
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={!hasChanges || !!validationError || isSaving || isGenerating}
            data-testid="project-settings-save"
          >
            {isSaving ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            Save
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}

function snapshotFromProject(project: StagingProject) {
  return {
    name: project.name ?? "",
    // Issue 002 of project-settings-completeness: prefer the design
    // brief's global_instructions when present (with the same
    // _is_nonempty_str gate the backend mirror uses). Falls back to the
    // legacy project.prompt for projects with no brief.
    prompt: derivePromptForSettings(project),
    variations_per_room: project.settings?.variations_per_room ?? 5,
    model: project.settings?.model ?? "gpt-image-2",
    quality: project.settings?.quality ?? "high",
    size: project.settings?.size ?? "auto",
  };
}
