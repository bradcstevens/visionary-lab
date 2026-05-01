"use client"

import { useMemo, useState } from "react";
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
 */

const MODEL_OPTIONS: { value: string; label: string }[] = [
  { value: "gpt-image-2", label: "GPT Image 2 (default)" },
  { value: "gpt-image-1-mini", label: "GPT Image 1 mini" },
  { value: "flux-kontext-pro", label: "Flux Kontext Pro" },
];

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
}

/**
 * Compute the diff between current form values and the original project
 * values. Returns only the keys that actually changed. Used both to
 * decide whether the Save button is enabled and to build the PATCH
 * body so the request stays minimal.
 *
 * Pure helper — exported only for testing.
 */
export function computeProjectSettingsDiff(
  initial: { name: string; prompt: string; variations_per_room: number; model: string; quality: string; size: string },
  current: { name: string; prompt: string; variations_per_room: number; model: string; quality: string; size: string },
): UpdateProjectBody {
  const diff: UpdateProjectBody = {};
  if (current.name.trim() !== initial.name) diff.name = current.name.trim();
  if (current.prompt.trim() !== initial.prompt) diff.prompt = current.prompt.trim();
  const settingsDiff: Partial<{ variations_per_room: number; model: string; quality: string; size: string }> = {};
  if (current.variations_per_room !== initial.variations_per_room) {
    settingsDiff.variations_per_room = current.variations_per_room;
  }
  if (current.model !== initial.model) settingsDiff.model = current.model;
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
}: ProjectSettingsSheetProps) {
  // Snapshot the project values when the sheet opens. The "initial"
  // values stay frozen for the duration of the sheet's open lifecycle
  // so the diff comparison is stable even if the parent reloads
  // ``project`` mid-edit (e.g., a regen completed in the background).
  const [initialValues, setInitialValues] = useState(() =>
    snapshotFromProject(project),
  );

  // Form state.
  const [name, setName] = useState(initialValues.name);
  const [prompt, setPrompt] = useState(initialValues.prompt);
  const [variationsPerRoom, setVariationsPerRoom] = useState(
    initialValues.variations_per_room,
  );
  const [model, setModel] = useState(initialValues.model);
  const [quality, setQuality] = useState(initialValues.quality);
  const [size, setSize] = useState(initialValues.size);
  const [isSaving, setIsSaving] = useState(false);

  const diff = useMemo(() => {
    const current = { name, prompt, variations_per_room: variationsPerRoom, model, quality, size };
    return computeProjectSettingsDiff(initialValues, current);
  }, [initialValues, name, prompt, variationsPerRoom, model, quality, size]);
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

  // Re-snapshot the project values on each open so the form starts
  // from the current persisted values, not whatever was left over
  // from a previous open. Done in onOpenChange (NOT useEffect) to
  // sidestep the ``react-hooks/set-state-in-effect`` lint rule and
  // to make the trigger explicit.
  const handleOpenChange = (next: boolean) => {
    if (next) {
      const snap = snapshotFromProject(project);
      setInitialValues(snap);
      setName(snap.name);
      setPrompt(snap.prompt);
      setVariationsPerRoom(snap.variations_per_room);
      setModel(snap.model);
      setQuality(snap.quality);
      setSize(snap.size);
    }
    onOpenChange(next);
  };

  const handleSave = async () => {
    if (!hasChanges || validationError || isSaving) return;
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

  return (
    <Sheet open={open} onOpenChange={handleOpenChange}>
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
          </div>

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

          <div className="space-y-2">
            <Label htmlFor="project-settings-model">Model</Label>
            <Select value={model} onValueChange={setModel} disabled={isSaving}>
              <SelectTrigger
                id="project-settings-model"
                data-testid="project-settings-model-select"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {MODEL_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
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
            disabled={!hasChanges || !!validationError || isSaving}
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
    prompt: project.prompt ?? "",
    variations_per_room: project.settings?.variations_per_room ?? 5,
    model: project.settings?.model ?? "gpt-image-2",
    quality: project.settings?.quality ?? "high",
    size: project.settings?.size ?? "auto",
  };
}
