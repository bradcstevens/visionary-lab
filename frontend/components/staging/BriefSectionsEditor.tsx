"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Save, Wand2, RotateCcw, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import {
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from "@/components/ui/tabs";
import {
  BRIEF_SECTIONS,
  type BriefSection,
} from "@/services/briefSectionRegistry";
import { composeBriefMarkdown } from "@/services/promptComposer";
import type {
  DesignBrief,
  StagingProject,
  UpdateProjectBody,
} from "@/services/stagingApi";

/**
 * Issue 019 of the image-pipeline-and-project-ux-overhaul PRD.
 *
 * Registry-driven sections + read-only preview + raw_override toggle
 * + post-save Regenerate button. The settings panel and the wizard
 * both expose the same eight `BRIEF_SECTIONS` so users see a single
 * coherent shape for the structured Design Brief.
 *
 * Save semantics
 * --------------
 * `onSave` receives a partial `UpdateProjectBody` that only carries
 * `design_brief.sections` and `design_brief.raw_override` plus the
 * other brief fields preserved verbatim from `project.design_brief`.
 * The backend's existing PATCH handler merges this against the
 * persisted document. NO regeneration jobs are created on save —
 * `onRegenerate` is a separate explicit affordance.
 *
 * Preview semantics
 * -----------------
 * The Preview tab shows the composed markdown of the currently
 * SAVED state (read from `project.design_brief`), not the unsaved
 * draft. Per the PRD: "preview updates after each save". This means
 * users see exactly what will hit the model — the composer used here
 * mirrors the backend's `compose_brief_markdown` byte-for-byte.
 *
 * Regenerate semantics
 * --------------------
 * The "Regenerate affected images" button is hidden by default and
 * appears only after the user has saved a change to a section
 * (or to `raw_override`). Clicking it invokes `onRegenerate` (which
 * the parent wires to `regenerateProjectJobs`). The button is
 * dismissed on click — re-edits of more sections will surface it
 * again on the next save.
 */

export interface BriefSectionsEditorProps {
  project: StagingProject;
  onSave: (updates: UpdateProjectBody) => Promise<unknown>;
  onRegenerate: () => Promise<unknown>;
  disabled?: boolean;
}

function readSavedSections(project: StagingProject): Record<string, string> {
  return { ...(project.design_brief?.sections ?? {}) };
}

function readSavedOverride(project: StagingProject): string | null {
  const v = project.design_brief?.raw_override;
  return v == null ? null : v;
}

function sectionsEqual(
  a: Record<string, string>,
  b: Record<string, string>,
): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const k of keys) {
    if ((a[k] ?? "") !== (b[k] ?? "")) return false;
  }
  return true;
}

export function BriefSectionsEditor({
  project,
  onSave,
  onRegenerate,
  disabled = false,
}: BriefSectionsEditorProps) {
  const savedSections = useMemo(() => readSavedSections(project), [project]);
  const savedOverride = useMemo(() => readSavedOverride(project), [project]);

  const [draftSections, setDraftSections] = useState<Record<string, string>>(
    () => readSavedSections(project),
  );
  // `useRawOverride` is the toggle state. Drives whether the saved
  // `raw_override` is non-null. Initialised from the project so a
  // legacy project that already has a non-null override opens with
  // the toggle ON.
  const [useRawOverride, setUseRawOverride] = useState<boolean>(
    () => savedOverride != null && savedOverride.length > 0,
  );
  const [draftOverride, setDraftOverride] = useState<string>(
    () => savedOverride ?? "",
  );
  const [isSaving, setIsSaving] = useState(false);
  const [activeTab, setActiveTab] = useState<string>(BRIEF_SECTIONS[0].id);
  // Surface "Regenerate affected images" only after a save that
  // actually moved either the sections or the raw_override. Cleared
  // on Regenerate click and on the next mid-edit (so the user
  // doesn't see a stale button after they've started typing again).
  const [showRegenerate, setShowRegenerate] = useState(false);

  // Keep the draft in sync when the parent swaps in a fresh project
  // (e.g., after a successful save the parent re-renders with the
  // server-returned doc). We use a serialized-prev-saved ref so the
  // draft is replaced ONLY when the saved values actually changed —
  // a parent re-render that didn't touch the brief does not stomp
  // the user's mid-edit draft. Mirrors the rising-edge ref pattern
  // in ProjectSettingsSheet's `prevOpenRef`; verified to not trigger
  // the `react-hooks/set-state-in-effect` lint rule under the guard.
  const savedKey = useMemo(
    () =>
      JSON.stringify([
        project.design_brief?.sections ?? {},
        project.design_brief?.raw_override ?? null,
      ]),
    [project.design_brief?.sections, project.design_brief?.raw_override],
  );
  const prevSavedKeyRef = useRef(savedKey);
  useEffect(() => {
    if (prevSavedKeyRef.current !== savedKey) {
      prevSavedKeyRef.current = savedKey;
      setDraftSections(readSavedSections(project));
      const newOverride = readSavedOverride(project);
      setDraftOverride(newOverride ?? "");
      setUseRawOverride(newOverride != null && newOverride.length > 0);
    }
  }, [savedKey, project]);

  const isDirty = useMemo(() => {
    const sectionsChanged = !sectionsEqual(draftSections, savedSections);
    const effectiveOverride: string | null = useRawOverride
      ? draftOverride
      : null;
    const overrideChanged =
      (effectiveOverride ?? "") !== (savedOverride ?? "");
    return sectionsChanged || overrideChanged;
  }, [
    draftSections,
    savedSections,
    useRawOverride,
    draftOverride,
    savedOverride,
  ]);

  const handleSectionChange = (id: string, value: string) => {
    setDraftSections((prev) => ({ ...prev, [id]: value }));
    if (showRegenerate) setShowRegenerate(false);
  };

  const handleOverrideChange = (value: string) => {
    setDraftOverride(value);
    if (showRegenerate) setShowRegenerate(false);
  };

  const handleToggleOverride = (next: boolean) => {
    setUseRawOverride(next);
    if (showRegenerate) setShowRegenerate(false);
  };

  const handleRevert = () => {
    setUseRawOverride(false);
    setDraftOverride("");
  };

  const handleSave = async () => {
    if (!isDirty || isSaving || disabled) return;
    setIsSaving(true);
    const effectiveOverride: string | null = useRawOverride
      ? draftOverride
      : null;
    // Build the design_brief patch by preserving every non-section /
    // non-override field already on the project. The backend treats
    // a missing field as "don't touch" but we send the full brief
    // here to keep parity with the existing wizard save path.
    const baseBrief: DesignBrief = project.design_brief ?? {
      global_instructions: project.prompt ?? "",
      object_palette: [],
      placement_guide: { back_row: "" },
      per_image_notes: {},
      per_image_objects: {},
      preserve_elements: [],
      settings: {
        variations_per_room: project.settings?.variations_per_room ?? 5,
        model: project.settings?.model ?? "gpt-image-2",
        quality: project.settings?.quality ?? "high",
        size: project.settings?.size ?? "auto",
      },
    };
    const updates: UpdateProjectBody = {
      design_brief: {
        ...baseBrief,
        sections: { ...draftSections },
        raw_override: effectiveOverride,
      },
    };
    try {
      await onSave(updates);
      setShowRegenerate(true);
    } catch {
      // Parent surfaces the error toast; keep the editor open with
      // the user's draft so they can retry.
    } finally {
      setIsSaving(false);
    }
  };

  const handleRegenerate = async () => {
    try {
      await onRegenerate();
    } finally {
      setShowRegenerate(false);
    }
  };

  // Preview reflects the EFFECTIVE current state — i.e., what would
  // be sent to the model if the user saved right now. After the
  // parent re-renders post-save, the sync effect above pulls the
  // saved values into the draft so the preview tracks the saved
  // state without a manual "snap-to-saved" branch. This also makes
  // the Revert affordance feel responsive: toggling off raw_override
  // immediately re-renders the preview with the structured
  // composition, no save round-trip required.
  const effectiveOverrideForPreview: string | null = useRawOverride
    ? draftOverride
    : null;
  const previewMarkdown = composeBriefMarkdown(
    draftSections,
    effectiveOverrideForPreview,
  );

  return (
    <div className="space-y-3" data-testid="brief-sections-editor">
      <div className="flex items-center justify-between gap-2">
        <Label className="text-sm font-medium">Design brief sections</Label>
        <div className="flex items-center gap-2">
          <Label
            htmlFor="brief-raw-override-toggle"
            className="text-xs text-muted-foreground"
          >
            Raw override
          </Label>
          <Switch
            id="brief-raw-override-toggle"
            data-testid="brief-raw-override-toggle"
            checked={useRawOverride}
            onCheckedChange={handleToggleOverride}
            disabled={disabled || isSaving}
          />
        </div>
      </div>

      {useRawOverride && (
        <div
          data-testid="brief-raw-override-banner"
          className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 dark:border-amber-900/30 dark:bg-amber-950/20 p-3 text-sm text-amber-900 dark:text-amber-200"
        >
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" aria-hidden="true" />
          <div className="flex-1 space-y-2">
            <p>
              Raw override is on. Your typed prompt will be sent verbatim and
              the eight structured sections will be ignored.
            </p>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={handleRevert}
              data-testid="brief-raw-override-revert"
              disabled={disabled || isSaving}
            >
              <RotateCcw className="h-3 w-3 mr-1.5" aria-hidden="true" />
              Revert to structured
            </Button>
          </div>
        </div>
      )}

      <Tabs
        value={activeTab}
        onValueChange={setActiveTab}
        className="w-full"
      >
        <TabsList className="flex-wrap h-auto">
          {BRIEF_SECTIONS.map((s: BriefSection) => (
            <TabsTrigger
              key={s.id}
              value={s.id}
              data-section-id={s.id}
              data-testid={`brief-section-tab-${s.id}`}
            >
              {s.title}
            </TabsTrigger>
          ))}
          <TabsTrigger value="__preview__" data-testid="brief-preview-tab">
            Preview
          </TabsTrigger>
        </TabsList>

        {useRawOverride ? (
          <TabsContent value={BRIEF_SECTIONS[0].id} className="mt-3" forceMount>
            <Label
              htmlFor="brief-raw-override-editor"
              className="text-xs text-muted-foreground"
            >
              Raw prompt sent to the model
            </Label>
            <Textarea
              id="brief-raw-override-editor"
              data-testid="brief-raw-override-editor"
              value={draftOverride}
              onChange={(e) => handleOverrideChange(e.target.value)}
              rows={10}
              disabled={disabled || isSaving}
              className="mt-1"
            />
          </TabsContent>
        ) : (
          BRIEF_SECTIONS.map((s) => (
            <TabsContent
              key={s.id}
              value={s.id}
              className="mt-3 space-y-2"
              forceMount
              hidden={activeTab !== s.id}
            >
              <p className="text-xs text-muted-foreground">{s.description}</p>
              <Textarea
                id={`brief-section-editor-${s.id}`}
                data-testid={`brief-section-editor-${s.id}`}
                value={draftSections[s.id] ?? ""}
                onChange={(e) => handleSectionChange(s.id, e.target.value)}
                rows={6}
                disabled={disabled || isSaving}
              />
            </TabsContent>
          ))
        )}

        <TabsContent
          value="__preview__"
          className="mt-3"
          forceMount
          hidden={activeTab !== "__preview__"}
        >
          <p className="text-xs text-muted-foreground mb-2">
            This is exactly what will be sent to the model. Updates after
            each save.
          </p>
          <pre
            data-testid="brief-preview-content"
            className="whitespace-pre-wrap break-words rounded-md border bg-muted/40 p-3 text-xs font-mono"
          >
            {previewMarkdown || "(empty — author at least one section)"}
          </pre>
        </TabsContent>
      </Tabs>

      <div className="flex items-center justify-between gap-2 pt-2">
        <div className="flex items-center gap-2">
          {showRegenerate && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleRegenerate}
              data-testid="brief-regenerate-button"
              disabled={disabled}
            >
              <Wand2 className="h-3 w-3 mr-1.5" aria-hidden="true" />
              Regenerate affected images
            </Button>
          )}
        </div>
        <Button
          type="button"
          size="sm"
          onClick={handleSave}
          disabled={!isDirty || isSaving || disabled}
          data-testid="brief-sections-save"
        >
          {isSaving ? (
            <Loader2 className="h-3 w-3 mr-1.5 animate-spin" aria-hidden="true" />
          ) : (
            <Save className="h-3 w-3 mr-1.5" aria-hidden="true" />
          )}
          Save sections
        </Button>
      </div>
    </div>
  );
}
