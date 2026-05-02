"use client";

/**
 * Collapsible prompt header for the project detail page.
 *
 * Issue 014 of the image-pipeline-and-project-ux-overhaul PRD.
 *
 * Renders the project's ``prompt_summary`` collapsed by default with a
 * "Show full prompt" affordance. Expanding swaps the summary for an
 * editable textarea + Save / Cancel buttons. ``onSave`` is the only
 * side-effect surface — the project page wires it to the existing
 * PATCH /projects/{id} call (no regeneration jobs are enqueued; the
 * backend regenerates ``prompt_summary`` server-side and the parent
 * re-renders this component with the refreshed prop on success).
 *
 * The component is intentionally headless of any data-fetching or
 * job-enqueue surface so the AC bullet "no regeneration jobs are
 * created on save" is structurally guaranteed at this layer.
 */

import { useState } from "react";
import { ChevronDown, ChevronUp, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

const COLLAPSED_FALLBACK_MAX = 240;

export interface CollapsiblePromptProps {
  prompt: string;
  promptSummary?: string | null;
  onSave: (newPrompt: string) => Promise<void>;
  disabled?: boolean;
}

export function CollapsiblePrompt({
  prompt,
  promptSummary,
  onSave,
  disabled,
}: CollapsiblePromptProps) {
  const [expanded, setExpanded] = useState(false);
  const [draft, setDraft] = useState(prompt);
  const [saving, setSaving] = useState(false);

  // Note: ``draft`` is intentionally NOT auto-synced to ``prompt`` via
  // an effect. ``handleToggle`` resets ``draft`` to the live ``prompt``
  // every time the editor is opened, and ``handleSave`` collapses
  // before the parent re-renders with the new prompt — so the user
  // never sees stale ``draft`` content. This avoids the
  // setState-in-effect smell that lint rightly flags.

  const trimmedDraft = draft.trim();
  const dirty = trimmedDraft.length > 0 && draft !== prompt;

  const collapsedText = (() => {
    const summary = (promptSummary ?? "").trim();
    if (summary.length > 0) return summary;
    if (prompt.length <= COLLAPSED_FALLBACK_MAX) return prompt;
    return `${prompt.slice(0, COLLAPSED_FALLBACK_MAX).trimEnd()}…`;
  })();

  const hasContent = collapsedText.length > 0 || prompt.length > 0;

  const handleToggle = () => {
    if (expanded) {
      // Collapse: discard any in-progress edits and reset the draft.
      setDraft(prompt);
      setExpanded(false);
    } else {
      setDraft(prompt);
      setExpanded(true);
    }
  };

  const handleSave = async () => {
    if (!dirty || saving) return;
    setSaving(true);
    try {
      await onSave(draft);
      // Successful save: collapse so the user sees the refreshed summary
      // (parent will pass the new ``promptSummary`` on its next render).
      setExpanded(false);
    } catch {
      // Keep the editor open on failure so the user can adjust + retry.
      // Toast/error surfacing is the parent's responsibility.
    } finally {
      setSaving(false);
    }
  };

  if (!expanded) {
    return (
      <div className="space-y-1 max-w-3xl">
        <p
          data-testid="prompt-summary"
          className="text-muted-foreground leading-relaxed"
        >
          {collapsedText}
        </p>
        {hasContent && (
          <Button
            type="button"
            variant="link"
            size="sm"
            className="px-0 h-auto text-xs text-muted-foreground hover:text-foreground"
            onClick={handleToggle}
            data-testid="prompt-toggle"
            disabled={disabled}
          >
            <ChevronDown className="h-3 w-3 mr-1" />
            Show full prompt
          </Button>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-2 max-w-3xl" data-testid="prompt-editor-region">
      <Textarea
        data-testid="prompt-editor"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        rows={8}
        disabled={saving || disabled}
        className="resize-y font-normal"
      />
      <div className="flex items-center gap-2">
        <Button
          type="button"
          size="sm"
          onClick={handleSave}
          disabled={!dirty || saving || disabled}
          data-testid="prompt-save"
        >
          {saving && <Loader2 className="h-3 w-3 mr-1 animate-spin" />}
          Save
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={handleToggle}
          disabled={saving}
          data-testid="prompt-cancel"
        >
          Cancel
        </Button>
        <Button
          type="button"
          variant="link"
          size="sm"
          className="ml-auto px-0 h-auto text-xs text-muted-foreground"
          onClick={handleToggle}
          disabled={saving}
        >
          <ChevronUp className="h-3 w-3 mr-1" />
          Hide
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Saving updates the prompt and refreshes its summary. It does not
        regenerate any images.
      </p>
    </div>
  );
}
