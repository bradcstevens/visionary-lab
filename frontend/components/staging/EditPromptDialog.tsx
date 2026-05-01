"use client"

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, AlertTriangle } from "lucide-react";

/**
 * Issue 004 of the projects-page-improvements PRD: Edit Prompt dialog.
 *
 * Lets the user edit the prompt that produced a variation and submit
 * it to the new ``/edit-prompt`` endpoint, which APPENDS a fresh
 * variation rather than mutating the existing one. The original
 * variation is preserved for A/B comparison.
 *
 * Default value precedence (matches PRD § Solution → 4):
 *   1. ``initialPrompt`` (the source variation's
 *      ``generation_metadata.adapted_prompt`` — the typical case for
 *      a completed variation).
 *   2. ``fallbackPrompt`` (the project-level prompt) — surfaces a
 *      visible notice so the user knows the dialog couldn't recover
 *      the original adapted prompt.
 *
 * Component contract: the parent CONDITIONALLY mounts this dialog
 * (``{target && <EditPromptDialog open ... />}``) so each open is a
 * fresh mount. The draft state is initialized from props at mount
 * time, which avoids the set-state-in-effect lint rule that the
 * useEffect-based reset would trigger. On submit failure the dialog
 * stays open and the parent does not unmount it, so the user's
 * draft is preserved across retry without any effect.
 */

interface EditPromptDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Source variation's ``generation_metadata.adapted_prompt``. May be empty. */
  initialPrompt?: string;
  /** Fallback (typically the project's top-level prompt) when initialPrompt is missing. */
  fallbackPrompt: string;
  /** 1-indexed display label, e.g. "Variation 3" — used for the dialog title. */
  variationLabel?: string;
  /** Whether a generation is currently in flight (disables Generate). */
  isBlocked?: boolean;
  /**
   * Submit handler — receives the user-typed prompt. The dialog stays
   * open while the promise is pending; on resolution the parent is
   * expected to call ``onOpenChange(false)``. On rejection the dialog
   * stays open with the user's draft preserved.
   */
  onSubmit: (adaptedPrompt: string) => Promise<void>;
}

export function EditPromptDialog({
  open,
  onOpenChange,
  initialPrompt,
  fallbackPrompt,
  variationLabel = "variation",
  isBlocked,
  onSubmit,
}: EditPromptDialogProps) {
  // ``usingFallback`` is true when we couldn't recover the original
  // adapted prompt. Drives the visible notice below the textarea so
  // the user understands why the prefilled text might not match what
  // they remember seeing.
  const sourcePrompt = initialPrompt && initialPrompt.trim().length > 0
    ? initialPrompt
    : fallbackPrompt;
  const usingFallback = !initialPrompt || initialPrompt.trim().length === 0;

  const [draft, setDraft] = useState<string>(sourcePrompt);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const trimmed = draft.trim();
  const canGenerate = trimmed.length > 0 && !isSubmitting && !isBlocked;

  const handleGenerate = async () => {
    if (!canGenerate) return;
    setIsSubmitting(true);
    try {
      await onSubmit(trimmed);
      // Parent closes the dialog on success via onOpenChange(false)
      // (which unmounts this component, so no draft reset is needed).
    } catch {
      // Keep dialog open with draft preserved so the user can retry.
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    if (isSubmitting) return;
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-xl"
        data-testid="edit-prompt-dialog"
        onInteractOutside={(e) => {
          if (isSubmitting) e.preventDefault();
        }}
      >
        <DialogHeader>
          <DialogTitle>Edit prompt for {variationLabel}</DialogTitle>
          <DialogDescription>
            Edit the prompt below and click Generate to create a new
            variation alongside the original. The original {variationLabel} stays
            unchanged so you can compare the two.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <Textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={8}
            disabled={isSubmitting}
            placeholder="Describe the variation you want to generate"
            data-testid="edit-prompt-textarea"
            aria-label="Adapted prompt"
          />
          {usingFallback && (
            <div
              className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-2 text-xs text-amber-700 dark:text-amber-300"
              data-testid="edit-prompt-fallback-notice"
            >
              <AlertTriangle aria-hidden="true" className="h-3.5 w-3.5 mt-0.5 shrink-0" />
              <span>
                We couldn&apos;t recover the original adapted prompt for this
                variation, so we&apos;ve prefilled the project&apos;s top-level prompt
                instead. Edit it as needed before clicking Generate.
              </span>
            </div>
          )}
        </div>
        <DialogFooter>
          <Button
            variant="ghost"
            onClick={handleCancel}
            disabled={isSubmitting}
            data-testid="edit-prompt-cancel"
          >
            Cancel
          </Button>
          <Button
            onClick={handleGenerate}
            disabled={!canGenerate}
            data-testid="edit-prompt-generate"
          >
            {isSubmitting && <Loader2 aria-hidden="true" className="h-4 w-4 mr-2 animate-spin" />}
            Generate
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
