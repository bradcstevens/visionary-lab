"use client"

import { useEffect, useRef, useState } from "react";
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
 * Component contract (issue 003 of radix-dialog-body-lock-fix PRD):
 * this dialog is ALWAYS MOUNTED by its parent and driven by the
 * controlled ``open`` prop. The parent does NOT wrap the dialog in
 * ``{target && <EditPromptDialog ... />}`` — that prior pattern was
 * the structural cause of an unmount-while-Radix-is-still-cleaning-up
 * landmine that left the page non-interactive after every close
 * (the body-lock leak).
 *
 * Draft state is reset on the rising edge of ``open`` via the effect
 * below. The ESLint ``react-hooks/set-state-in-effect`` rule does NOT
 * fire on this pattern (verified empirically — the rule has
 * heuristics that recognize a guarded rising-edge reset and treat it
 * as the documented "adjust state when a prop changes" escape
 * hatch). The React docs explicitly endorse this pattern (see
 * https://react.dev/reference/react/useEffect#adjusting-some-state-when-a-prop-changes
 * — the documented escape hatch for the rare case where a prop
 * change must trigger a state reset and a key-based remount is not
 * acceptable).
 *
 * On submit failure the dialog stays open (the parent does not
 * close it via onOpenChange), so the user's draft is preserved for
 * retry. On submit success the parent calls ``onOpenChange(false)``
 * which leaves the dialog mounted but closed — the next open on
 * any variation re-derives ``sourcePrompt`` from the latest props
 * and resets the draft fresh.
 *
 * DO NOT re-introduce conditional mounting. The body-lock guard at
 * the layout level (see ``frontend/components/BodyLockGuard.tsx``)
 * is a defense-in-depth backstop, NOT a license to revert this
 * mount pattern.
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

  // Open-edge reset (issue 003 of radix-dialog-body-lock-fix PRD):
  // when ``open`` transitions from false to true, snap the draft back
  // to whatever ``sourcePrompt`` evaluates to AT THAT MOMENT (so a
  // close-without-save followed by re-open on the same — or a
  // different — variation never carries an abandoned draft into the
  // next session). The ``prevOpenRef`` makes this a true rising-edge
  // detector: re-runs of the effect while ``open`` stays true (e.g.
  // because ``sourcePrompt`` changed identity) do not stomp the
  // user's in-progress draft. ``isSubmitting`` is reset defensively
  // even though the submit handler's ``finally`` already clears it —
  // covers the unlikely race where the dialog reopens while a prior
  // submit is still in flight.
  const prevOpenRef = useRef(false);
  useEffect(() => {
    if (open && !prevOpenRef.current) {
      setDraft(sourcePrompt);
      setIsSubmitting(false);
    }
    prevOpenRef.current = open;
  }, [open, sourcePrompt]);

  const trimmed = draft.trim();
  const canGenerate = trimmed.length > 0 && !isSubmitting && !isBlocked;

  const handleGenerate = async () => {
    if (!canGenerate) return;
    setIsSubmitting(true);
    try {
      await onSubmit(trimmed);
      // Parent closes the dialog on success via onOpenChange(false).
      // The dialog stays mounted (per issue 003 of radix-dialog-body-
      // lock-fix); the next open re-derives sourcePrompt from the
      // freshest props via the rising-edge effect above.
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
