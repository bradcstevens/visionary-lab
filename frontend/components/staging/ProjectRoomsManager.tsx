"use client";

import { useRef, useState } from "react";
import { ImagePlus, Loader2, Pencil, Save, Trash2, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { StorageImage } from "./StorageImage";
import {
  analyzeImages,
  getProject,
  removeRoom,
  updateRoom,
  uploadRooms,
  type StagingProject,
} from "@/services/stagingApi";

/**
 * Project rooms manager — issues 004 + 005 + 006 of the project-
 * settings-completeness PRD.
 *
 * Mounted on the Project Settings sheet between the project-level
 * fields (name, prompt) and the generation settings (variations, model,
 * quality, size). Renders the project's rooms as a vertical list with
 * an inline rename affordance, an inline delete-with-confirm
 * affordance per row, and an "Add photos" affordance below the list
 * that runs the same upload→analyze→refresh sequence the wizard does.
 *
 * Design constraints (per PRD § Implementation Decisions and the
 * issues' "Acceptance criteria" sections):
 *
 *   - Narrow prop interface: `{ project, onProjectUpdate, disabled }`.
 *     The component does NOT import routing primitives, lightbox
 *     components, SSE clients, or any module that exposes generation
 *     state — only the staging API client functions it needs and shared
 *     UI primitives. Keeping the interface this narrow lets the
 *     component be unit-testable in isolation and prevents the rooms
 *     manager from leaking knowledge of any other surface.
 *   - Server-confirmed renames AND deletes (matches the rename-project
 *     pattern the PRD calls out). The component awaits the API call
 *     BEFORE calling `onProjectUpdate(updatedProject)` so local state
 *     and server state never diverge — there is no optimistic-then-
 *     rollback dance.
 *   - Rooms persist immediately per action — there is no Save / Discard
 *     for room edits at the sheet level (matches the PRD's "room
 *     operations persist immediately per action" rule).
 *   - Per-row pending state prevents double-submit: rapid double-click
 *     on Save or "Yes, delete" does not send two effective calls
 *     because the second click sees the pending state and returns
 *     early.
 *   - Empty / whitespace-only label disables Save (mirrors the
 *     backend's 422 rule so the user gets the disabled-button
 *     feedback locally instead of a toast bouncing them back).
 *   - An unchanged trimmed label also disables Save so a
 *     `" Living Room "` edit when the persisted label is
 *     `"Living Room"` is a no-op rather than a wasted round-trip.
 *   - The `disabled` prop is forwarded to the rename input, the
 *     pencil/save/cancel buttons, AND the trash button. Issue 007
 *     will set this from `project.status === 'processing'`; this
 *     slice forwards whatever value the parent passes (the sheet
 *     currently passes its own `isSaving` flag for visual consistency
 *     with the rest of the sheet's controls).
 *
 * Mutual exclusion (issue 005, rubber-duck non-blocking finding):
 * only ONE row at a time can be in any active mode (rename-edit,
 * rename-saving, delete-confirm, deleting). Other rows' pencil and
 * trash are disabled while the active row is in any of those
 * states. Without this, a user could open two delete-confirm
 * prompts at once or click delete on row B mid-rename of row A,
 * losing the rename draft and ending up in a confusing transition.
 *
 * Error UX (issue 005):
 *   - Rename failure: a sonner toast surfaces the error and the row
 *     reverts to view mode (the existing pattern from issue 004).
 *   - Delete failure: an INLINE error message renders inside the
 *     confirm row and the confirm row STAYS VISIBLE so the user can
 *     retry or cancel. This is deliberately NOT a toast that auto-
 *     dismisses (matches the PRD's "the confirm row stays visible
 *     with an inline error and the room row is preserved" rule).
 *     Clicking "Yes, delete" again retries (the per-row pending
 *     guard cannot fire because the prior call already settled).
 */

const DELETE_CONFIRM_PROMPT = "Delete this room and all variations?";

export interface ProjectRoomsManagerProps {
  project: StagingProject;
  onProjectUpdate: (project: StagingProject) => void | Promise<void>;
  disabled: boolean;
}

export function ProjectRoomsManager({
  project,
  onProjectUpdate,
  disabled,
}: ProjectRoomsManagerProps) {
  // Inline edit state lives at the manager level so only one row can
  // be in edit mode at a time. Storing the editing room id (rather
  // than per-row local state) keeps the per-row UI a pure function
  // of `editingRoomId`.
  const [editingRoomId, setEditingRoomId] = useState<string | null>(null);
  const [draftLabel, setDraftLabel] = useState<string>("");
  const [savingRoomId, setSavingRoomId] = useState<string | null>(null);

  // Issue 005 — inline delete-with-confirm state. Like the rename
  // state, only one row at a time can be in delete-confirm mode.
  // `deleteError` is scoped to whichever row is currently in
  // delete-confirm (the rubber-duck non-blocking finding suggested
  // a single scoped error state rather than a Record keyed by room
  // id, since only one confirm row can exist at a time).
  const [deleteConfirmRoomId, setDeleteConfirmRoomId] = useState<string | null>(null);
  const [deletingRoomId, setDeletingRoomId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Issue 006 — add-photos state. Tracks the upload→analyze→refresh
  // pipeline so the button can show a spinner / disable other row
  // affordances while in flight, and so analysis-only retries (after
  // a partial failure) reuse the same in-flight guard. The hidden
  // file <input> is reset on each open so selecting the same file
  // twice in a row still re-triggers `onChange`.
  const [addingPhotos, setAddingPhotos] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Mutual exclusion (issue 005): a row is "elsewhere active" if any
  // OTHER row is currently editing, saving a rename, in delete-confirm,
  // or being deleted. Used to disable a row's pencil + trash buttons
  // when another row is in any of those states. Returns true ONLY
  // when some row is active AND that row is NOT this row.
  const someRowIsActive =
    editingRoomId !== null ||
    savingRoomId !== null ||
    deleteConfirmRoomId !== null ||
    deletingRoomId !== null ||
    addingPhotos;

  const handleEditClick = (roomId: string, currentLabel: string) => {
    setEditingRoomId(roomId);
    setDraftLabel(currentLabel);
  };

  const handleCancel = () => {
    setEditingRoomId(null);
    setDraftLabel("");
  };

  const handleSave = async (roomId: string, originalLabel: string) => {
    if (savingRoomId !== null) {
      // Another save is in flight (per-row pending state — double-
      // submit guard). Drop this click on the floor.
      return;
    }
    const trimmed = draftLabel.trim();
    // Local guards mirror the backend's 422 rule so the user sees
    // disabled-button feedback rather than a network round-trip + toast.
    if (trimmed.length === 0) return;
    if (trimmed === originalLabel) {
      // No-op rename: skip the round-trip and just exit edit mode.
      handleCancel();
      return;
    }

    setSavingRoomId(roomId);
    try {
      const updated = await updateRoom(project.id, roomId, { label: trimmed });
      await onProjectUpdate(updated);
      setEditingRoomId(null);
      setDraftLabel("");
    } catch (err) {
      // Revert by clearing the draft state. The view-mode row reads
      // from `project.rooms`, which is still the pre-rename value
      // (server hasn't been updated, so the parent's project state
      // also hasn't been replaced).
      const message = err instanceof Error ? err.message : "Failed to rename room";
      toast.error(message);
      handleCancel();
    } finally {
      setSavingRoomId(null);
    }
  };

  // Issue 005 — delete handlers.

  const handleDeleteClick = (roomId: string) => {
    setDeleteConfirmRoomId(roomId);
    // Clear any stale error from a prior failed attempt.
    setDeleteError(null);
  };

  const handleCancelDelete = () => {
    setDeleteConfirmRoomId(null);
    setDeleteError(null);
  };

  const handleConfirmDelete = async (roomId: string) => {
    if (deletingRoomId !== null) {
      // Per-row pending guard — drop rapid duplicate clicks while
      // the DELETE is in flight.
      return;
    }
    setDeletingRoomId(roomId);
    // Optimistically clear any error from a prior failed attempt so
    // the retry-after-failure path shows a clean inline state during
    // the second attempt.
    setDeleteError(null);
    try {
      const updated = await removeRoom(project.id, roomId);
      await onProjectUpdate(updated);
      // Success: collapse the confirm row entirely.
      setDeleteConfirmRoomId(null);
    } catch (err) {
      // Failure: keep the confirm row visible with an INLINE error
      // message so the user can retry or cancel. Deliberately NOT a
      // toast that auto-dismisses (matches PRD's "the confirm row
      // stays visible with an inline error and the room row is
      // preserved" rule).
      const message =
        err instanceof Error ? err.message : "Failed to remove room";
      setDeleteError(message);
    } finally {
      setDeletingRoomId(null);
    }
  };

  // Issue 006 — add-photos handlers.
  //
  // The flow is upload → analyze → refresh, executed in order:
  //
  //   1. `uploadRooms` creates the new room records on the server.
  //   2. `analyzeImages` runs the same per-image analysis pipeline
  //      the wizard uses so labels and per-image notes are populated.
  //   3. `getProject` refetches the canonical project state so the
  //      caller's `onProjectUpdate` receives the SAS-resolved
  //      payload (the page-level handler runs `resolveImageUrls`
  //      then `setProject`).
  //
  // Failure handling matches the issue 006 acceptance criteria:
  //   - Step 1 (upload) failure: error toast, no project state
  //     mutation.
  //   - Step 2 (analyze) failure: the upload IS preserved (rooms
  //     exist on the server). We still refetch + propagate so the
  //     new rows render, then surface a non-blocking toast offering
  //     a "Retry analysis" action that re-runs analyzeImages +
  //     refresh. The retry path reuses `addingPhotos` as its
  //     in-flight guard so a click-spam during retry is harmless.
  //   - Step 3 (refresh) failure: surfaced as an error toast. The
  //     server-side state (rooms + analyses) IS persisted, so the
  //     next mount or reload will pick it up.
  //
  // The design brief is intentionally NOT regenerated as part of
  // this flow (the user keeps their edits; the existing Brief tab
  // + Regenerate banner remain the path).

  const refreshProjectAfterAnalysis = async () => {
    const updated = await getProject(project.id);
    await onProjectUpdate(updated);
  };

  const retryAnalysis = async () => {
    if (addingPhotos) return;
    setAddingPhotos(true);
    try {
      await analyzeImages(project.id);
      await refreshProjectAfterAnalysis();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to analyze the new photos";
      toast.error("Couldn't analyze the new photos.", {
        description: message,
        action: { label: "Retry analysis", onClick: () => void retryAnalysis() },
      });
    } finally {
      setAddingPhotos(false);
    }
  };

  const handleAddPhotosClick = () => {
    if (disabled || addingPhotos || someRowIsActive) return;
    fileInputRef.current?.click();
  };

  const handleFilesSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    // Reset the input value so re-selecting the same file fires
    // onChange again. Done eagerly so an early return below still
    // leaves the input in a clean state.
    event.target.value = "";
    if (files.length === 0) return;
    if (addingPhotos) return;

    setAddingPhotos(true);
    try {
      // Step 1: upload. A failure here is hard — no rooms were
      // created server-side, so we surface an error toast and bail
      // without mutating project state.
      try {
        await uploadRooms(
          project.id,
          files.map((file) => ({ file, name: file.name })),
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to upload photos";
        toast.error(message);
        return;
      }

      // Step 2: analyze. A failure here is soft — the rooms exist
      // on the server, so we DON'T abort. We still refetch the
      // project so the new rows render, then offer a retry action.
      let analyzeError: string | null = null;
      try {
        await analyzeImages(project.id);
      } catch (err) {
        analyzeError = err instanceof Error ? err.message : "Failed to analyze the new photos";
      }

      // Step 3: refresh. Always run so the new rows propagate
      // regardless of whether analysis succeeded.
      try {
        await refreshProjectAfterAnalysis();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to refresh project";
        toast.error(message);
        return;
      }

      if (analyzeError) {
        toast.error("Couldn't analyze the new photos.", {
          description: analyzeError,
          action: { label: "Retry analysis", onClick: () => void retryAnalysis() },
        });
      }
    } finally {
      setAddingPhotos(false);
    }
  };

  const addPhotosDisabled = disabled || addingPhotos || someRowIsActive;
  const addPhotosFooter = (
    <div className="flex items-center gap-2 pt-1">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        data-testid="project-rooms-manager-add-photos-input"
        onChange={handleFilesSelected}
      />
      <Button
        size="sm"
        variant="outline"
        onClick={handleAddPhotosClick}
        disabled={addPhotosDisabled}
        data-testid="project-rooms-manager-add-photos"
      >
        {addingPhotos ? (
          <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
        ) : (
          <ImagePlus className="mr-1.5 h-3.5 w-3.5" />
        )}
        Add photos
      </Button>
    </div>
  );

  if (project.rooms.length === 0) {
    return (
      <div className="space-y-2" data-testid="project-rooms-manager">
        <h3 className="text-sm font-medium">Rooms</h3>
        <p
          className="text-xs text-muted-foreground"
          data-testid="project-rooms-manager-empty"
        >
          No rooms yet.
        </p>
        {addPhotosFooter}
      </div>
    );
  }

  return (
    <div className="space-y-2" data-testid="project-rooms-manager">
      <h3 className="text-sm font-medium">Rooms</h3>
      <ul className="space-y-2">
        {project.rooms.map((room) => {
          const isEditing = editingRoomId === room.id;
          const isSaving = savingRoomId === room.id;
          const isConfirmingDelete = deleteConfirmRoomId === room.id;
          const isDeleting = deletingRoomId === room.id;
          const isThisRowActive =
            isEditing || isSaving || isConfirmingDelete || isDeleting;
          // The OTHER rows' pencil + trash should be disabled while
          // any row is active. For THIS row, we use the in-mode
          // rendering branch instead of disabling the buttons.
          const disabledByOtherRow = someRowIsActive && !isThisRowActive;

          const trimmedDraft = draftLabel.trim();
          const saveDisabled =
            disabled ||
            isSaving ||
            trimmedDraft.length === 0 ||
            trimmedDraft === room.label;

          return (
            <li
              key={room.id}
              data-testid={`project-rooms-manager-row-${room.id}`}
              className="flex flex-col gap-2 rounded-md border p-2"
            >
              <div className="flex items-center gap-3">
                {/* Thumbnail. Falls back to a small placeholder via
                    StorageImage's built-in fallback if the URL fails. */}
                <div className="h-10 w-10 shrink-0 overflow-hidden rounded">
                  <StorageImage
                    src={room.original_thumbnail_url ?? room.original_image_url}
                    alt={room.label}
                    className="h-10 w-10 object-cover"
                    fallbackClassName="h-10 w-10 bg-muted text-[10px]"
                    fallbackText={room.label.slice(0, 2).toUpperCase()}
                  />
                </div>

                {isEditing ? (
                  <>
                    <Input
                      autoFocus
                      value={draftLabel}
                      onChange={(e) => setDraftLabel(e.target.value)}
                      disabled={disabled || isSaving}
                      data-testid={`project-rooms-manager-input-${room.id}`}
                      className="flex-1"
                    />
                    <Button
                      size="sm"
                      onClick={() => handleSave(room.id, room.label)}
                      disabled={saveDisabled}
                      data-testid={`project-rooms-manager-save-${room.id}`}
                      aria-label={`Save rename of ${room.label}`}
                    >
                      {isSaving ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Save className="h-3.5 w-3.5" />
                      )}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={handleCancel}
                      disabled={disabled || isSaving}
                      data-testid={`project-rooms-manager-cancel-${room.id}`}
                      aria-label={`Cancel rename of ${room.label}`}
                    >
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  </>
                ) : isConfirmingDelete ? (
                  // Issue 005 — inline delete-with-confirm. The
                  // prompt + buttons live inside the row's own
                  // container — NOT a portal-mounted modal — to
                  // satisfy the PRD's "no modal popover from a
                  // different mount point — keep the deep-module
                  // rule" requirement.
                  <div
                    className="flex flex-1 items-center gap-2"
                    data-testid={`project-rooms-manager-confirm-${room.id}`}
                    role="alertdialog"
                    aria-label={`Confirm deletion of ${room.label}`}
                  >
                    <span className="flex-1 text-sm">{DELETE_CONFIRM_PROMPT}</span>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => handleConfirmDelete(room.id)}
                      disabled={disabled || isDeleting}
                      data-testid={`project-rooms-manager-confirm-yes-${room.id}`}
                    >
                      {isDeleting ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        "Yes, delete"
                      )}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={handleCancelDelete}
                      disabled={disabled || isDeleting}
                      data-testid={`project-rooms-manager-confirm-cancel-${room.id}`}
                    >
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <>
                    <span
                      className="flex-1 text-sm"
                      data-testid={`project-rooms-manager-label-${room.id}`}
                    >
                      {room.label}
                    </span>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-7 w-7"
                      onClick={() => handleEditClick(room.id, room.label)}
                      disabled={disabled || disabledByOtherRow}
                      data-testid={`project-rooms-manager-edit-${room.id}`}
                      aria-label={`Rename ${room.label}`}
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-7 w-7"
                      onClick={() => handleDeleteClick(room.id)}
                      disabled={disabled || disabledByOtherRow}
                      data-testid={`project-rooms-manager-delete-${room.id}`}
                      aria-label={`Delete ${room.label}`}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </>
                )}
              </div>

              {/* Inline error for the delete-confirm row. Only renders
                  for the row that's currently in delete-confirm AND
                  has an error from a prior attempt. The confirm row
                  itself stays visible above this error so the user
                  can retry or cancel. */}
              {isConfirmingDelete && deleteError && (
                <p
                  className="text-xs text-destructive"
                  data-testid={`project-rooms-manager-confirm-error-${room.id}`}
                  role="alert"
                >
                  {deleteError}
                </p>
              )}
            </li>
          );
        })}
      </ul>
      {addPhotosFooter}
    </div>
  );
}
