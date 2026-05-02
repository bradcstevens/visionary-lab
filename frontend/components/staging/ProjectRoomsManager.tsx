"use client";

import { useState } from "react";
import { Loader2, Pencil, Save, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { StorageImage } from "./StorageImage";
import { updateRoom, type StagingProject } from "@/services/stagingApi";

/**
 * Project rooms manager — issue 004 of the project-settings-completeness
 * PRD.
 *
 * Mounted on the Project Settings sheet between the project-level
 * fields (name, prompt) and the generation settings (variations, model,
 * quality, size). Renders the project's rooms as a vertical list with
 * an inline rename affordance per row. Future slices add inline delete
 * (issue 005) and add-photos (issue 006); this slice ships rename only.
 *
 * Design constraints (per PRD § Implementation Decisions and the
 * issue's "Acceptance criteria" section):
 *
 *   - Narrow prop interface: `{ project, onProjectUpdate, disabled }`.
 *     The component does NOT import routing primitives, lightbox
 *     components, SSE clients, or any module that exposes generation
 *     state — only the staging API client functions it needs and shared
 *     UI primitives. Keeping the interface this narrow lets future
 *     slices unit-test the component in isolation and prevents the
 *     rooms manager from leaking knowledge of any other surface.
 *   - Server-confirmed renames (matches the rename-project pattern the
 *     PRD calls out). The component awaits `updateRoom(...)` BEFORE
 *     calling `onProjectUpdate(updatedProject)` so local state and
 *     server state never diverge — there is no optimistic-then-rollback
 *     dance.
 *   - Rooms persist immediately per action — there is no Save / Discard
 *     for room edits at the sheet level (matches the PRD's "room
 *     operations persist immediately per action" rule).
 *   - Per-row pending state prevents double-submit (rubber-duck
 *     finding from issue 004 design review): rapid double-click on
 *     Save does not send two effective renames because the second
 *     click sees `savingRoomId === room.id` and returns early.
 *   - Empty / whitespace-only label disables Save (mirrors the
 *     backend's 422 rule so the user gets the disabled-button
 *     feedback locally instead of a toast bouncing them back).
 *   - An unchanged trimmed label also disables Save (rubber-duck
 *     finding) so a `" Living Room "` edit when the persisted label
 *     is `"Living Room"` is a no-op rather than a wasted round-trip.
 *   - The `disabled` prop is forwarded to the rename input and the
 *     pencil/save/cancel buttons. Issue 007 will set this from
 *     `project.status === 'processing'`; this slice forwards whatever
 *     value the parent passes (the sheet currently passes its own
 *     `isSaving` flag for visual consistency with the rest of the
 *     sheet's controls).
 *
 * Error UX: on `updateRoom` failure the row reverts to view mode with
 * the original label and a toast surfaces the error. Matches the
 * existing project-rename failure pattern.
 */

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
      // Another save is in flight (per-row pending state — rubber-duck
      // double-submit guard). Drop this click on the floor.
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
              className="flex items-center gap-3 rounded-md border p-2"
            >
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
                    disabled={disabled || savingRoomId !== null}
                    data-testid={`project-rooms-manager-edit-${room.id}`}
                    aria-label={`Rename ${room.label}`}
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </Button>
                </>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
