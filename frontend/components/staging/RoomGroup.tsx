"use client"

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";
import { RefreshCw, Clock, Info, AlertTriangle, Pencil, Loader2 } from "lucide-react";
import { VariationThumbnail } from "./VariationThumbnail";
import { StorageImage } from "./StorageImage";
import { Room } from "@/services/stagingApi";

interface RoomGroupProps {
  room: Room;
  onVariationClick?: (room: Room, variationIndex: number) => void;
  onRetryVariation?: (room: Room, variationIndex: number) => void;
  onRegenerateRoom?: (room: Room) => void;
  onRegenerateVariation?: (room: Room, variationIndex: number, strategy: 'retry' | 'fresh') => void;
  // Issue 004 of projects-page-improvements PRD: opens the Edit Prompt
  // dialog for the chosen variation (parent owns the dialog state).
  onEditPromptVariation?: (room: Room, variationIndex: number) => void;
  onUpdateAddendum?: (room: Room, promptAddendum: string | null) => Promise<void>;
  regeneratingVariationId?: string | null;
  /**
   * Issue 007 of projects-page-improvements PRD: per-room busy gate.
   * True iff a project-level OR this-room-level generation is in flight.
   * Replaces the prior global ``isGenerating`` prop. Disables the
   * room-level Regenerate button. NOT set true by an in-flight variation
   * regen in this room — that case is handled by the per-variation gate
   * below (preserves the retry-queue scenario 3 supersede semantic where
   * Regenerate Room can be clicked while a variation in the room is
   * mid-regen).
   */
  isRoomBusy?: boolean;
  /**
   * Per-variation in-flight set. The menu items on a variation are gated
   * on `!isRoomBusy && !inFlightVariationIds.has(variation.id)` so a
   * variation already mid-regen / mid-edit-prompt does not re-fire.
   */
  inFlightVariationIds?: ReadonlySet<string>;
  queuedVariationIds?: ReadonlySet<string>;
  /**
   * Issue 009 of image-pipeline-and-project-ux-overhaul PRD: per-variation
   * live job map (variation_id → most-recent ProjectJob), sourced from
   * useProjectJobs. Drives the per-image progress overlay rendered inside
   * VariationThumbnail. Optional so callers that haven't been updated keep
   * rendering without overlays.
   */
  jobsByVariationId?: ReadonlyMap<string, import("@/context/jobs-context").ProjectJob>;
  /**
   * Issue 005 of projects-page-improvements PRD: positional context.
   * 1-based index of this room within `project.rooms`. Combined with
   * `totalRooms` to render a small "Image N of M" label next to the
   * room title. Both props are optional so callers that haven't been
   * updated still render correctly (label is suppressed when either
   * is missing).
   */
  roomIndex?: number;
  totalRooms?: number;
}

export function RoomGroup({ room, onVariationClick, onRetryVariation, onRegenerateRoom, onRegenerateVariation, onEditPromptVariation, onUpdateAddendum, regeneratingVariationId, isRoomBusy, inFlightVariationIds, queuedVariationIds, jobsByVariationId, roomIndex, totalRooms }: RoomGroupProps) {
  // Pencil-icon popover state for the per-room prompt addendum (issue 003 of
  // the projects-page-improvements PRD). The draft is reset to the persisted
  // value every time the popover opens so a Cancel followed by another open
  // shows the saved state, not the stale draft.
  const [isAddendumOpen, setIsAddendumOpen] = useState(false);
  const [addendumDraft, setAddendumDraft] = useState<string>(room.prompt_addendum ?? "");
  const [isSavingAddendum, setIsSavingAddendum] = useState(false);

  const handleAddendumOpenChange = (open: boolean) => {
    if (open) {
      // Resync the draft from the persisted value on every open so a
      // post-save reload (which changes ``room.prompt_addendum``) is
      // reflected before the user types again.
      setAddendumDraft(room.prompt_addendum ?? "");
    }
    setIsAddendumOpen(open);
  };

  const handleSaveAddendum = async () => {
    if (!onUpdateAddendum) return;
    setIsSavingAddendum(true);
    try {
      // Empty / whitespace-only sends `null` so the backend clears the
      // field. The server normalizes both to `null` on persist; sending
      // explicit `null` keeps the request body unambiguous.
      const trimmed = addendumDraft.trim();
      const next: string | null = trimmed.length === 0 ? null : trimmed;
      await onUpdateAddendum(room, next);
      setIsAddendumOpen(false);
    } finally {
      setIsSavingAddendum(false);
    }
  };

  const getStatusVariant = (status: string): "default" | "secondary" | "destructive" | "outline" => {
    switch (status) {
      case 'completed':
        return 'default';
      case 'processing':
        return 'secondary';
      case 'failed':
        return 'destructive';
      case 'pending':
      default:
        return 'outline';
    }
  };

  const completedCount = room.variations.filter(v => v.status === 'completed').length;
  const failedCount = room.variations.filter(v => v.status === 'failed').length;
  const totalCount = room.variations.length;

  const getStatusMessage = () => {
    switch (room.status) {
      case 'pending':
        return 'Waiting for generation to start — click "Generate" to begin';
      case 'processing':
        return `Generating variations... ${completedCount}/${totalCount} done`;
      case 'completed':
        if (failedCount > 0) {
          return `${completedCount}/${totalCount} variations generated — ${failedCount} failed`;
        }
        return `All ${totalCount} variations generated`;
      case 'failed':
        return `${failedCount} variation${failedCount !== 1 ? 's' : ''} failed — click retry to regenerate`;
      default:
        return '';
    }
  };

  return (
    // Issue 005 of projects-page-improvements PRD: each room is wrapped
    // in the existing Card primitive so the per-room Generate /
    // Regenerate button is visibly enclosed in the same container as
    // the images it acts on. The Card primitive defaults to
    // `flex flex-col gap-6 rounded-xl border py-6 shadow-sm`; we
    // override `py-6 → p-4` for a tighter padding and `gap-6 → gap-3`
    // to achieve the issue spec's `space-y-3` intent (writing literal
    // `space-y-3` would stack 12px margin-top on top of the existing
    // 24px flex gap because tailwind-merge does not collapse
    // `space-y-*` against `gap-*`). Per the PRD this should be a
    // subtle boundary, not heavy chrome.
    <Card data-testid={`room-card-${room.id}`} className="p-4 gap-3">
      {/* Room Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-semibold">{room.label}</h3>
          {/* Positional context: small "Image N of M" label reflecting
              this room's position in the project's room list. Rendered
              before the addendum pencil so the title row reads
              "title — positional context — controls — status". */}
          {roomIndex !== undefined && totalRooms !== undefined && totalRooms > 0 && (
            <span
              className="text-xs text-muted-foreground"
              data-testid={`room-position-${room.id}`}
            >
              Image {roomIndex} of {totalRooms}
            </span>
          )}
          {onUpdateAddendum && (
            <Popover open={isAddendumOpen} onOpenChange={handleAddendumOpenChange}>
              <PopoverTrigger asChild>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-7 w-7"
                  aria-label={
                    room.prompt_addendum
                      ? `Edit per-image addendum for ${room.label}`
                      : `Add per-image addendum for ${room.label}`
                  }
                  data-testid={`room-addendum-trigger-${room.id}`}
                  data-has-addendum={room.prompt_addendum ? "true" : "false"}
                >
                  <Pencil className="h-3.5 w-3.5" />
                </Button>
              </PopoverTrigger>
              <PopoverContent
                align="start"
                className="w-80 space-y-3"
                data-testid={`room-addendum-popover-${room.id}`}
              >
                <div className="space-y-1">
                  <h4 className="text-sm font-medium">Per-image addendum</h4>
                  <p className="text-xs text-muted-foreground">
                    Appended to this room&apos;s prompt for future generations.
                    Existing variations are not changed.
                  </p>
                </div>
                <Textarea
                  value={addendumDraft}
                  onChange={(e) => setAddendumDraft(e.target.value)}
                  placeholder="e.g. always in front of the fence, never behind"
                  rows={4}
                  disabled={isSavingAddendum}
                  data-testid={`room-addendum-textarea-${room.id}`}
                />
                <div className="flex justify-end gap-2">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setIsAddendumOpen(false)}
                    disabled={isSavingAddendum}
                  >
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    onClick={handleSaveAddendum}
                    disabled={isSavingAddendum}
                    data-testid={`room-addendum-save-${room.id}`}
                  >
                    {isSavingAddendum && <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />}
                    Save
                  </Button>
                </div>
              </PopoverContent>
            </Popover>
          )}
          <Badge variant={getStatusVariant(room.status)} className="text-xs">
            {room.status}
          </Badge>
          {totalCount > 0 && (
            <span className="text-xs text-muted-foreground">
              {completedCount}/{totalCount} variations
            </span>
          )}
        </div>
        {onRegenerateRoom && (room.status === 'failed' || room.status === 'completed' || room.status === 'processing') && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => onRegenerateRoom(room)}
            disabled={isRoomBusy}
          >
            <RefreshCw className="h-3.5 w-3.5 mr-1" />
            Regenerate
          </Button>
        )}
      </div>

      {/* Status insight message */}
      {(room.status !== 'completed' || failedCount > 0) && (
        <div className={`flex items-center gap-2 text-xs px-3 py-2 rounded-md ${
          room.status === 'pending' ? 'bg-muted/50 text-muted-foreground' :
          room.status === 'processing' ? 'bg-blue-500/10 text-blue-600 dark:text-blue-400' :
          room.status === 'failed' ? 'bg-destructive/10 text-destructive' :
          failedCount > 0 ? 'bg-amber-500/10 text-amber-600 dark:text-amber-400' : ''
        }`}>
          {room.status === 'pending' && <Clock className="h-3.5 w-3.5" />}
          {room.status === 'processing' && <Info className="h-3.5 w-3.5" />}
          {room.status === 'failed' && <Info className="h-3.5 w-3.5" />}
          {room.status === 'completed' && failedCount > 0 && <AlertTriangle className="h-3.5 w-3.5" />}
          {getStatusMessage()}
        </div>
      )}

      {/* Room Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
        {/* Original Image */}
        <div className="relative">
          <div className="aspect-square w-full min-h-[120px] relative">
            <StorageImage
              src={room.original_image_url}
              alt={`${room.label} original`}
              className="w-full h-full object-cover rounded-lg border-2 border-amber-400"
              fallbackClassName="w-full h-full rounded-lg border-2 border-amber-400"
              fallbackText="Image unavailable — check storage access"
              overlay={
                <Badge 
                  variant="secondary" 
                  className="absolute top-2 right-2 bg-amber-400 text-amber-900 text-xs font-medium"
                >
                  ORIGINAL
                </Badge>
              }
            />
          </div>
        </div>

        {/* Variation Thumbnails */}
        {room.variations.map((variation, index) => {
          // Issue 007: per-variation gating — a variation's regen / edit-
          // prompt menu is hidden when EITHER the room is busy (project
          // op or this room's own regen) OR this specific variation is
          // already mid-regen / mid-edit-prompt. The two conditions catch
          // distinct cases: room-busy hides the menu on every variation;
          // per-variation hides only the one in flight (matching the
          // pre-issue-007 behavior where the spinner appeared exactly on
          // the in-flight variation while siblings stayed clickable
          // outside of room-level generation).
          const variationInFlight = inFlightVariationIds?.has(variation.id) ?? false;
          const variationActionsAllowed = !isRoomBusy && !variationInFlight;
          return (
            <VariationThumbnail
              key={variation.id}
              imageUrl={variation.image_url}
              thumbUrl={variation.thumb_url}
              status={variation.status}
              error={variation.error}
              index={index}
              onClick={
                variation.status === 'completed' && onVariationClick
                  ? () => onVariationClick(room, index)
                  : undefined
              }
              onRetry={
                variation.status === 'failed' && onRetryVariation
                  ? () => onRetryVariation(room, index)
                  : undefined
              }
              onRegenerate={
                variation.status === 'completed' && onRegenerateVariation && variationActionsAllowed
                  ? (strategy) => onRegenerateVariation(room, index, strategy)
                  : undefined
              }
              onEditPrompt={
                variation.status === 'completed' && onEditPromptVariation && variationActionsAllowed
                  ? () => onEditPromptVariation(room, index)
                  : undefined
              }
              isRegenerating={regeneratingVariationId === variation.id}
              isQueued={queuedVariationIds?.has(variation.id) ?? false}
              job={jobsByVariationId?.get(variation.id) ?? null}
            />
          );
        })}
      </div>
    </Card>
  );
}