"use client"

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, Plus, RefreshCw, Loader2, Play, AlertTriangle, Trash2, MoreHorizontal, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { RoomGroup } from "@/components/staging/RoomGroup";
import { ProgressTracker } from "@/components/staging/ProgressTracker";
import { getProject, deleteProject, streamGeneration, streamRoomRegeneration, StagingProject, Room, StagingStreamEvent } from "@/services/stagingApi";
import { sasTokenService } from "@/services/sas-token";
import { toast } from "sonner";
import { parseApiError } from "@/utils/error-utils";
import { useActivityLog } from "@/context/activity-log-context";

export default function ProjectDetailPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  const [project, setProject] = useState<StagingProject | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const streamCleanupRef = useRef<(() => void) | null>(null);
  const reloadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestLoadIdRef = useRef(0);

  const activityLog = useActivityLog();

  useEffect(() => {
    activityLog.clear();
    return () => activityLog.clear();
  }, []);

  // Abort any active stream and pending reloads on unmount
  useEffect(() => {
    return () => {
      streamCleanupRef.current?.();
      if (reloadTimerRef.current) clearTimeout(reloadTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (projectId) {
      loadProject();
    }
  }, [projectId]);

  const resolveImageUrls = async (data: StagingProject) => {
    try {
      const tokens = await sasTokenService.getTokens();
      for (const room of data.rooms) {
        if (room.original_image_url && !room.original_image_url.includes('?')) {
          room.original_image_url = `${room.original_image_url}?${tokens.imageSasToken}`;
        }
        for (const variation of room.variations) {
          if (variation.image_url && !variation.image_url.includes('?')) {
            variation.image_url = `${variation.image_url}?${tokens.imageSasToken}`;
          }
        }
      }
    } catch (sasError) {
      console.warn('Failed to get SAS tokens, images may not load:', sasError);
    }
  };

  const loadProject = useCallback(async () => {
    const loadId = ++latestLoadIdRef.current;
    try {
      setIsLoading(true);
      const data = await getProject(projectId);
      // Discard stale responses from overlapping fetches
      if (loadId !== latestLoadIdRef.current) return;
      await resolveImageUrls(data);
      if (loadId !== latestLoadIdRef.current) return;
      setProject(data);
    } catch (error) {
      if (loadId !== latestLoadIdRef.current) return;
      console.error('Failed to load project:', error);
      toast.error('Failed to load project');
      router.push('/projects');
    } finally {
      if (loadId === latestLoadIdRef.current) {
        setIsLoading(false);
      }
    }
  }, [projectId]);

  /** Debounced reload — coalesces rapid SSE events into a single fetch. */
  const debouncedReload = useCallback(() => {
    if (reloadTimerRef.current) clearTimeout(reloadTimerRef.current);
    reloadTimerRef.current = setTimeout(() => {
      reloadTimerRef.current = null;
      loadProject();
    }, 500);
  }, [loadProject]);

  const handleStreamEvent = useCallback((event: StagingStreamEvent) => {
    switch (event.type) {
      case 'room_started':
        activityLog.log({
          level: 'info',
          icon: '▶',
          message: `Starting generation for "${(event as any).label ?? 'room'}"`,
          detail: `Room ${(event as any).room_id?.slice(0, 8) ?? ''}`,
        });
        debouncedReload();
        break;
      case 'variation_completed':
        activityLog.log({
          level: 'success',
          icon: '✓',
          message: `Variation ${((event as any).variation_index ?? 0) + 1} saved`,
          detail: [
            (event as any).model,
            (event as any).tokens_used ? `${Number((event as any).tokens_used).toLocaleString()} tokens` : null,
            (event as any).elapsed_ms ? `${((event as any).elapsed_ms / 1000).toFixed(1)}s` : null,
          ].filter(Boolean).join(' · ') || undefined,
        });
        debouncedReload();
        break;
      case 'variation_failed':
        activityLog.log({
          level: 'error',
          icon: '✕',
          message: `Variation ${((event as any).variation_index ?? 0) + 1} failed`,
          detail: (event as any).error || 'Unknown error',
        });
        debouncedReload();
        break;
      case 'room_uploaded':
        debouncedReload();
        break;
      case 'room_completed':
        activityLog.log({
          level: 'success',
          icon: '✓',
          message: 'Room complete',
          detail: `Room ${(event as any).room_id?.slice(0, 8) ?? ''}`,
        });
        debouncedReload();
        break;
      case 'room_failed':
        activityLog.log({
          level: 'error',
          icon: '✕',
          message: 'Room failed',
          detail: (event as any).error || 'Unknown error',
        });
        debouncedReload();
        break;
      case 'project_completed':
        // Cancel any pending debounced reload
        if (reloadTimerRef.current) clearTimeout(reloadTimerRef.current);
        activityLog.log({
          level: 'success',
          icon: '🎉',
          message: 'Generation complete!',
        });
        setIsGenerating(false);
        setGenerationError(null);
        toast.success('Generation completed!');
        loadProject();
        break;
      case 'stream_ended':
        // Fallback: stream closed without a terminal event — reconcile state
        if (reloadTimerRef.current) clearTimeout(reloadTimerRef.current);
        setIsGenerating(false);
        loadProject();
        break;
      case 'error':
        if (reloadTimerRef.current) clearTimeout(reloadTimerRef.current);
        activityLog.log({
          level: 'error',
          icon: '✕',
          message: 'Generation error',
          detail: event.error || 'Unknown error',
        });
        setIsGenerating(false);
        setGenerationError(event.error || 'Generation failed');
        toast.error(event.error || 'Generation failed');
        loadProject();
        break;
    }
  }, [activityLog, debouncedReload, loadProject]);

  const startGeneration = useCallback(() => {
    if (isGenerating) return;
    // Abort any existing stream before starting a new one
    streamCleanupRef.current?.();
    setIsGenerating(true);
    setGenerationError(null);
    activityLog.log({
      level: 'info',
      icon: '▶',
      message: `Starting generation for "${project?.name}"`,
      detail: `${totalVariations} variations queued across ${project?.rooms.length} images`,
    });
    streamCleanupRef.current = streamGeneration(projectId, handleStreamEvent);
  }, [activityLog, project, totalVariations, isGenerating, projectId, handleStreamEvent]);

  const handleRegenerateRoom = useCallback((room: Room) => {
    if (isGenerating) return;
    // Abort any existing stream before starting a new one
    streamCleanupRef.current?.();
    setIsGenerating(true);
    setGenerationError(null);
    toast.info(`Regenerating ${room.label}...`);
    streamCleanupRef.current = streamRoomRegeneration(projectId, room.id, handleStreamEvent);
  }, [isGenerating, projectId, handleStreamEvent]);

  const handleRegenerateAll = () => {
    if (isGenerating) return;
    startGeneration();
    toast.info('Regenerating all rooms...');
  };

  const handleVariationClick = (room: Room, variationIndex: number) => {
    const variation = room.variations[variationIndex];
    if (variation.status === 'completed' && variation.image_url) {
      window.open(variation.image_url, '_blank');
    }
  };

  const handleRetryVariation = (room: Room, _variationIndex: number) => {
    handleRegenerateRoom(room);
  };

  const handleAddRooms = () => {
    toast.info('Add rooms feature coming soon');
  };

  const handleDeleteProject = async () => {
    setIsDeleting(true);
    try {
      await deleteProject(projectId);
      toast.success('Project and all artifacts deleted');
      router.push('/projects');
    } catch (error) {
      console.error('Failed to delete project:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to delete project');
      setIsDeleting(false);
      setShowDeleteConfirm(false);
    }
  };

  // Computed state
  const allPending = project?.rooms.every(r => r.status === 'pending') ?? false;
  const hasFailed = project?.rooms.some(r => r.status === 'failed' || r.variations.some(v => v.status === 'failed')) ?? false;
  const totalVariations = project?.rooms.reduce((sum, r) => sum + r.variations.length, 0) ?? 0;
  const completedVariations = project?.rooms.reduce((sum, r) => sum + r.variations.filter(v => v.status === 'completed').length, 0) ?? 0;

  if (isLoading || !project) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-center h-64">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
            Loading project...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" asChild>
            <Link href="/projects">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Projects
            </Link>
          </Button>
        </div>

        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-bold">{project.name}</h1>
              <Badge variant={project.status === 'completed' ? 'default' : project.status === 'failed' ? 'destructive' : 'outline'} className="text-xs">
                {project.status}
              </Badge>
            </div>
            <p className="text-muted-foreground leading-relaxed max-w-3xl">
              {project.prompt}
            </p>
            {totalVariations > 0 && (
              <p className="text-xs text-muted-foreground">
                {completedVariations}/{totalVariations} variations complete across {project.rooms.length} images
              </p>
            )}
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {/* Primary action — most common action gets the prominent button */}
            {allPending && project.rooms.length > 0 ? (
              <Button onClick={startGeneration} disabled={isGenerating}>
                {isGenerating ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
                Generate
              </Button>
            ) : !allPending ? (
              <Button variant="outline" onClick={handleRegenerateAll} disabled={isGenerating}>
                {isGenerating ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <RefreshCw className="h-4 w-4 mr-2" />}
                Regenerate All
              </Button>
            ) : null}

            {/* Overflow menu — secondary and destructive actions */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon" disabled={isGenerating || isDeleting}>
                  <MoreHorizontal className="h-4 w-4" />
                  <span className="sr-only">More actions</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={handleAddRooms} disabled={isGenerating}>
                  <Plus className="h-4 w-4 mr-2" />
                  Add more images
                </DropdownMenuItem>
                {!allPending && (
                  <DropdownMenuItem onClick={handleRegenerateAll} disabled={isGenerating}>
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Regenerate all
                  </DropdownMenuItem>
                )}
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="text-destructive focus:text-destructive"
                  onClick={() => setShowDeleteConfirm(true)}
                  disabled={isGenerating || isDeleting}
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete project
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>

      {/* Generation error banner */}
      {generationError && (() => {
        const parsed = parseApiError(generationError);
        return (
          <div className="overflow-hidden rounded-lg border border-destructive/20 bg-destructive/[0.04]">
            <div className="flex items-start gap-3 p-4">
              <AlertTriangle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0 space-y-1">
                <p className="text-sm font-medium text-destructive">
                  Generation encountered an error
                  {parsed.statusCode ? ` (${parsed.statusCode})` : ""}
                </p>
                {parsed.detail && (
                  <Collapsible>
                    <p className="text-xs text-destructive/80 line-clamp-2 break-words">
                      {parsed.detail}
                    </p>
                    {(parsed.isTruncated || (parsed.detail?.length ?? 0) > 120) && (
                      <CollapsibleTrigger className="group inline-flex items-center gap-1 text-[11px] text-destructive/60 hover:text-destructive transition-colors mt-1 cursor-pointer">
                        <ChevronDown className="h-3 w-3 transition-transform group-data-[state=open]:rotate-180" />
                        Full error
                      </CollapsibleTrigger>
                    )}
                    <CollapsibleContent>
                      <pre className="mt-2 rounded-md bg-destructive/[0.06] border border-destructive/10 px-3 py-2 text-[11px] text-destructive/70 font-mono whitespace-pre-wrap break-all max-h-32 overflow-y-auto">
                        {parsed.detail}{parsed.isTruncated && "…"}
                      </pre>
                    </CollapsibleContent>
                  </Collapsible>
                )}
              </div>
              <Button size="sm" variant="outline" onClick={handleRegenerateAll} className="shrink-0">
                <RefreshCw className="h-3.5 w-3.5 mr-1" />
                Retry
              </Button>
            </div>
          </div>
        );
      })()}

      {/* Call-to-action for pending projects */}
      {allPending && project.rooms.length > 0 && !isGenerating && (
        <div className="flex flex-col items-center gap-4 py-8 px-6 bg-muted/30 border border-dashed border-muted-foreground/25 rounded-xl">
          <div className="text-center space-y-2">
            <h3 className="text-lg font-semibold">Ready to generate</h3>
            <p className="text-sm text-muted-foreground max-w-md">
              {project.rooms.length} image{project.rooms.length !== 1 ? 's' : ''} uploaded with {totalVariations} variations queued. 
              Click generate to start the AI image generation pipeline.
            </p>
          </div>
          <Button size="lg" onClick={startGeneration}>
            <Play className="h-4 w-4 mr-2" />
            Generate {totalVariations} Variations
          </Button>
        </div>
      )}

      {/* Generating progress */}
      {isGenerating && (
        <div className="flex items-center gap-3 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
          <Loader2 className="h-5 w-5 animate-spin text-blue-500 flex-shrink-0" />
          <div>
            <p className="text-sm font-medium">Generating variations...</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              {completedVariations}/{totalVariations} complete — this may take a few minutes per image
            </p>
          </div>
        </div>
      )}

      {/* Progress Tracker */}
      <ProgressTracker project={project} />

      {/* Room Groups */}
      <div className="space-y-12">
        {project.rooms.length === 0 ? (
          <div className="text-center py-12">
            <div className="space-y-3">
              <h3 className="text-xl font-semibold">No rooms uploaded</h3>
              <p className="text-muted-foreground">
                Add room images to start generating staged variations
              </p>
              <Button onClick={handleAddRooms}>
                <Plus className="h-4 w-4 mr-2" />
                Add Rooms
              </Button>
            </div>
          </div>
        ) : (
          project.rooms.map((room) => (
            <RoomGroup
              key={room.id}
              room={room}
              onVariationClick={handleVariationClick}
              onRetryVariation={handleRetryVariation}
              onRegenerateRoom={handleRegenerateRoom}
              isGenerating={isGenerating}
            />
          ))
        )}
      </div>

      {/* Delete confirmation dialog */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-background border rounded-xl shadow-lg p-6 max-w-md mx-4 space-y-4">
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">Delete project?</h3>
              <p className="text-sm text-muted-foreground">
                This will permanently delete <strong>{project.name}</strong>, including all {project.rooms.length} uploaded images and {totalVariations} generated variations from Azure storage. This cannot be undone.
              </p>
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowDeleteConfirm(false)} disabled={isDeleting}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleDeleteProject} disabled={isDeleting}>
                {isDeleting ? (
                  <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Deleting...</>
                ) : (
                  <><Trash2 className="h-4 w-4 mr-2" />Delete Project</>
                )}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}