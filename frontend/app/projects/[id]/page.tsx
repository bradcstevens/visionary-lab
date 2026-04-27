"use client"

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, Plus, RefreshCw, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { RoomGroup } from "@/components/staging/RoomGroup";
import { ProgressTracker } from "@/components/staging/ProgressTracker";
import { getProject, streamGeneration, streamRoomRegeneration, StagingProject, Room, StagingStreamEvent } from "@/services/stagingApi";
import { sasTokenService } from "@/services/sas-token";
import { toast } from "sonner";

export default function ProjectDetailPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  const [project, setProject] = useState<StagingProject | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    if (projectId) {
      loadProject();
    }
  }, [projectId]);

  const loadProject = async () => {
    try {
      setIsLoading(true);
      const data = await getProject(projectId);

      // Resolve blob URLs with SAS tokens so <img> tags can load them
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

      setProject(data);

      // Start streaming if project is processing
      if (data.status === 'processing') {
        startStreaming();
      }
    } catch (error) {
      console.error('Failed to load project:', error);
      toast.error('Failed to load project');
      router.push('/projects');
    } finally {
      setIsLoading(false);
    }
  };

  const startStreaming = () => {
    if (isStreaming) return;

    setIsStreaming(true);
    const cleanup = streamGeneration(projectId, handleStreamEvent);

    // Cleanup function will be called when component unmounts
    return cleanup;
  };

  const handleStreamEvent = (event: StagingStreamEvent) => {
    console.log('Stream event:', event);

    switch (event.type) {
      case 'variation_completed':
      case 'variation_failed':
      case 'room_uploaded':
        // Refresh project data to get latest state
        loadProject();
        break;

      case 'project_completed':
        setIsStreaming(false);
        toast.success('Project generation completed!');
        loadProject();
        break;

      case 'error':
        setIsStreaming(false);
        toast.error(event.error || 'Generation failed');
        break;

      default:
        break;
    }
  };

  const handleVariationClick = (room: Room, variationIndex: number) => {
    const variation = room.variations[variationIndex];
    if (variation.status === 'completed' && variation.image_url) {
      // Open image in a modal or new tab
      window.open(variation.image_url, '_blank');
    }
  };

  const handleRetryVariation = async (room: Room, variationIndex: number) => {
    try {
      toast.info('Retrying variation generation...');
      // Start room regeneration stream
      const cleanup = streamRoomRegeneration(projectId, room.id, handleStreamEvent);
      
      // Cleanup after some time or when component unmounts
      setTimeout(() => {
        if (cleanup) cleanup();
      }, 30000);
    } catch (error) {
      console.error('Failed to retry variation:', error);
      toast.error('Failed to retry variation');
    }
  };

  const handleAddRooms = () => {
    // Navigate to add rooms page (could be implemented later)
    toast.info('Add rooms feature coming soon');
  };

  const handleRegenerateAll = () => {
    // Start regeneration for all rooms
    toast.info('Regenerate all feature coming soon');
  };

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
            <h1 className="text-3xl font-bold">{project.name}</h1>
            <p className="text-muted-foreground leading-relaxed max-w-3xl">
              {project.prompt}
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="outline" onClick={handleAddRooms}>
              <Plus className="h-4 w-4 mr-2" />
              Add Rooms
            </Button>
            <Button variant="outline" onClick={handleRegenerateAll}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Regenerate All
            </Button>
          </div>
        </div>
      </div>

      {/* Progress Tracker (only visible if processing) */}
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
            />
          ))
        )}
      </div>
    </div>
  );
}