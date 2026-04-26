"use client"

import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { StagingProject } from "@/services/stagingApi";

interface ProgressTrackerProps {
  project: StagingProject;
}

export function ProgressTracker({ project }: ProgressTrackerProps) {
  // Only show if project is processing
  if (project.status !== 'processing') {
    return null;
  }

  const progressPercentage = project.total_variations > 0 
    ? (project.completed_variations / project.total_variations) * 100 
    : 0;

  const getRoomStatusVariant = (status: string): "default" | "secondary" | "destructive" | "outline" => {
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

  return (
    <div className="space-y-4 p-4 bg-muted/30 rounded-lg border">
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-sm text-muted-foreground">Generation Progress</h3>
        <Badge variant="secondary" className="animate-pulse">
          Processing...
        </Badge>
      </div>

      {/* Overall Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span>Overall Progress</span>
          <span className="font-medium">
            {project.completed_variations}/{project.total_variations} variations
          </span>
        </div>
        <Progress value={progressPercentage} className="h-2" />
        <div className="text-xs text-muted-foreground text-right">
          {Math.round(progressPercentage)}% complete
        </div>
      </div>

      {/* Per-room Status Pills */}
      <div className="space-y-2">
        <div className="text-sm font-medium text-muted-foreground">Room Status</div>
        <div className="flex flex-wrap gap-2">
          {project.rooms.map((room) => {
            const roomCompletedVariations = room.variations.filter(v => v.status === 'completed').length;
            const roomTotalVariations = room.variations.length;

            return (
              <div key={room.id} className="flex items-center gap-2">
                <Badge variant={getRoomStatusVariant(room.status)} className="text-xs">
                  {room.name}
                </Badge>
                <span className="text-xs text-muted-foreground">
                  {roomCompletedVariations}/{roomTotalVariations}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}