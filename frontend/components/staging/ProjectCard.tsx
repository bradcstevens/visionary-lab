"use client"

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { StagingProject } from "@/services/stagingApi";
import { formatDistanceToNow } from "date-fns";

interface ProjectCardProps {
  project: StagingProject;
}

export function ProjectCard({ project }: ProjectCardProps) {
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

  // Get up to 4 room thumbnails for preview
  const previewRooms = project.rooms.slice(0, 4);
  const remainingRoomsCount = Math.max(0, project.rooms.length - 4);

  return (
    <Link href={`/projects/${project.id}`} className="block transition-transform hover:scale-[1.02]">
      <Card className="h-full hover:shadow-lg transition-shadow">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-2">
            <CardTitle className="text-lg line-clamp-2">{project.name}</CardTitle>
            <Badge variant={getStatusVariant(project.status)} className="text-xs shrink-0">
              {project.status}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="pb-3">
          {/* Room Thumbnails Preview */}
          <div className="grid grid-cols-4 gap-2 mb-3">
            {previewRooms.map((room, index) => (
              <div key={room.id} className="aspect-square relative">
                <img
                  src={room.original_image_url}
                  alt={`${room.label} preview`}
                  className="w-full h-full object-cover rounded-md bg-muted"
                />
                {/* Show a small indicator if room has completed variations */}
                {room.variations.some(v => v.status === 'completed') && (
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-background" />
                )}
              </div>
            ))}

            {/* Show overflow count if more than 4 rooms */}
            {remainingRoomsCount > 0 && (
              <div className="aspect-square bg-muted rounded-md flex items-center justify-center">
                <span className="text-sm font-medium text-muted-foreground">
                  +{remainingRoomsCount}
                </span>
              </div>
            )}

            {/* Fill empty slots if less than 4 rooms */}
            {previewRooms.length < 4 && remainingRoomsCount === 0 && (
              Array.from({ length: 4 - previewRooms.length }).map((_, index) => (
                <div key={`empty-${index}`} className="aspect-square bg-muted/50 rounded-md border border-dashed border-muted-foreground/30" />
              ))
            )}
          </div>

          {/* Project Stats */}
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>{project.rooms.length} room{project.rooms.length !== 1 ? 's' : ''}</span>
            <span>
              {project.completed_variations}/{project.total_variations} variations
            </span>
          </div>
        </CardContent>

        <CardFooter className="pt-0 flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            {formatDistanceToNow(new Date(project.created_at), { addSuffix: true })}
          </span>
        </CardFooter>
      </Card>
    </Link>
  );
}