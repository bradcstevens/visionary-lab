"use client"

import Link from "next/link";
import { Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { StagingProject } from "@/services/stagingApi";
import { formatDistanceToNow } from "date-fns";

interface ProjectCardProps {
  project: StagingProject;
  onDelete?: (projectId: string) => void;
}

export function ProjectCard({ project, onDelete }: ProjectCardProps) {
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

  const completedVariations = project.rooms.reduce((sum, r) => sum + r.variations.filter(v => v.status === 'completed').length, 0);
  const totalVariations = project.rooms.reduce((sum, r) => sum + r.variations.length, 0);
  const previewRooms = project.rooms.slice(0, 4);
  const remainingRoomsCount = Math.max(0, project.rooms.length - 4);

  return (
    <Card className="h-full hover:shadow-lg transition-shadow relative group">
      <Link href={`/projects/${project.id}`} className="block">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-2">
            <CardTitle className="text-lg line-clamp-2">{project.name}</CardTitle>
            <Badge variant={getStatusVariant(project.status)} className="text-xs shrink-0">
              {project.status}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="pb-3">
          <div className="grid grid-cols-4 gap-2 mb-3">
            {previewRooms.map((room) => (
              <div key={room.id} className="aspect-square relative">
                <img
                  src={room.original_image_url}
                  alt={`${room.label} preview`}
                  className="w-full h-full object-cover rounded-md bg-muted"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                />
                {room.variations.some(v => v.status === 'completed') && (
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-background" />
                )}
              </div>
            ))}

            {remainingRoomsCount > 0 && (
              <div className="aspect-square bg-muted rounded-md flex items-center justify-center">
                <span className="text-sm font-medium text-muted-foreground">+{remainingRoomsCount}</span>
              </div>
            )}

            {previewRooms.length < 4 && remainingRoomsCount === 0 && (
              Array.from({ length: 4 - previewRooms.length }).map((_, i) => (
                <div key={`empty-${i}`} className="aspect-square bg-muted/50 rounded-md border border-dashed border-muted-foreground/30" />
              ))
            )}
          </div>

          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>{project.rooms.length} image{project.rooms.length !== 1 ? 's' : ''}</span>
            <span>{completedVariations}/{totalVariations} variations</span>
          </div>
        </CardContent>

        <CardFooter className="pt-0 flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            {project.created_at ? formatDistanceToNow(new Date(project.created_at), { addSuffix: true }) : ''}
          </span>
        </CardFooter>
      </Link>

      {/* Delete button — visible on hover */}
      {onDelete && (
        <Button
          size="icon"
          variant="ghost"
          className="absolute top-2 right-12 h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity text-destructive hover:bg-destructive/10"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onDelete(project.id);
          }}
          title="Delete project"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
      )}
    </Card>
  );
}