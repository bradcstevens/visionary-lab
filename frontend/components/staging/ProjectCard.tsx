"use client"

import Link from "next/link";
import { MoreHorizontal, Trash2, ExternalLink } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { StagingProject } from "@/services/stagingApi";
import { formatDistanceToNow } from "date-fns";

interface ProjectCardProps {
  project: StagingProject;
  onDelete?: (projectId: string) => void;
}

export function ProjectCard({ project, onDelete }: ProjectCardProps) {
  const getStatusVariant = (status: string): "default" | "secondary" | "destructive" | "outline" => {
    switch (status) {
      case 'completed': return 'default';
      case 'processing': return 'secondary';
      case 'failed': return 'destructive';
      default: return 'outline';
    }
  };

  const completedVariations = project.rooms.reduce((sum, r) => sum + r.variations.filter(v => v.status === 'completed').length, 0);
  const totalVariations = project.rooms.reduce((sum, r) => sum + r.variations.length, 0);
  const previewRooms = project.rooms.slice(0, 4);
  const remainingRoomsCount = Math.max(0, project.rooms.length - 4);

  return (
    <Card className="h-full hover:shadow-lg transition-shadow group">
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
              <div key={room.id} className="aspect-square relative rounded-md overflow-hidden bg-muted">
                <img
                  src={room.original_image_url}
                  alt={`${room.label} preview`}
                  className="w-full h-full object-cover"
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
      </Link>

      <CardFooter className="pt-0 flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {project.created_at ? formatDistanceToNow(new Date(project.created_at), { addSuffix: true }) : ''}
        </span>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-7 w-7 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 transition-opacity">
              <MoreHorizontal className="h-4 w-4" />
              <span className="sr-only">Project actions</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem asChild>
              <Link href={`/projects/${project.id}`}>
                <ExternalLink className="h-3.5 w-3.5 mr-2" />
                Open project
              </Link>
            </DropdownMenuItem>
            {onDelete && (
              <DropdownMenuItem
                className="text-destructive focus:text-destructive"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(project.id);
                }}
              >
                <Trash2 className="h-3.5 w-3.5 mr-2" />
                Delete project
              </DropdownMenuItem>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </CardFooter>
    </Card>
  );
}
