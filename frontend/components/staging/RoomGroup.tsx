"use client"

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { RefreshCw, Play, Clock, Info } from "lucide-react";
import { VariationThumbnail } from "./VariationThumbnail";
import { Room } from "@/services/stagingApi";

interface RoomGroupProps {
  room: Room;
  onVariationClick?: (room: Room, variationIndex: number) => void;
  onRetryVariation?: (room: Room, variationIndex: number) => void;
  onRegenerateRoom?: (room: Room) => void;
  isGenerating?: boolean;
}

export function RoomGroup({ room, onVariationClick, onRetryVariation, onRegenerateRoom, isGenerating }: RoomGroupProps) {
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
        return `All ${totalCount} variations generated`;
      case 'failed':
        return `${failedCount} variation${failedCount !== 1 ? 's' : ''} failed — click retry to regenerate`;
      default:
        return '';
    }
  };

  return (
    <div className="space-y-4">
      {/* Room Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-semibold">{room.label}</h3>
          <Badge variant={getStatusVariant(room.status)} className="text-xs">
            {room.status}
          </Badge>
          {totalCount > 0 && (
            <span className="text-xs text-muted-foreground">
              {completedCount}/{totalCount} variations
            </span>
          )}
        </div>
        {onRegenerateRoom && (room.status === 'failed' || room.status === 'completed') && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => onRegenerateRoom(room)}
            disabled={isGenerating}
          >
            <RefreshCw className="h-3.5 w-3.5 mr-1" />
            Regenerate
          </Button>
        )}
      </div>

      {/* Status insight message */}
      {room.status !== 'completed' && (
        <div className={`flex items-center gap-2 text-xs px-3 py-2 rounded-md ${
          room.status === 'pending' ? 'bg-muted/50 text-muted-foreground' :
          room.status === 'processing' ? 'bg-blue-500/10 text-blue-600 dark:text-blue-400' :
          room.status === 'failed' ? 'bg-destructive/10 text-destructive' : ''
        }`}>
          {room.status === 'pending' && <Clock className="h-3.5 w-3.5" />}
          {room.status === 'processing' && <Info className="h-3.5 w-3.5" />}
          {room.status === 'failed' && <Info className="h-3.5 w-3.5" />}
          {getStatusMessage()}
        </div>
      )}

      {/* Room Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
        {/* Original Image */}
        <div className="relative">
          <div className="aspect-square w-full min-h-[120px] relative">
            <img 
              src={room.original_image_url} 
              alt={`${room.label} original`}
              className="w-full h-full object-cover rounded-lg border-2 border-amber-400"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                target.parentElement!.classList.add('bg-muted', 'flex', 'items-center', 'justify-content-center');
                target.parentElement!.innerHTML = '<span class="text-xs text-muted-foreground p-2 text-center">Image unavailable — check storage access</span>';
              }}
            />
            <Badge 
              variant="secondary" 
              className="absolute top-2 right-2 bg-amber-400 text-amber-900 text-xs font-medium"
            >
              ORIGINAL
            </Badge>
          </div>
        </div>

        {/* Variation Thumbnails */}
        {room.variations.map((variation, index) => (
          <VariationThumbnail
            key={variation.id}
            imageUrl={variation.image_url}
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
          />
        ))}
      </div>
    </div>
  );
}