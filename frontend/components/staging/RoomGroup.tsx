"use client"

import { Badge } from "@/components/ui/badge";
import { VariationThumbnail } from "./VariationThumbnail";
import { Room } from "@/services/stagingApi";

interface RoomGroupProps {
  room: Room;
  onVariationClick?: (room: Room, variationIndex: number) => void;
  onRetryVariation?: (room: Room, variationIndex: number) => void;
}

export function RoomGroup({ room, onVariationClick, onRetryVariation }: RoomGroupProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500';
      case 'processing':
        return 'bg-blue-500';
      case 'failed':
        return 'bg-red-500';
      case 'pending':
      default:
        return 'bg-gray-500';
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

  return (
    <div className="space-y-4">
      {/* Room Header */}
      <div className="flex items-center gap-3">
        <h3 className="text-lg font-semibold">{room.name}</h3>
        <Badge variant={getStatusVariant(room.status)} className="text-xs">
          {room.status}
        </Badge>
      </div>

      {/* Room Grid */}
      <div className="grid grid-cols-6 gap-4">
        {/* Original Image */}
        <div className="relative">
          <div className="aspect-square w-full min-h-[120px] relative">
            <img 
              src={room.original_image_url} 
              alt={`${room.name} original`}
              className="w-full h-full object-cover rounded-lg border-2 border-amber-400"
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