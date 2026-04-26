"use client"

import { AlertCircle, RefreshCw, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/utils/cn";

interface VariationThumbnailProps {
  imageUrl?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  error?: string;
  index: number;
  onClick?: () => void;
  onRetry?: () => void;
}

export function VariationThumbnail({ 
  imageUrl, 
  status, 
  error, 
  index, 
  onClick, 
  onRetry 
}: VariationThumbnailProps) {
  const renderContent = () => {
    switch (status) {
      case 'completed':
        return (
          <div className="relative w-full h-full group cursor-pointer" onClick={onClick}>
            <img 
              src={imageUrl} 
              alt={`Variation ${index + 1}`}
              className="w-full h-full object-cover rounded-lg"
            />
            <Badge 
              variant="secondary" 
              className="absolute top-2 right-2 bg-black/70 text-white text-xs"
            >
              {index + 1}
            </Badge>
          </div>
        );

      case 'processing':
        return (
          <div className="w-full h-full bg-muted rounded-lg flex items-center justify-center">
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              <Badge variant="secondary" className="text-xs">
                {index + 1}
              </Badge>
            </div>
          </div>
        );

      case 'failed':
        return (
          <div className="w-full h-full bg-destructive/10 rounded-lg flex items-center justify-center border-2 border-destructive/20">
            <div className="flex flex-col items-center gap-2">
              <AlertCircle className="h-6 w-6 text-destructive" />
              <Badge variant="destructive" className="text-xs">
                {index + 1}
              </Badge>
              {onRetry && (
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 w-6 p-0 hover:bg-destructive/20"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRetry();
                  }}
                  title={error || "Retry generation"}
                >
                  <RefreshCw className="h-3 w-3" />
                </Button>
              )}
            </div>
          </div>
        );

      case 'pending':
      default:
        return (
          <div className="w-full h-full bg-muted/50 rounded-lg border-2 border-dashed border-muted-foreground/30 flex items-center justify-center">
            <Badge variant="outline" className="text-xs bg-background">
              {index + 1}
            </Badge>
          </div>
        );
    }
  };

  return (
    <div 
      className={cn(
        "aspect-square w-full min-h-[120px] transition-all duration-200",
        onClick && status === 'completed' && "hover:scale-[1.02] hover:shadow-md"
      )}
    >
      {renderContent()}
    </div>
  );
}