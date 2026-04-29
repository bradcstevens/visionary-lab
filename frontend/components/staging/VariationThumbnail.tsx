"use client"

import { AlertCircle, RefreshCw, Loader2, RotateCcw, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { StorageImage } from "./StorageImage";
import { cn } from "@/utils/cn";

interface VariationThumbnailProps {
  imageUrl?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  error?: string;
  index: number;
  onClick?: () => void;
  onRetry?: () => void;
  onRegenerate?: (strategy: 'retry' | 'fresh') => void;
  isRegenerating?: boolean;
}

export function VariationThumbnail({ 
  imageUrl, 
  status, 
  error, 
  index, 
  onClick, 
  onRetry,
  onRegenerate,
  isRegenerating,
}: VariationThumbnailProps) {
  const renderContent = () => {
    switch (status) {
      case 'completed':
        if (isRegenerating) {
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
        }
        return (
          <div className="relative w-full h-full group cursor-pointer" onClick={onClick}>
            <StorageImage
              src={imageUrl}
              alt={`Variation ${index + 1}`}
              className="w-full h-full object-cover rounded-lg"
              fallbackClassName="w-full h-full rounded-lg"
              fallbackText="Preview unavailable"
              overlay={
                <>
                  <Badge 
                    variant="secondary" 
                    className="absolute top-2 right-2 bg-black/70 text-white text-xs"
                  >
                    {index + 1}
                  </Badge>
                  {onRegenerate && (
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-colors duration-200 rounded-lg flex items-center justify-center">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            size="sm"
                            variant="secondary"
                            className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 h-8 w-8 p-0 rounded-full bg-white/90 hover:bg-white text-gray-700 shadow-md"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <RefreshCw className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="center" side="top" className="w-48">
                          <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onRegenerate('retry'); }}>
                            <RotateCcw className="h-4 w-4 mr-2" />
                            Retry Same Prompt
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onRegenerate('fresh'); }}>
                            <Sparkles className="h-4 w-4 mr-2" />
                            Try Something New
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  )}
                </>
              }
            />
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

      case 'failed': {
        const shortError = error
          ? error.length > 60 ? error.slice(0, 57) + "…" : error
          : "Generation failed";

        const thumbnail = (
          <div className="w-full h-full bg-destructive/10 rounded-lg border-2 border-destructive/20 flex flex-col items-center justify-center overflow-hidden p-2 gap-1">
            <AlertCircle className="h-5 w-5 text-destructive shrink-0" />
            <Badge variant="destructive" className="text-xs shrink-0">
              {index + 1}
            </Badge>
            <span className="text-[10px] text-destructive/80 text-center leading-tight line-clamp-2 break-words w-full px-1">
              {shortError}
            </span>
            {onRetry && (
              <Button
                size="sm"
                variant="ghost"
                className="h-6 px-2 text-[10px] hover:bg-destructive/20 shrink-0"
                onClick={(e) => {
                  e.stopPropagation();
                  onRetry();
                }}
              >
                <RefreshCw className="h-3 w-3 mr-1" />
                Retry
              </Button>
            )}
          </div>
        );

        if (error && error.length > 60) {
          return (
            <Tooltip>
              <TooltipTrigger asChild>
                {thumbnail}
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-xs break-words whitespace-pre-wrap text-xs">
                {error}
              </TooltipContent>
            </Tooltip>
          );
        }

        return thumbnail;
      }

      case 'pending':
      default:
        return (
          <div className="w-full h-full bg-muted/50 rounded-lg border-2 border-dashed border-muted-foreground/30 flex items-center justify-center">
            <div className="flex flex-col items-center gap-1.5 p-2">
              <Badge variant="outline" className="text-xs bg-background">
                {index + 1}
              </Badge>
              <span className="text-[10px] text-muted-foreground text-center leading-tight">
                Awaiting generation
              </span>
            </div>
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