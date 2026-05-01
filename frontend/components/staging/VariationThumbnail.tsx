"use client"

import { AlertCircle, RefreshCw, Loader2, RotateCcw, Pencil, Clock } from "lucide-react";
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
  // Issue 004 of projects-page-improvements PRD: per-variation Edit
  // Prompt opens a dialog with the variation's prior adapted_prompt
  // prefilled. Distinct from onRegenerate('fresh') (the prior "Try
  // Something New") because Edit Prompt APPENDS a new variation
  // instead of mutating in place.
  onEditPrompt?: () => void;
  isRegenerating?: boolean;
  isQueued?: boolean;
}

export function VariationThumbnail({ 
  imageUrl, 
  status, 
  error, 
  index, 
  onClick, 
  onRetry,
  onRegenerate,
  onEditPrompt,
  isRegenerating,
  isQueued,
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
                  {(onRegenerate || onEditPrompt) && (
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          size="sm"
                          variant="secondary"
                          aria-label={`Regenerate variation ${index + 1}`}
                          className="absolute bottom-2 right-2 h-8 w-8 p-0 rounded-full bg-white/80 hover:bg-white text-gray-700 shadow-sm backdrop-blur-sm"
                          onClick={(e) => e.stopPropagation()}
                          data-testid={`variation-${index + 1}-regen-trigger`}
                        >
                          <RefreshCw className="h-3.5 w-3.5" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" side="top" className="w-48">
                        {onRegenerate && (
                          <DropdownMenuItem
                            onClick={(e) => { e.stopPropagation(); onRegenerate('retry'); }}
                            data-testid={`variation-${index + 1}-retry-same-prompt`}
                          >
                            <RotateCcw className="h-4 w-4 mr-2" />
                            Retry Same Prompt
                          </DropdownMenuItem>
                        )}
                        {onEditPrompt && (
                          <DropdownMenuItem
                            onClick={(e) => { e.stopPropagation(); onEditPrompt(); }}
                            data-testid={`variation-${index + 1}-edit-prompt`}
                          >
                            <Pencil className="h-4 w-4 mr-2" />
                            Edit Prompt
                          </DropdownMenuItem>
                        )}
                      </DropdownMenuContent>
                    </DropdownMenu>
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

        const queuedIndicator = isQueued ? (
          <div
            data-testid={`variation-${index + 1}-queued`}
            className="flex flex-col items-center gap-1 shrink-0 mt-1"
          >
            <div className="flex items-center gap-1">
              <Loader2
                aria-hidden="true"
                className="h-3 w-3 animate-spin text-amber-600 dark:text-amber-400"
              />
              <Badge
                variant="outline"
                className="h-5 px-1.5 py-0 text-[10px] border-amber-500 text-amber-700 dark:text-amber-300 bg-amber-500/10"
              >
                <Clock aria-hidden="true" className="h-2.5 w-2.5 mr-0.5" />
                Queued
              </Badge>
            </div>
            <span className="text-[9px] text-amber-700 dark:text-amber-300 text-center leading-tight px-1">
              Will retry when generation finishes
            </span>
          </div>
        ) : null;

        const thumbnail = (
          <div
            aria-busy={!!isQueued}
            className="w-full h-full bg-destructive/10 rounded-lg border-2 border-destructive/20 flex flex-col items-center justify-center overflow-hidden p-2 gap-1"
          >
            <AlertCircle className="h-5 w-5 text-destructive shrink-0" />
            <Badge variant="destructive" className="text-xs shrink-0">
              {index + 1}
            </Badge>
            <span className="text-[10px] text-destructive/80 text-center leading-tight line-clamp-2 break-words w-full px-1">
              {shortError}
            </span>
            {isQueued
              ? queuedIndicator
              : onRetry && (
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
      aria-busy={!!isRegenerating}
      className={cn(
        "aspect-square w-full min-h-[120px] transition-all duration-200",
        onClick && status === 'completed' && "hover:scale-[1.02] hover:shadow-md"
      )}
    >
      {renderContent()}
    </div>
  );
}