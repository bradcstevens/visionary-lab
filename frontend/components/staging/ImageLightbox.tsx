"use client"

import { useEffect, useCallback, useMemo } from "react";
import { Dialog, DialogPortal, DialogOverlay, DialogTitle } from "@/components/ui/dialog";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X, ExternalLink, RefreshCw, RotateCcw, Sparkles, Loader2, ChevronLeft, ChevronRight, ImageOff } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { StorageImage } from "./StorageImage";
import { cn } from "@/utils/cn";
import type { Variation } from "@/services/stagingApi";

export interface LightboxImage {
  url: string;
  roomLabel: string;
  variationIndex: number;
  /** All variations for this room — enables prev/next navigation. */
  variations: Variation[];
}

interface ImageLightboxProps {
  image: LightboxImage | null;
  onClose: () => void;
  onNavigate: (variationIndex: number) => void;
  onRegenerate?: (strategy: 'retry' | 'fresh') => void;
  isRegenerating?: boolean;
}

export function ImageLightbox({ image, onClose, onNavigate, onRegenerate, isRegenerating }: ImageLightboxProps) {
  // Completed variations for navigation
  const completedIndices = useMemo(() => {
    if (!image) return [];
    return image.variations
      .map((v, i) => (v.status === "completed" && v.image_url ? i : -1))
      .filter((i) => i >= 0);
  }, [image]);

  const currentPos = image ? completedIndices.indexOf(image.variationIndex) : -1;
  const hasPrev = currentPos > 0;
  const hasNext = currentPos < completedIndices.length - 1;

  const goPrev = useCallback(() => {
    if (hasPrev) onNavigate(completedIndices[currentPos - 1]);
  }, [hasPrev, currentPos, completedIndices, onNavigate]);

  const goNext = useCallback(() => {
    if (hasNext) onNavigate(completedIndices[currentPos + 1]);
  }, [hasNext, currentPos, completedIndices, onNavigate]);

  // Scoped keyboard navigation — only while dialog is open
  useEffect(() => {
    if (!image) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") { e.preventDefault(); goPrev(); }
      if (e.key === "ArrowRight") { e.preventDefault(); goNext(); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [image, goPrev, goNext]);

  return (
    <Dialog open={!!image} onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogPortal>
        <DialogOverlay className="bg-black/95" />
        <DialogPrimitive.Content
          className={cn(
            "fixed inset-0 z-50 flex flex-col items-center",
            "p-3 pt-4 sm:p-6 sm:pt-5 lg:p-10 lg:pt-6",
            "data-[state=open]:animate-in data-[state=closed]:animate-out",
            "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
          )}
          onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
        >
          {/* Accessible title (visually hidden) */}
          <DialogTitle className="sr-only">
            {image ? `${image.roomLabel} — Variation ${image.variationIndex + 1}` : "Image preview"}
          </DialogTitle>

          {/* ── Toolbar — anchored at top ── */}
          <div className={cn(
            "flex items-center justify-between w-full max-w-5xl mb-auto",
            "rounded-xl px-3 py-2 sm:px-4 sm:py-2.5",
            "bg-white/[0.08] border border-white/[0.1]",
            "backdrop-blur-2xl shadow-lg shadow-black/30",
          )}>
            {/* Left: label + counter */}
            <div className="flex items-center gap-2 sm:gap-3 min-w-0">
              <p className="text-sm text-white/90 font-medium truncate">
                {image?.roomLabel} — Variation {(image?.variationIndex ?? 0) + 1}
              </p>
              {completedIndices.length > 1 && (
                <span className="text-xs text-white/50 font-mono tabular-nums whitespace-nowrap">
                  {currentPos + 1} / {completedIndices.length}
                </span>
              )}
            </div>

            {/* Right: actions */}
            <div className="flex items-center gap-0.5 sm:gap-1 shrink-0">
              {onRegenerate && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-white/70 hover:text-white hover:bg-white/10 h-8 px-2"
                      disabled={isRegenerating}
                      aria-label="Regenerate this variation"
                    >
                      {isRegenerating ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <RefreshCw className="h-4 w-4" />
                      )}
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-48">
                    <DropdownMenuItem onClick={() => onRegenerate('retry')}>
                      <RotateCcw className="h-4 w-4 mr-2" />
                      Retry Same Prompt
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => onRegenerate('fresh')}>
                      <Sparkles className="h-4 w-4 mr-2" />
                      Try Something New
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
              {image?.url && (
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-white/60 hover:text-white hover:bg-white/10 h-8 px-2 sm:px-2.5 rounded-lg transition-colors"
                  onClick={() => window.open(image.url, "_blank")}
                  aria-label="Open full image in new tab"
                >
                  <ExternalLink className="h-4 w-4" />
                </Button>
              )}
              <DialogPrimitive.Close asChild>
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-white/60 hover:text-white hover:bg-white/10 h-8 w-8 p-0 rounded-lg transition-colors"
                  aria-label="Close"
                >
                  <X className="h-4.5 w-4.5" />
                </Button>
              </DialogPrimitive.Close>
            </div>
          </div>

          {/* ── Image area — vertically centered in remaining space ── */}
          {image && (
            <div
              className={cn(
                "relative flex items-center justify-center w-full max-w-5xl",
                "my-auto",
                "animate-in zoom-in-[0.97] fade-in-0 duration-300 ease-out",
              )}
              style={{ maxHeight: "calc(100vh - 8rem)" }}
            >
              {/* Previous arrow */}
              {hasPrev && (
                <button
                  onClick={(e) => { e.stopPropagation(); goPrev(); }}
                  className={cn(
                    "absolute left-1 sm:left-3 z-10",
                    "h-10 w-10 sm:h-11 sm:w-11 rounded-full flex items-center justify-center",
                    "bg-black/50 border border-white/[0.12] backdrop-blur-md",
                    "text-white/80 hover:text-white hover:bg-black/70 hover:border-white/20",
                    "transition-all duration-200 cursor-pointer",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/50",
                  )}
                  aria-label="Previous variation"
                >
                  <ChevronLeft className="h-5 w-5" />
                </button>
              )}

              {/* Image frame */}
              <div className={cn(
                "relative rounded-xl overflow-hidden",
                "ring-1 ring-white/[0.08]",
                "shadow-[0_8px_60px_-12px_rgba(0,0,0,0.8)]",
                "bg-neutral-900/80",
              )}>
                <StorageImage
                  src={image.url}
                  alt={`${image.roomLabel} variation ${image.variationIndex + 1}`}
                  className={cn(
                    "block max-w-[88vw] sm:max-w-[82vw] lg:max-w-4xl max-h-[calc(100vh-10rem)] w-auto h-auto object-contain",
                    isRegenerating && "opacity-40"
                  )}
                  fallbackClassName="w-[60vw] sm:w-[50vw] lg:w-[40vw] aspect-[4/3] rounded-xl bg-neutral-900/80"
                  fallbackText="Image could not be loaded"
                  overlay={
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 pointer-events-none">
                      <div className="h-12 w-12 rounded-full bg-white/[0.05] flex items-center justify-center">
                        <ImageOff className="h-6 w-6 text-white/20" />
                      </div>
                      <span className="text-sm text-white/30">Image could not be loaded</span>
                    </div>
                  }
                />
                {isRegenerating && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="flex flex-col items-center gap-3">
                      <Loader2 className="h-8 w-8 animate-spin text-white" />
                      <span className="text-sm text-white/80 font-medium">Regenerating...</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Next arrow */}
              {hasNext && (
                <button
                  onClick={(e) => { e.stopPropagation(); goNext(); }}
                  className={cn(
                    "absolute right-1 sm:right-3 z-10",
                    "h-10 w-10 sm:h-11 sm:w-11 rounded-full flex items-center justify-center",
                    "bg-black/50 border border-white/[0.12] backdrop-blur-md",
                    "text-white/80 hover:text-white hover:bg-black/70 hover:border-white/20",
                    "transition-all duration-200 cursor-pointer",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/50",
                  )}
                  aria-label="Next variation"
                >
                  <ChevronRight className="h-5 w-5" />
                </button>
              )}
            </div>
          )}

          {/* Bottom spacer to balance top toolbar */}
          <div className="mb-auto" />
        </DialogPrimitive.Content>
      </DialogPortal>
    </Dialog>
  );
}
