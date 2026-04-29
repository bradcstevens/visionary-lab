"use client"

import { useState, useCallback, useRef } from "react";
import { ImageOff } from "lucide-react";
import { sasTokenService } from "@/services/sas-token";
import { cn } from "@/utils/cn";

interface StorageImageProps {
  /** Image URL — may include SAS token or be a bare blob URL. */
  src: string | undefined | null;
  alt: string;
  className?: string;
  /** Class applied to the fallback container shown on error. */
  fallbackClassName?: string;
  /** Text shown in the fallback state. */
  fallbackText?: string;
  /** Overlay elements (badges, icons) rendered on top of the image. */
  overlay?: React.ReactNode;
}

/**
 * Image component for Azure Blob Storage URLs.
 *
 * On the first load error it automatically retries with a freshly-fetched
 * SAS token (invalidating the cached one).  If the second attempt also
 * fails it shows a compact fallback placeholder.
 */
export function StorageImage({
  src,
  alt,
  className,
  fallbackClassName,
  fallbackText = "Preview unavailable",
  overlay,
}: StorageImageProps) {
  const [status, setStatus] = useState<"loading" | "loaded" | "error">("loading");
  const [currentSrc, setCurrentSrc] = useState(src ?? "");
  const retriedRef = useRef(false);

  // Reset state when src changes externally (e.g. page re-fetch)
  const lastSrcRef = useRef(src);
  if (src !== lastSrcRef.current) {
    lastSrcRef.current = src;
    retriedRef.current = false;
    setCurrentSrc(src ?? "");
    setStatus(src ? "loading" : "error");
  }

  const handleError = useCallback(async () => {
    if (retriedRef.current || !src) {
      setStatus("error");
      return;
    }
    retriedRef.current = true;

    try {
      sasTokenService.invalidate();
      const tokens = await sasTokenService.getTokens();
      const bareUrl = src.split("?")[0];
      setCurrentSrc(`${bareUrl}?${tokens.imageSasToken}`);
    } catch {
      setStatus("error");
    }
  }, [src]);

  if (!src || status === "error") {
    return (
      <div className={cn("bg-muted flex items-center justify-center", fallbackClassName)}>
        <div className="flex flex-col items-center gap-1 p-2">
          <ImageOff className="h-5 w-5 text-muted-foreground/50" />
          <span className="text-[10px] text-muted-foreground text-center leading-tight">
            {fallbackText}
          </span>
        </div>
      </div>
    );
  }

  return (
    <>
      {status === "loading" && (
        <div className={cn("absolute inset-0 bg-muted animate-pulse", fallbackClassName)} />
      )}
      <img
        src={currentSrc}
        alt={alt}
        className={className}
        onLoad={() => setStatus("loaded")}
        onError={handleError}
      />
      {overlay}
    </>
  );
}
