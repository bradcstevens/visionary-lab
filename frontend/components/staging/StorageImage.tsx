"use client"

import { useState, useCallback, useRef } from "react";
import { ImageOff, RefreshCw } from "lucide-react";
import { sasTokenService } from "@/services/sas-token";
import { cn } from "@/utils/cn";

/**
 * Issue 011 of image-pipeline-and-project-ux-overhaul PRD: derived
 * variant the caller wants. The component picks the matching prop:
 *
 *   - "thumb"    -> thumbUrl    (grid tiles, 512px WebP)
 *   - "md"       -> mdUrl       (lightbox preview, 1024px WebP)
 *   - "original" -> originalUrl (download / external open)
 *
 * Legacy callers omit ``variant`` and pass ``src`` directly.
 */
export type StorageImageVariant = "thumb" | "md" | "original";

interface StorageImageProps {
  /**
   * Legacy single-URL prop. Used when ``variant`` is unset. Existing
   * call sites (``RoomGroup``, ``ProjectCard``, ``ProjectRoomsManager``)
   * still pass this directly.
   */
  src?: string | null;
  /**
   * Issue 011: explicit variant URLs. Each is independent so a caller
   * can pass thumb + md + original for the lightbox while a grid only
   * passes thumb. Missing-but-requested = skeleton, not silent fallback,
   * so the bug "thumbnail grid renders the multi-MB original because
   * thumb_url was missing" can never recur.
   */
  thumbUrl?: string | null;
  mdUrl?: string | null;
  originalUrl?: string | null;
  /** Issue 011: which variant URL to render. */
  variant?: StorageImageVariant;
  alt: string;
  className?: string;
  /** Class applied to the skeleton/error container. */
  fallbackClassName?: string;
  /** Text shown in the terminal-error state. */
  fallbackText?: string;
  /**
   * Optional reflection of the upstream job status. When the variant
   * URL is missing we render a skeleton; if status is "failed" the
   * skeleton swaps for an error glyph + retry. Defaults to undefined
   * (treat missing URL as "still loading").
   */
  jobStatus?: "pending" | "processing" | "completed" | "failed";
  /** Overlay elements (badges, icons, dropdowns) rendered on top of the loaded image. */
  overlay?: React.ReactNode;
  /**
   * Optional ARIA / test-id label applied to the retry button so
   * Playwright assertions can target it deterministically.
   */
  retryLabel?: string;
}

function pickUrl(
  variant: StorageImageVariant | undefined,
  thumbUrl: string | null | undefined,
  mdUrl: string | null | undefined,
  originalUrl: string | null | undefined,
  src: string | null | undefined,
): string | undefined {
  if (!variant) return src ?? undefined;
  switch (variant) {
    case "thumb": return thumbUrl ?? undefined;
    case "md": return mdUrl ?? undefined;
    case "original": return originalUrl ?? undefined;
  }
}

/**
 * Image component for Azure Blob Storage URLs.
 *
 * On the first load error it auto-retries with a freshly-fetched SAS
 * token (the cached token may have expired). If the second attempt
 * also fails, a manual retry button is shown — clicking it
 * invalidates the SAS cache and re-attempts the load. This replaces
 * the prior silent ``ImageOff`` fallback so users always have a
 * recovery affordance per issue 011 AC bullet 3.
 *
 * When the chosen variant URL is missing entirely (e.g. ``thumb_url``
 * not yet backfilled, or status pending), a skeleton is shown
 * instead of an error placeholder.
 */
export function StorageImage({
  src,
  thumbUrl,
  mdUrl,
  originalUrl,
  variant,
  alt,
  className,
  fallbackClassName,
  fallbackText = "Preview unavailable",
  jobStatus,
  overlay,
  retryLabel,
}: StorageImageProps) {
  const resolvedSrc = pickUrl(variant, thumbUrl, mdUrl, originalUrl, src);

  const [status, setStatus] = useState<"loading" | "loaded" | "error">(
    resolvedSrc ? "loading" : "error",
  );
  // ``displaySrc`` is what we actually render. We diverge from
  // resolvedSrc after a SAS-refresh retry so the cache-busted URL
  // doesn't get clobbered by a parent re-render with the same src.
  const [displaySrc, setDisplaySrc] = useState(resolvedSrc ?? "");
  // Manual-retry counter forces React to remount the <img> after the
  // user clicks Retry even if the URL is unchanged.
  const [retryNonce, setRetryNonce] = useState(0);
  const autoRetriedRef = useRef(false);

  // Reset state when the resolved URL changes externally (e.g. parent
  // refetches the project). Done synchronously during render so the
  // first paint after a src change is the loading skeleton, not a
  // stale loaded image. ``lastSrcRef`` mirrors the previously
  // committed resolvedSrc so we only reset on actual change.
  const lastSrcRef = useRef(resolvedSrc);
  if (resolvedSrc !== lastSrcRef.current) {
    lastSrcRef.current = resolvedSrc;
    autoRetriedRef.current = false;
    setDisplaySrc(resolvedSrc ?? "");
    setStatus(resolvedSrc ? "loading" : "error");
    setRetryNonce(0);
  }

  const handleError = useCallback(async () => {
    if (!resolvedSrc) {
      setStatus("error");
      return;
    }
    if (autoRetriedRef.current) {
      // Second failure: surface the manual retry control.
      setStatus("error");
      return;
    }
    autoRetriedRef.current = true;
    try {
      sasTokenService.invalidate();
      const tokens = await sasTokenService.getTokens();
      const bareUrl = resolvedSrc.split("?")[0];
      setDisplaySrc(`${bareUrl}?${tokens.imageSasToken}`);
    } catch {
      setStatus("error");
    }
  }, [resolvedSrc]);

  const handleManualRetry = useCallback(async () => {
    if (!resolvedSrc) return;
    sasTokenService.invalidate();
    autoRetriedRef.current = false;
    try {
      const tokens = await sasTokenService.getTokens();
      const bareUrl = resolvedSrc.split("?")[0];
      setDisplaySrc(`${bareUrl}?${tokens.imageSasToken}`);
    } catch {
      // Keep the current URL; the load attempt below will surface
      // the error again via onError if it still fails.
      setDisplaySrc(resolvedSrc);
    }
    setRetryNonce((n) => n + 1);
    setStatus("loading");
  }, [resolvedSrc]);

  // Skeleton: variant URL missing AND we're not in a known-failed state.
  // Skeleton also shown for jobStatus pending/processing so a still-
  // generating tile never collapses to an error placeholder.
  const isSkeleton =
    !resolvedSrc &&
    jobStatus !== "failed";

  if (isSkeleton) {
    return (
      <div
        data-testid="storage-image-skeleton"
        aria-busy="true"
        aria-label={alt}
        className={cn(
          "bg-muted animate-pulse rounded-lg",
          fallbackClassName,
        )}
      />
    );
  }

  if (status === "error") {
    return (
      <div
        data-testid="storage-image-error"
        className={cn(
          "bg-muted flex items-center justify-center",
          fallbackClassName,
        )}
      >
        <div className="flex flex-col items-center gap-1.5 p-2">
          <ImageOff className="h-5 w-5 text-muted-foreground/50" />
          <span className="text-[10px] text-muted-foreground text-center leading-tight">
            {fallbackText}
          </span>
          {resolvedSrc && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                void handleManualRetry();
              }}
              aria-label={retryLabel ?? "Retry loading image"}
              data-testid="storage-image-retry"
              className="mt-1 inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] text-foreground bg-background hover:bg-muted-foreground/10 border border-border"
            >
              <RefreshCw className="h-3 w-3" />
              Retry
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <>
      {status === "loading" && (
        <div
          data-testid="storage-image-skeleton"
          aria-busy="true"
          className={cn("absolute inset-0 bg-muted animate-pulse", fallbackClassName)}
        />
      )}
      <img
        key={retryNonce}
        src={displaySrc}
        alt={alt}
        className={className}
        onLoad={() => setStatus("loaded")}
        onError={handleError}
      />
      {overlay}
    </>
  );
}
