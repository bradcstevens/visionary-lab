/**
 * Utilities for parsing and displaying API errors in a user-friendly way.
 */

export interface ParsedError {
  /** Short, human-readable summary (always safe to render) */
  title: string;
  /** Longer detail text, if available (may be truncated raw response) */
  detail: string | null;
  /** HTTP status code, if detected */
  statusCode: number | null;
  /** Whether the raw error was truncated */
  isTruncated: boolean;
}

const STATUS_TITLES: Record<number, string> = {
  400: "Bad Request",
  401: "Authentication Required",
  403: "Access Denied",
  404: "Not Found",
  408: "Request Timeout",
  409: "Conflict",
  422: "Validation Error",
  429: "Too Many Requests",
  500: "Server Error",
  502: "Bad Gateway",
  503: "Service Unavailable",
  504: "Gateway Timeout",
};

const MAX_DETAIL_LENGTH = 500;

/**
 * Parse an error (typically from an API call) into structured, display-friendly parts.
 * Handles raw Error objects, strings, and unknown values.
 */
export function parseApiError(error: unknown): ParsedError {
  const raw = errorToString(error);

  const statusMatch = raw.match(/\b(\d{3})\b/);
  const statusCode = statusMatch ? parseInt(statusMatch[1], 10) : null;

  let title = "Something went wrong";
  let detail: string | null = null;

  if (statusCode && STATUS_TITLES[statusCode]) {
    title = STATUS_TITLES[statusCode];
  }

  // Strip the common "Failed to X: 400 " prefix from API service errors
  const prefixMatch = raw.match(/^Failed to \w[\w\s]*:\s*\d{3}\s*/);
  if (prefixMatch) {
    detail = raw.slice(prefixMatch[0].length).trim() || null;
  } else {
    detail = raw;
  }

  // Try to extract a message from JSON error bodies
  if (detail) {
    try {
      const parsed = JSON.parse(detail);
      if (typeof parsed === "object" && parsed !== null) {
        const msg =
          parsed.detail ?? parsed.message ?? parsed.error ?? parsed.title;
        if (typeof msg === "string" && msg.length > 0) {
          detail = msg;
        }
      }
    } catch {
      // Not JSON — keep raw detail
    }
  }

  const isTruncated = (detail?.length ?? 0) > MAX_DETAIL_LENGTH;
  if (detail && detail.length > MAX_DETAIL_LENGTH) {
    detail = detail.slice(0, MAX_DETAIL_LENGTH);
  }

  return { title, detail, statusCode, isTruncated };
}

function errorToString(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}
