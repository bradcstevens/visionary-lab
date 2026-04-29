"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/utils/utils";
import type { LogEntry as LogEntryType } from "@/context/activity-log-context";

interface LogEntryProps {
  entry: LogEntryType;
  /** Timestamp of the very first entry in the session — used for relative offsets. */
  sessionStart?: Date;
}

const levelConfig: Record<
  LogEntryType["level"],
  { bar: string; text: string; bg: string }
> = {
  info: {
    bar: "bg-blue-500 dark:bg-blue-400",
    text: "text-blue-700 dark:text-blue-300",
    bg: "hover:bg-blue-50 dark:hover:bg-blue-400/[0.04]",
  },
  success: {
    bar: "bg-emerald-500 dark:bg-emerald-400",
    text: "text-emerald-700 dark:text-emerald-300",
    bg: "hover:bg-emerald-50 dark:hover:bg-emerald-400/[0.04]",
  },
  error: {
    bar: "bg-red-500 dark:bg-red-400",
    text: "text-red-700 dark:text-red-300",
    bg: "hover:bg-red-50 dark:hover:bg-red-400/[0.04]",
  },
  warn: {
    bar: "bg-amber-500 dark:bg-amber-400",
    text: "text-amber-700 dark:text-amber-300",
    bg: "hover:bg-amber-50 dark:hover:bg-amber-400/[0.04]",
  },
};

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatRelative(entry: Date, start: Date): string {
  const delta = Math.max(0, Math.round((entry.getTime() - start.getTime()) / 1000));
  if (delta < 60) return `+${delta}s`;
  const m = Math.floor(delta / 60);
  const s = delta % 60;
  return `+${m}m${s > 0 ? ` ${s}s` : ""}`;
}

/** Parse a detail string containing ` · ` separated chips into structured tags. */
function parseDetailChips(detail: string): string[] {
  return detail.split(" · ").map((s) => s.trim()).filter(Boolean);
}

export function LogEntryRow({ entry, sessionStart }: LogEntryProps) {
  const [expanded, setExpanded] = useState(false);
  const cfg = levelConfig[entry.level];
  const hasLongDetail = (entry.detail?.length ?? 0) > 80;
  const chips = entry.detail ? parseDetailChips(entry.detail) : [];
  const isStructured = chips.length > 1;

  return (
    <div
      className={cn(
        "group relative flex gap-0 border-b border-border/40 transition-colors",
        cfg.bg
      )}
    >
      {/* Level indicator bar */}
      <div className={cn("w-[3px] shrink-0 rounded-full my-1.5 ml-2", cfg.bar)} />

      {/* Body */}
      <div className="flex-1 min-w-0 px-3 py-2.5">
        {/* Top row: icon + message + timestamp */}
        <div className="flex items-start justify-between gap-2">
          <div className={cn("text-[12px] font-medium leading-snug", cfg.text)}>
            {entry.icon && <span className="mr-1.5">{entry.icon}</span>}
            {entry.message}
          </div>
          <div className="shrink-0 flex items-center gap-1.5 text-[10px] text-muted-foreground/40 font-mono tabular-nums pt-px">
            {sessionStart && (
              <span className="text-muted-foreground/25">
                {formatRelative(entry.timestamp, sessionStart)}
              </span>
            )}
            <span>{formatTime(entry.timestamp)}</span>
          </div>
        </div>

        {/* Detail row — structured chips or plain text */}
        {entry.detail && (
          <div className="mt-1.5">
            {isStructured && !hasLongDetail ? (
              <div className="flex flex-wrap items-center gap-1.5">
                {chips.map((chip, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono bg-muted text-muted-foreground/70 border border-border/50"
                  >
                    {chip}
                  </span>
                ))}
              </div>
            ) : hasLongDetail ? (
              <div>
                <div
                  className={cn(
                    "text-[11px] text-muted-foreground/50 font-mono leading-relaxed break-words",
                    !expanded && "line-clamp-2"
                  )}
                >
                  {entry.detail}
                </div>
                <button
                  onClick={() => setExpanded(!expanded)}
                  className="mt-1 inline-flex items-center gap-0.5 text-[10px] text-muted-foreground/40 hover:text-muted-foreground/70 transition-colors cursor-pointer"
                >
                  <ChevronDown
                    className={cn(
                      "h-3 w-3 transition-transform",
                      expanded && "rotate-180"
                    )}
                  />
                  {expanded ? "Show less" : "Show more"}
                </button>
              </div>
            ) : (
              <div className="text-[11px] text-muted-foreground/50 font-mono leading-relaxed">
                {entry.detail}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
