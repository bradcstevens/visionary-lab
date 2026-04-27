"use client";

import { cn } from "@/utils/utils";
import type { LogEntry as LogEntryType } from "@/context/activity-log-context";

interface LogEntryProps {
  entry: LogEntryType;
}

const levelColors: Record<LogEntryType["level"], string> = {
  info: "text-blue-400",
  success: "text-green-400",
  error: "text-red-400",
  warn: "text-amber-400",
};

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function LogEntryRow({ entry }: LogEntryProps) {
  return (
    <div className="px-3 py-1.5 border-b border-white/5 font-mono text-[11px] leading-relaxed">
      <div className="text-muted-foreground/50">{formatTime(entry.timestamp)}</div>
      <div className={cn(levelColors[entry.level])}>
        {entry.icon && <span className="mr-1">{entry.icon}</span>}
        {entry.message}
      </div>
      {entry.detail && (
        <div className="text-muted-foreground/40 text-[10px]">{entry.detail}</div>
      )}
    </div>
  );
}
