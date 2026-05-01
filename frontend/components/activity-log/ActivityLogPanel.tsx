"use client";

import { useRef, useEffect, useState, useMemo } from "react";
import { Trash2, ArrowDown, X, CheckCircle2, AlertCircle, Info, AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useActivityLog } from "@/context/activity-log-context";
import { LogEntryRow } from "./LogEntry";
import { InFlightSection } from "./InFlightSection";
import { cn } from "@/utils/utils";

/** Summary counters by log level. */
function useLevelCounts(entries: { level: string }[]) {
  return useMemo(() => {
    const counts = { info: 0, success: 0, warn: 0, error: 0 };
    for (const e of entries) {
      if (e.level in counts) counts[e.level as keyof typeof counts]++;
    }
    return counts;
  }, [entries]);
}

export function ActivityLogPanel() {
  const { entries, inFlight, clear, isOpen, setOpen } = useActivityLog();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const prevLengthRef = useRef(entries.length);
  const counts = useLevelCounts(entries);

  const sessionStart = entries.length > 0 ? entries[0].timestamp : undefined;

  useEffect(() => {
    if (entries.length > prevLengthRef.current && autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
    prevLengthRef.current = entries.length;
  }, [entries.length, autoScroll]);

  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const nearBottom = scrollHeight - scrollTop - clientHeight < 50;
    setAutoScroll(nearBottom);
  };

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      setAutoScroll(true);
    }
  };

  return (
    <div
      className={cn(
        "shrink-0 overflow-hidden transition-all duration-300 ease-in-out border-l border-border/40",
        isOpen ? "w-[420px]" : "w-0 border-l-0"
      )}
    >
      {/* Inner wrapper keeps content at full width even while the outer collapses */}
      <div className="w-[420px] h-full flex flex-col bg-background">
        {/* ── Header ────────────────────────────────────────── */}
        <div className="flex items-center justify-between pl-4 pr-2 py-2.5 border-b border-border/60 shrink-0">
          <div className="flex items-center gap-2.5">
            <h2 className="text-[13px] font-semibold tracking-tight text-foreground/90">
              Activity
            </h2>
            {entries.length > 0 && (
              <Badge
                variant="secondary"
                className="text-[10px] px-1.5 py-0 h-[18px] font-mono bg-muted text-muted-foreground/60 border-0"
              >
                {entries.length}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-0.5">
            {entries.length > 0 && (
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 text-muted-foreground/40 hover:text-muted-foreground"
                onClick={clear}
                title="Clear log"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            )}
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-muted-foreground/40 hover:text-muted-foreground"
              onClick={() => setOpen(false)}
              title="Close panel"
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>

        {/* ── Summary bar ───────────────────────────────────── */}
        {entries.length > 0 && (
          <div className="flex items-center gap-3 px-4 py-2 border-b border-border/30 bg-muted/20 shrink-0">
            {counts.success > 0 && (
              <span className="inline-flex items-center gap-1 text-[10px] font-mono text-emerald-600 dark:text-emerald-400/70">
                <CheckCircle2 className="h-3 w-3" />
                {counts.success}
              </span>
            )}
            {counts.error > 0 && (
              <span className="inline-flex items-center gap-1 text-[10px] font-mono text-red-600 dark:text-red-400/70">
                <AlertCircle className="h-3 w-3" />
                {counts.error}
              </span>
            )}
            {counts.warn > 0 && (
              <span className="inline-flex items-center gap-1 text-[10px] font-mono text-amber-600 dark:text-amber-400/70">
                <AlertTriangle className="h-3 w-3" />
                {counts.warn}
              </span>
            )}
            {counts.info > 0 && (
              <span className="inline-flex items-center gap-1 text-[10px] font-mono text-blue-600 dark:text-blue-400/70">
                <Info className="h-3 w-3" />
                {counts.info}
              </span>
            )}
          </div>
        )}

        {/* ── In Flight section (issue 008) ─────────────────── */}
        <InFlightSection inFlight={inFlight} />

        {/* ── Log body ──────────────────────────────────────── */}
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto min-h-0"
        >
          {entries.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-2 text-muted-foreground/25">
              <div className="w-8 h-8 rounded-full border border-dashed border-muted-foreground/15 flex items-center justify-center">
                <Info className="h-3.5 w-3.5" />
              </div>
              <span className="text-xs">No activity yet</span>
            </div>
          ) : (
            entries.map((entry) => (
              <LogEntryRow
                key={entry.id}
                entry={entry}
                sessionStart={sessionStart}
              />
            ))
          )}
        </div>

        {/* Jump-to-bottom pill */}
        {!autoScroll && entries.length > 0 && (
          <div className="absolute bottom-12 left-1/2 -translate-x-1/2 z-10">
            <Button
              size="sm"
              variant="secondary"
              className="text-xs h-7 shadow-lg bg-card border border-border hover:bg-accent"
              onClick={scrollToBottom}
            >
              <ArrowDown className="h-3 w-3 mr-1" />
              New events
            </Button>
          </div>
        )}

        {/* ── Footer ────────────────────────────────────────── */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-border/60 shrink-0 bg-muted/20">
          <span className="text-[10px] text-muted-foreground/30 font-mono">
            Auto-scroll {autoScroll ? "on" : "paused"}
          </span>
          <span className="text-[10px] text-emerald-600 dark:text-emerald-400/50 flex items-center gap-1.5 font-mono">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 dark:bg-emerald-400/70 animate-pulse" />
            Live
          </span>
        </div>
      </div>
    </div>
  );
}
