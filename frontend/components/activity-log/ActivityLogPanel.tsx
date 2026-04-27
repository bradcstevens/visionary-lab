"use client";

import { useRef, useEffect, useState } from "react";
import { Trash2, ArrowDown } from "lucide-react";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useActivityLog } from "@/context/activity-log-context";
import { LogEntryRow } from "./LogEntry";

export function ActivityLogPanel() {
  const { entries, clear, isOpen, setOpen } = useActivityLog();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const prevLengthRef = useRef(entries.length);

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
    <Sheet open={isOpen} onOpenChange={setOpen}>
      <SheetContent
        side="right"
        className="w-[380px] sm:max-w-[380px] p-0 flex flex-col bg-[#0d1117] border-l border-border/50"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 shrink-0">
          <div className="flex items-center gap-2">
            <SheetTitle className="text-sm font-semibold text-foreground">Activity Log</SheetTitle>
            {entries.length > 0 && (
              <Badge variant="secondary" className="text-[10px] px-1.5 py-0 h-5">
                {entries.length}
              </Badge>
            )}
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-muted-foreground hover:text-foreground"
            onClick={clear}
            title="Clear log"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* Log body */}
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto min-h-0"
        >
          {entries.length === 0 ? (
            <div className="flex items-center justify-center h-full text-muted-foreground/40 text-xs font-mono">
              No activity yet
            </div>
          ) : (
            entries.map((entry) => <LogEntryRow key={entry.id} entry={entry} />)
          )}
        </div>

        {/* New events button when auto-scroll is paused */}
        {!autoScroll && entries.length > 0 && (
          <div className="absolute bottom-10 left-1/2 -translate-x-1/2">
            <Button
              size="sm"
              variant="secondary"
              className="text-xs h-7 shadow-lg"
              onClick={scrollToBottom}
            >
              <ArrowDown className="h-3 w-3 mr-1" />
              New events
            </Button>
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-white/10 shrink-0">
          <span className="text-[10px] text-muted-foreground/40 font-mono">
            Auto-scroll: {autoScroll ? "on" : "paused"}
          </span>
          <span className="text-[10px] text-green-400/60 flex items-center gap-1 font-mono">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-green-400" />
            Ready
          </span>
        </div>
      </SheetContent>
    </Sheet>
  );
}
