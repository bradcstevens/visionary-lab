"use client";

import { useEffect, useState } from "react";
import { Terminal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useActivityLog } from "@/context/activity-log-context";
import { cn } from "@/utils/utils";

export function ActivityLogToggle() {
  const { isOpen, setOpen, entries } = useActivityLog();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const errorCount = entries.filter((e) => e.level === "error").length;

  if (!mounted) {
    return null;
  }

  return (
    <Button
      variant="ghost"
      size="sm"
      className={cn(
        "relative h-8 gap-1.5 px-2.5 text-xs font-mono",
        isOpen && "bg-accent"
      )}
      onClick={() => setOpen(!isOpen)}
      title={isOpen ? "Hide activity log" : "Show activity log"}
    >
      <Terminal className="h-3.5 w-3.5" />
      {entries.length > 0 && (
        <span className="tabular-nums text-muted-foreground">{entries.length}</span>
      )}
      {!isOpen && errorCount > 0 && (
        <span className="absolute -top-0.5 -right-0.5 h-2.5 w-2.5 rounded-full bg-red-500 border-2 border-background" />
      )}
      {!isOpen && errorCount === 0 && entries.length > 0 && (
        <span className="absolute -top-0.5 -right-0.5 h-2.5 w-2.5 rounded-full bg-blue-500 border-2 border-background" />
      )}
    </Button>
  );
}
