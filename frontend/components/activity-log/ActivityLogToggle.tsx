"use client";

import { Terminal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useActivityLog } from "@/context/activity-log-context";

export function ActivityLogToggle() {
  const { isOpen, setOpen, hasActivity, entries } = useActivityLog();

  if (!hasActivity) return null;

  return (
    <Button
      variant="ghost"
      size="icon"
      className="relative h-8 w-8"
      onClick={() => setOpen(!isOpen)}
      title={isOpen ? "Hide activity log" : "Show activity log"}
    >
      <Terminal className="h-4 w-4" />
      {!isOpen && entries.length > 0 && (
        <span className="absolute -top-0.5 -right-0.5 h-2.5 w-2.5 rounded-full bg-blue-500 border-2 border-background" />
      )}
    </Button>
  );
}
