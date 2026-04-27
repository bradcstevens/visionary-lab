"use client";

import { createContext, useContext, useRef, useState, useCallback, type ReactNode } from "react";

export interface LogEntry {
  id: string;
  timestamp: Date;
  level: "info" | "success" | "error" | "warn";
  message: string;
  detail?: string;
  icon?: string;
}

interface ActivityLogContextValue {
  entries: LogEntry[];
  log: (entry: Omit<LogEntry, "id" | "timestamp">) => void;
  clear: () => void;
  isOpen: boolean;
  setOpen: (open: boolean) => void;
  hasActivity: boolean;
}

const MAX_ENTRIES = 500;

const ActivityLogContext = createContext<ActivityLogContextValue | null>(null);

export function ActivityLogProvider({ children }: { children: ReactNode }) {
  const entriesRef = useRef<LogEntry[]>([]);
  const [revision, setRevision] = useState(0);
  const [isOpen, setIsOpen] = useState(false);

  const log = useCallback((entry: Omit<LogEntry, "id" | "timestamp">) => {
    const newEntry: LogEntry = {
      ...entry,
      id: crypto.randomUUID(),
      timestamp: new Date(),
    };
    entriesRef.current = [...entriesRef.current, newEntry].slice(-MAX_ENTRIES);

    // Auto-open on first entry
    if (entriesRef.current.length === 1) {
      setIsOpen(true);
    }

    setRevision((r) => r + 1);
  }, []);

  const clear = useCallback(() => {
    entriesRef.current = [];
    setRevision((r) => r + 1);
  }, []);

  const value: ActivityLogContextValue = {
    entries: entriesRef.current,
    log,
    clear,
    isOpen,
    setOpen: setIsOpen,
    hasActivity: entriesRef.current.length > 0,
  };

  return (
    <ActivityLogContext.Provider value={value}>
      {children}
    </ActivityLogContext.Provider>
  );
}

export function useActivityLog(): ActivityLogContextValue {
  const context = useContext(ActivityLogContext);
  if (!context) {
    return {
      entries: [],
      log: () => {},
      clear: () => {},
      isOpen: false,
      setOpen: () => {},
      hasActivity: false,
    };
  }
  return context;
}
