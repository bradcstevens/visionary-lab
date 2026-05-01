import { useCallback, useEffect, useRef, useState } from 'react';
import type { Room, StagingProject } from '@/services/stagingApi';

export type EnqueueOutcome = 'dispatched' | 'queued' | 'deduped';

export type RetryDispatch = (
  room: Room,
  variationIndex: number,
  strategy: 'retry' | 'fresh',
) => void;

export interface UseRetryQueueParams {
  project: StagingProject | null;
  isGenerating: boolean;
  regeneratingVariationId: string | null;
  onDispatch: RetryDispatch;
  onDrop?: (variationId: string) => void;
}

export interface UseRetryQueueResult {
  enqueue: (variationId: string) => EnqueueOutcome;
  clear: () => void;
  queuedIds: ReadonlySet<string>;
}

interface VariationLocation {
  room: Room;
  variationIndex: number;
}

function findFailedVariation(
  project: StagingProject | null,
  variationId: string,
): VariationLocation | null {
  if (!project) return null;
  for (const room of project.rooms) {
    const variationIndex = room.variations.findIndex((v) => v.id === variationId);
    if (variationIndex === -1) continue;
    if (room.variations[variationIndex].status !== 'failed') return null;
    return { room, variationIndex };
  }
  return null;
}

export function useRetryQueue(params: UseRetryQueueParams): UseRetryQueueResult {
  const { project, isGenerating, regeneratingVariationId, onDispatch, onDrop } = params;

  const projectRef = useRef(project);
  const onDispatchRef = useRef(onDispatch);
  const onDropRef = useRef(onDrop);
  useEffect(() => {
    projectRef.current = project;
  }, [project]);
  useEffect(() => {
    onDispatchRef.current = onDispatch;
  }, [onDispatch]);
  useEffect(() => {
    onDropRef.current = onDrop;
  }, [onDrop]);

  const queueRef = useRef<string[]>([]);
  const inFlightRef = useRef<string | null>(null);
  const [queuedIds, setQueuedIds] = useState<ReadonlySet<string>>(new Set());

  const syncQueuedIds = useCallback(() => {
    setQueuedIds(new Set(queueRef.current));
  }, []);

  const dispatchSync = useCallback(
    (variationId: string): boolean => {
      const location = findFailedVariation(projectRef.current, variationId);
      if (!location) return false;
      inFlightRef.current = variationId;
      onDispatchRef.current(location.room, location.variationIndex, 'fresh');
      return true;
    },
    [],
  );

  const drain = useCallback(() => {
    if (queueRef.current.length === 0) return;
    while (queueRef.current.length > 0) {
      const next = queueRef.current[0];
      queueRef.current = queueRef.current.slice(1);
      const location = findFailedVariation(projectRef.current, next);
      if (!location) {
        onDropRef.current?.(next);
        continue;
      }
      inFlightRef.current = next;
      onDispatchRef.current(location.room, location.variationIndex, 'fresh');
      break;
    }
    syncQueuedIds();
  }, [syncQueuedIds]);

  useEffect(() => {
    if (isGenerating || regeneratingVariationId !== null) return;
    inFlightRef.current = null;
    drain();
  }, [isGenerating, regeneratingVariationId, drain]);

  const enqueue = useCallback(
    (variationId: string): EnqueueOutcome => {
      if (
        inFlightRef.current === variationId ||
        queueRef.current.includes(variationId)
      ) {
        return 'deduped';
      }

      const busy =
        isGenerating ||
        regeneratingVariationId !== null ||
        inFlightRef.current !== null;

      if (busy) {
        queueRef.current = [...queueRef.current, variationId];
        syncQueuedIds();
        return 'queued';
      }

      dispatchSync(variationId);
      return 'dispatched';
    },
    [isGenerating, regeneratingVariationId, dispatchSync, syncQueuedIds],
  );

  const clear = useCallback(() => {
    queueRef.current = [];
    syncQueuedIds();
  }, [syncQueuedIds]);

  return { enqueue, clear, queuedIds };
}
