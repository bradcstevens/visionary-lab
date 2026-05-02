/**
 * API service for virtual staging endpoints
 */

// API base URL configuration with GitHub Codespaces detection
const API_PROTOCOL = process.env.NEXT_PUBLIC_API_PROTOCOL || 'http';
const API_HOSTNAME = process.env.NEXT_PUBLIC_API_HOSTNAME || 'localhost';
// For GitHub Codespaces, port is part of the hostname, so this might be empty
const API_PORT = process.env.NEXT_PUBLIC_API_PORT || '8000';

// First build temporary base URL with conditional port inclusion
let API_BASE_URL = API_PORT 
  ? `${API_PROTOCOL}://${API_HOSTNAME}:${API_PORT}/api/v1` 
  : `${API_PROTOCOL}://${API_HOSTNAME}/api/v1`;

// Override with direct API URL if provided
if (process.env.NEXT_PUBLIC_API_URL) {
  console.log(`Overriding API URL with NEXT_PUBLIC_API_URL: ${process.env.NEXT_PUBLIC_API_URL}`);
  // Ensure API URL ends with /api/v1
  API_BASE_URL = process.env.NEXT_PUBLIC_API_URL.endsWith('/api/v1') 
    ? process.env.NEXT_PUBLIC_API_URL 
    : `${process.env.NEXT_PUBLIC_API_URL}/api/v1`;
}

// Export the final configured URL
export { API_BASE_URL };

// Enable debug mode to log API requests
const API_DEBUG = process.env.NEXT_PUBLIC_DEBUG_MODE === 'true';

// Types
export interface StagingSettings {
  variations_per_room: number;
  model: string;
  quality: string;
  size: string;
}

export interface GenerationMetadata {
  // Aligned with backend ``backend.models.staging.GenerationMetadata``.
  // Issue 002 of projects-page-improvements PRD: prior shape was a
  // bogus subset (``generated_at`` / ``model_version`` /
  // ``processing_time_ms`` / ``prompt_tokens`` / ``completion_tokens``)
  // that never matched what the backend actually serializes. Issue 004
  // (per-variation Edit Prompt) needs to read ``adapted_prompt`` to
  // prefill the dialog textarea, so the shape is now corrected.
  model?: string;
  adapted_prompt?: string;
  tokens_used?: number;
  generation_time_ms?: number;
}

export interface Variation {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  image_url?: string;
  // Issue 010: derived sibling variants for the storage image. ``thumb_url``
  // is the 512px-max-edge WebP used by grids; ``md_url`` is the 1024px-max-edge
  // WebP used by lightbox previews. Both are populated by the thumbnail
  // deriver on first generation and lazy-backfilled on read for legacy
  // variations (issue 012). Either can be absent until backfill lands.
  thumb_url?: string;
  md_url?: string;
  error?: string;
  // Backend serializes ``generation_metadata`` (issue 002 fix renamed
  // the prior frontend ``metadata?`` field which never matched).
  generation_metadata?: GenerationMetadata;
  created_at: string;
  updated_at: string;
}

export interface Room {
  id: string;
  label: string;
  original_image_url: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  variations: Variation[];
  prompt_addendum?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface StagingProject {
  id: string;
  name: string;
  prompt: string;
  status: 'uploading' | 'pending' | 'processing' | 'completed' | 'failed';
  settings: StagingSettings;
  rooms: Room[];
  // Issue 002 of projects-page-improvements PRD: design_brief is editable
  // through the Project Settings sheet. Optional/null when the project
  // hasn't run an AI Design Session yet.
  design_brief?: DesignBrief | null;
  // Issue 013/014 of image-pipeline-and-project-ux-overhaul PRD.
  // Backend-derived <=240-char summary used for the collapsed prompt
  // header; refreshed by the server on every PATCH that mutates
  // ``prompt`` (PromptSummarizer, deterministic truncation fallback).
  // Optional/null when the project predates the summarizer slice.
  prompt_summary?: string | null;
  created_at?: string;
  updated_at?: string;
  total_variations?: number;
  completed_variations?: number;
}

export interface CreateProjectRequest {
  name: string;
  prompt: string;
  settings?: {
    variations_per_room?: number;
    model?: string;
    quality?: string;
    size?: string;
  };
}

// SSE event types for streaming
export type StagingStreamEventType = 
  | 'project_created' 
  | 'room_uploaded' 
  | 'room_started'
  | 'room_completed'
  | 'room_failed'
  | 'variation_started' 
  | 'variation_completed' 
  | 'variation_failed' 
  | 'variation_fallback'
  | 'project_completed' 
  | 'stream_ended'
  | 'error';

export interface StagingStreamEvent {
  type: StagingStreamEventType;
  project_id?: string;
  room_id?: string;
  variation_id?: string;
  data?: any;
  message?: string;
  error?: string;
}

export type StagingStreamEventCallback = (event: StagingStreamEvent) => void;

// Design Brief types
export type ObjectCategory =
  | "plant"
  | "tree"
  | "rock"
  | "furniture"
  | "lighting"
  | "hardscape"
  | "decor"
  | "other";

export interface ObjectEntry {
  id: string;
  name: string;
  description?: string | null;
  category: ObjectCategory;
  default_quantity: number;
  size: string;
  placement: string;
  visual_notes?: string | null;
}

export interface ImageObjectOverride {
  object_id: string;
  // Required, must be >= 0. Frontend prefills with palette default_quantity
  // before any user edit; quantity=0 OR enabled=false means "skip this
  // object in this image" (the resolver treats both as equivalent skip
  // signals). Note `null` is NOT a valid value — use `enabled=false` for
  // skip and pull palette default for "no override yet".
  quantity: number;
  // `null` means inherit palette placement. Empty string is normalised to
  // null at the model boundary (the backend's pydantic validator coerces
  // empty / whitespace strings to None), but the frontend MUST send `null`
  // explicitly — never `undefined`.
  placement: string | null;
  enabled: boolean;
}

export interface PlacementGuide {
  back_row: string;
  middle_row?: string;
  front_row?: string;
  accent_areas?: string;
}

export interface DesignBrief {
  global_instructions: string;
  object_palette: ObjectEntry[];
  placement_guide: PlacementGuide;
  per_image_notes: Record<string, string>;
  // Sparse map: room_id -> override entries. The map only contains keys
  // for rooms that have at least one override; empty room keys are pruned
  // by DesignBriefEditor when the user clears the last override.
  per_image_objects: Record<string, ImageObjectOverride[]>;
  preserve_elements: string[];
  settings: {
    variations_per_room: number;
    model: string;
    quality: string;
    size: string;
  };
}

export interface ImageAnalysisResult {
  room_id: string;
  description: string;
  features: string[];
  zones: string[];
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  focused_image_id?: string;
}

export interface ChatResponse {
  reply: string;
  ready_for_brief: boolean;
  suggested_actions: string[];
}

// Issue 004 of the per-image-object-quantities PRD. The backend returns a
// non-zero ``dropped`` count when a regenerate-brief request carried prior
// per-image overrides whose object names no longer match (or were
// ambiguous). The wizard surfaces this in a non-blocking toast.
export interface ReconcileSummary {
  carried_forward: number;
  dropped: number;
}

export interface GenerateBriefResponse {
  brief: DesignBrief;
  reconciliation_summary: ReconcileSummary;
}

// API Functions

/**
 * Create a new staging project
 */
export async function createProject(request: CreateProjectRequest): Promise<StagingProject> {
  const url = `${API_BASE_URL}/staging/projects`;
  
  if (API_DEBUG) {
    console.log(`POST ${url}`);
    console.log('Request:', request);
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to create project: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  return data.project ?? data;
}
export async function listProjects(): Promise<StagingProject[]> {
  const url = `${API_BASE_URL}/staging/projects`;
  
  if (API_DEBUG) {
    console.log(`GET ${url}`);
  }

  const response = await fetch(url);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to list projects: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  const projects = data.projects ?? data;
  if (!Array.isArray(projects)) {
    console.error('listProjects: expected array, got:', typeof projects, projects);
    return [];
  }
  return projects;
}

/**
 * Get a specific staging project
 */
export async function getProject(projectId: string): Promise<StagingProject> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}`;
  
  if (API_DEBUG) {
    console.log(`GET ${url}`);
  }

  const response = await fetch(url);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to get project: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  return data.project ?? data;
}

/**
 * Delete a staging project
 */
export async function deleteProject(projectId: string): Promise<void> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}`;
  
  if (API_DEBUG) {
    console.log(`DELETE ${url}`);
  }

  const response = await fetch(url, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to delete project: ${response.status} ${errorText}`);
  }
}

/**
 * Reset a stuck project (force-reconcile all processing/failed items back to pending)
 */
export async function resetProject(projectId: string): Promise<StagingProject> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/reset`;
  
  if (API_DEBUG) {
    console.log(`POST ${url}`);
  }

  const response = await fetch(url, { method: 'POST' });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to reset project: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  return data.project ?? data;
}

/**
 * Upload rooms to a project
 */
export async function uploadRooms(projectId: string, roomFiles: { file: File; name: string }[]): Promise<Room[]> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/rooms`;
  
  if (API_DEBUG) {
    console.log(`POST ${url}`);
    console.log('Room files:', roomFiles.length);
  }

  const formData = new FormData();
  const labels: string[] = [];
  roomFiles.forEach(({ file, name }) => {
    formData.append('images', file, file.name);
    labels.push(name);
  });
  formData.append('labels', JSON.stringify(labels));

  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to upload rooms: ${response.status} ${errorText}`);
  }

  return response.json();
}

/**
 * Stream generation progress for a project
 */
export function streamGeneration(
  projectId: string,
  onEvent: StagingStreamEventCallback
): () => void {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/generate`;
  
  if (API_DEBUG) {
    console.log(`Starting SSE stream for staging generation`);
    console.log(`POST ${url}`);
  }

  // Create AbortController for cleanup
  const abortController = new AbortController();
  let receivedTerminalEvent = false;

  // Use fetch with ReadableStream to handle SSE from POST request
  // (EventSource only supports GET requests)
  fetch(url, {
    method: 'POST',
    signal: abortController.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const errorText = await response.text();
        onEvent({ type: 'error', error: `HTTP ${response.status}: ${errorText}` });
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        onEvent({ type: 'error', error: 'No response body' });
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';
      // Persist across chunks so events split across reads aren't lost
      let currentEventType: string | null = null;
      let currentData: string | null = null;

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          if (API_DEBUG) {
            console.log('SSE stream ended');
          }
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        
        // Parse SSE events from buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            currentData = line.slice(6);
          } else if (line === '' && currentEventType && currentData) {
            // End of event, parse and dispatch
            try {
              const parsedData = JSON.parse(currentData);
              const event: StagingStreamEvent = {
                type: currentEventType as StagingStreamEventType,
                ...parsedData,
              };
              
              if (currentEventType === 'project_completed' || currentEventType === 'error') {
                receivedTerminalEvent = true;
              }

              if (API_DEBUG) {
                console.log('SSE event:', event);
              }
              
              onEvent(event);
            } catch (parseError) {
              console.error('Failed to parse SSE data:', currentData, parseError);
            }
            
            currentEventType = null;
            currentData = null;
          }
        }
      }

      // Stream ended — dispatch fallback event if no terminal event was received
      if (!receivedTerminalEvent) {
        onEvent({ type: 'stream_ended' });
      }
    })
    .catch((error) => {
      if (error.name === 'AbortError') {
        if (API_DEBUG) {
          console.log('SSE stream aborted by user');
        }
        return;
      }
      console.error('SSE stream error:', error);
      onEvent({ type: 'error', error: error.message || 'Stream error' });
    });

  // Return cleanup function
  return () => {
    if (API_DEBUG) {
      console.log('Aborting SSE stream');
    }
    abortController.abort();
  };
}

/**
 * Stream room regeneration for a specific room
 */
export function streamRoomRegeneration(
  projectId: string,
  roomId: string,
  onEvent: StagingStreamEventCallback
): () => void {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/rooms/${roomId}/regenerate`;
  
  if (API_DEBUG) {
    console.log(`Starting SSE stream for room regeneration`);
    console.log(`POST ${url}`);
  }

  // Create AbortController for cleanup
  const abortController = new AbortController();
  let receivedTerminalEvent = false;

  // Use fetch with ReadableStream to handle SSE from POST request
  fetch(url, {
    method: 'POST',
    signal: abortController.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const errorText = await response.text();
        onEvent({ type: 'error', error: `HTTP ${response.status}: ${errorText}` });
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        onEvent({ type: 'error', error: 'No response body' });
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';
      // Persist across chunks so events split across reads aren't lost
      let currentEventType: string | null = null;
      let currentData: string | null = null;

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          if (API_DEBUG) {
            console.log('SSE stream ended');
          }
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        
        // Parse SSE events from buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            currentData = line.slice(6);
          } else if (line === '' && currentEventType && currentData) {
            // End of event, parse and dispatch
            try {
              const parsedData = JSON.parse(currentData);
              const event: StagingStreamEvent = {
                type: currentEventType as StagingStreamEventType,
                ...parsedData,
              };

              if (currentEventType === 'project_completed' || currentEventType === 'error') {
                receivedTerminalEvent = true;
              }

              if (API_DEBUG) {
                console.log('SSE event:', event);
              }
              
              onEvent(event);
            } catch (parseError) {
              console.error('Failed to parse SSE data:', currentData, parseError);
            }
            
            currentEventType = null;
            currentData = null;
          }
        }
      }

      // Stream ended — dispatch fallback event if no terminal event was received
      if (!receivedTerminalEvent) {
        onEvent({ type: 'stream_ended' });
      }
    })
    .catch((error) => {
      if (error.name === 'AbortError') {
        if (API_DEBUG) {
          console.log('SSE stream aborted by user');
        }
        return;
      }
      console.error('SSE stream error:', error);
      onEvent({ type: 'error', error: error.message || 'Stream error' });
    });

  // Return cleanup function
  return () => {
    if (API_DEBUG) {
      console.log('Aborting SSE stream');
    }
    abortController.abort();
  };
}

/**
 * Stream single variation regeneration
 */
export function streamVariationRegeneration(
  projectId: string,
  roomId: string,
  variationId: string,
  strategy: 'retry' | 'fresh',
  onEvent: StagingStreamEventCallback,
): () => void {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/rooms/${roomId}/variations/${variationId}/regenerate?strategy=${strategy}`;
  
  if (API_DEBUG) {
    console.log(`Starting SSE stream for variation regeneration (${strategy})`);
    console.log(`POST ${url}`);
  }

  const abortController = new AbortController();
  let receivedTerminalEvent = false;

  fetch(url, {
    method: 'POST',
    signal: abortController.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const errorText = await response.text();
        onEvent({ type: 'error', error: `HTTP ${response.status}: ${errorText}` });
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        onEvent({ type: 'error', error: 'No response body' });
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let currentEventType: string | null = null;
      let currentData: string | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            currentData = line.slice(6);
          } else if (line === '' && currentEventType && currentData) {
            try {
              const parsedData = JSON.parse(currentData);
              const event: StagingStreamEvent = {
                type: currentEventType as StagingStreamEventType,
                ...parsedData,
              };

              if (currentEventType === 'project_completed' || currentEventType === 'error') {
                receivedTerminalEvent = true;
              }

              if (API_DEBUG) {
                console.log('SSE event:', event);
              }

              onEvent(event);
            } catch (parseError) {
              console.error('Failed to parse SSE data:', currentData, parseError);
            }

            currentEventType = null;
            currentData = null;
          }
        }
      }

      if (!receivedTerminalEvent) {
        onEvent({ type: 'stream_ended' });
      }
    })
    .catch((error) => {
      if (error.name === 'AbortError') return;
      console.error('SSE stream error:', error);
      onEvent({ type: 'error', error: error.message || 'Stream error' });
    });

  return () => {
    abortController.abort();
  };
}

/**
 * Stream a per-variation Edit Prompt request — issue 004 of the
 * projects-page-improvements PRD.
 *
 * Posts to ``POST /projects/{pid}/rooms/{rid}/variations/{vid}/edit-prompt``
 * with a JSON body of ``{adapted_prompt: string}``. The backend appends
 * a fresh variation generated from the user-supplied prompt — the
 * source variation identified by ``variationId`` is preserved
 * untouched (the whole point of Edit Prompt vs Try Something New).
 *
 * Mirrors the SSE stream / abort / terminal-event accounting of
 * ``streamVariationRegeneration``. The only differences are the URL
 * suffix (``/edit-prompt`` vs ``/regenerate``) and the JSON body
 * (Edit Prompt sends the user-typed text instead of a query-string
 * strategy).
 */
export function streamVariationEditPrompt(
  projectId: string,
  roomId: string,
  variationId: string,
  adaptedPrompt: string,
  onEvent: StagingStreamEventCallback,
): () => void {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/rooms/${roomId}/variations/${variationId}/edit-prompt`;

  if (API_DEBUG) {
    console.log(`Starting SSE stream for variation edit-prompt`);
    console.log(`POST ${url}`);
  }

  const abortController = new AbortController();
  let receivedTerminalEvent = false;

  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ adapted_prompt: adaptedPrompt }),
    signal: abortController.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const errorText = await response.text();
        onEvent({ type: 'error', error: `HTTP ${response.status}: ${errorText}` });
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        onEvent({ type: 'error', error: 'No response body' });
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let currentEventType: string | null = null;
      let currentData: string | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            currentData = line.slice(6);
          } else if (line === '' && currentEventType && currentData) {
            try {
              const parsedData = JSON.parse(currentData);
              const event: StagingStreamEvent = {
                type: currentEventType as StagingStreamEventType,
                ...parsedData,
              };

              if (currentEventType === 'project_completed' || currentEventType === 'error') {
                receivedTerminalEvent = true;
              }

              if (API_DEBUG) {
                console.log('SSE event:', event);
              }

              onEvent(event);
            } catch (parseError) {
              console.error('Failed to parse SSE data:', currentData, parseError);
            }

            currentEventType = null;
            currentData = null;
          }
        }
      }

      if (!receivedTerminalEvent) {
        onEvent({ type: 'stream_ended' });
      }
    })
    .catch((error) => {
      if (error.name === 'AbortError') return;
      console.error('SSE stream error:', error);
      onEvent({ type: 'error', error: error.message || 'Stream error' });
    });

  return () => {
    abortController.abort();
  };
}

export async function analyzeImages(projectId: string): Promise<ImageAnalysisResult[]> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/analyze`;
  const response = await fetch(url, { method: 'POST' });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to analyze images: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return data.analyses ?? [];
}

export async function chatWithProject(
  projectId: string,
  message: string,
  conversationHistory: ChatMessage[],
  focusedImageId?: string,
): Promise<ChatResponse> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/chat`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      conversation_history: conversationHistory,
      focused_image_id: focusedImageId ?? null,
    }),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Chat failed: ${response.status} ${errorText}`);
  }
  return response.json();
}

export async function generateBrief(
  projectId: string,
  conversationHistory: ChatMessage[],
  previousBrief?: DesignBrief,
): Promise<GenerateBriefResponse> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/brief`;
  const body: {
    conversation_history: ChatMessage[];
    previous_brief?: DesignBrief;
  } = { conversation_history: conversationHistory };
  if (previousBrief) {
    // Issue 004 of the per-image-object-quantities PRD: pass the current
    // brief so per-image quantity / placement / skip overrides can be
    // carried forward by case-insensitive whitespace-trimmed name match
    // against the regenerated palette.
    body.previous_brief = previousBrief;
  }
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to generate brief: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return {
    brief: data.brief,
    reconciliation_summary: data.reconciliation_summary ?? {
      carried_forward: 0,
      dropped: 0,
    },
  };
}

export async function updateBrief(projectId: string, brief: DesignBrief): Promise<DesignBrief> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/brief`;
  const response = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(brief),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to update brief: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return data.brief;
}

export async function updateRoomAddendum(
  projectId: string,
  roomId: string,
  promptAddendum: string | null,
): Promise<StagingProject> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/rooms/${roomId}`;
  const response = await fetch(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt_addendum: promptAddendum }),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to update room addendum: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return data.project;
}

// Issue 004 of project-settings-completeness PRD. Generic
// PATCH /projects/{id}/rooms/{rid} body — both fields are optional
// and handled __fields_set__-aware on the backend (a label-only PATCH
// does NOT silently clear an existing addendum, and vice versa). The
// backend trims `label` and rejects empty / whitespace-only / null
// `label` with a 422.
//
// `updateRoomAddendum` above is kept as a backwards-compatible
// convenience for the existing per-room addendum popover in
// `RoomGroup.tsx`. New call sites that need a label-only or
// label+addendum update should call `updateRoom` directly.
export interface UpdateRoomBody {
  label?: string;
  prompt_addendum?: string | null;
}

export async function updateRoom(
  projectId: string,
  roomId: string,
  body: UpdateRoomBody,
): Promise<StagingProject> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/rooms/${roomId}`;
  const response = await fetch(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to update room: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return data.project;
}

// Issue 005 of project-settings-completeness PRD. DELETE the target
// room from a project. The backend cascades the delete:
//   - Removes the room from `project.rooms`.
//   - Prunes room-keyed metadata in `analyses` and `design_brief.
//     per_image_notes` / `per_image_objects` (rubber-duck blocker —
//     without this, stale references leak into future brief/regen
//     flows).
//   - Best-effort blob cleanup for the originals + the
//     `staging/{project_id}/variations/{room_id}/` prefix sweep.
//   - Returns 409 Conflict if `project.status === 'processing'` —
//     callers should prevent the click in that state, but the backend
//     guards programmatic / racing clients.
//
// On success, the response body is `{project}` with the updated full
// project (rooms minus the deleted one). Callers should pass it
// through `onProjectUpdate(updated)` so the page's `setProject` runs
// the existing `resolveImageUrls` pass first (preserves SAS tokens
// on in-place URLs).
//
// On failure (network, 4xx, 5xx), throws an Error whose message
// includes the response body — the rooms manager surfaces this as
// an INLINE error on the confirm row (NOT a toast that auto-
// dismisses) so the user sees the failure state and can retry or
// cancel. This matches the PRD's "the confirm row stays visible
// with an inline error and the room row is preserved" rule.
export async function removeRoom(
  projectId: string,
  roomId: string,
): Promise<StagingProject> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/rooms/${roomId}`;
  const response = await fetch(url, {
    method: 'DELETE',
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to remove room: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return data.project;
}

// Issue 002 of the projects-page-improvements PRD. Each field is
// optional — omit a field to leave it unchanged on the server. Sending
// ``design_brief: null`` explicitly clears the brief; sending null for
// the other three fields is a 422.
export interface UpdateProjectBody {
  name?: string;
  prompt?: string;
  // Partial settings — only the keys you supply are MERGED onto the
  // persisted settings (the backend preserves keys you don't send).
  settings?: Partial<StagingSettings>;
  design_brief?: DesignBrief | null;
}

export async function updateProject(
  projectId: string,
  updates: UpdateProjectBody,
): Promise<StagingProject> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}`;
  const response = await fetch(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to update project: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return data.project;
}
