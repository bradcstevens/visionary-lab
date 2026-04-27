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
  style: string;
  room_count: number;
  variations_per_room: number;
  output_format?: string;
  quality?: string;
}

export interface GenerationMetadata {
  generated_at: string;
  model_version: string;
  processing_time_ms: number;
  prompt_tokens?: number;
  completion_tokens?: number;
}

export interface Variation {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  image_url?: string;
  error?: string;
  metadata?: GenerationMetadata;
  created_at: string;
  updated_at: string;
}

export interface Room {
  id: string;
  label: string;
  original_image_url: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  variations: Variation[];
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
export interface PlantEntry {
  species: string;
  botanical_name?: string;
  quantity: number;
  size: string;
  placement: string;
  visual_notes?: string;
}

export interface PlacementGuide {
  back_row: string;
  middle_row?: string;
  front_row?: string;
  accent_areas?: string;
}

export interface DesignBrief {
  global_instructions: string;
  plant_palette: PlantEntry[];
  placement_guide: PlacementGuide;
  per_image_notes: Record<string, string>;
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

        let currentEventType: string | null = null;
        let currentData: string | null = null;

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

        let currentEventType: string | null = null;
        let currentData: string | null = null;

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

export async function generateBrief(projectId: string, conversationHistory: ChatMessage[]): Promise<DesignBrief> {
  const url = `${API_BASE_URL}/staging/projects/${projectId}/brief`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversation_history: conversationHistory }),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to generate brief: ${response.status} ${errorText}`);
  }
  const data = await response.json();
  return data.brief;
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
