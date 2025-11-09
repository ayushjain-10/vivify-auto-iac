/**
 * Chat API Service
 * Handles communication with the backend conversational agent
 */

import { ModelStep } from '../types';

const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';

export interface ChatRequest {
  message: string;
  session_id: string;
  history?: any[];
}

export const chatApi = {
  /**
   * Send message and stream response using SSE
   */
  async streamMessage(
    request: ChatRequest,
    onChunk: (chunk: ModelStep) => void
  ): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/chat/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error('No response body');
    }

    try {
      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        // Decode the chunk
        buffer += decoder.decode(value, { stream: true });
        
        // Split by newlines to get individual SSE messages
        const lines = buffer.split('\n');
        
        // Keep the last incomplete line in the buffer
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim();
            
            if (data) {
              try {
                const event = JSON.parse(data);
                onChunk(event as ModelStep);
              } catch (e) {
                console.error('Failed to parse SSE data:', data, e);
              }
            }
          }
        }
      }
      
      // Process any remaining data in buffer
      if (buffer.trim() && buffer.startsWith('data: ')) {
        const data = buffer.slice(6).trim();
        if (data) {
          try {
            const event = JSON.parse(data);
            onChunk(event as ModelStep);
          } catch (e) {
            console.error('Failed to parse final SSE data:', data, e);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  },

  /**
   * Clear chat session
   */
  async clearSession(sessionId: string): Promise<void> {
    await fetch(`${API_BASE_URL}/api/chat/sessions/${sessionId}`, {
      method: 'DELETE',
    });
  },

  /**
   * Check if agent is configured
   */
  async checkHealth(): Promise<{ configured: boolean; status: string }> {
    const response = await fetch(`${API_BASE_URL}/api/chat/health`);
    return response.json();
  },
};
