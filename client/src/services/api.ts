import axios from 'axios';
import type {
  QueryRequest,
  QueryResponse,
  StreamEvent,
  StreamMetadata,
  StreamChunk,
  StreamError,
} from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Health check endpoint
 */
export const healthCheck = async () => {
  const response = await apiClient.get('/api/health');
  return response.data;
};

/**
 * Regular query endpoint (non-streaming)
 */
export const queryRAG = async (request: QueryRequest): Promise<QueryResponse> => {
  const response = await apiClient.post<QueryResponse>('/api/query', request);
  return response.data;
};

/**
 * Streaming query endpoint
 *
 * Returns an async generator that yields stream events as they arrive.
 *
 * @example
 * ```ts
 * const stream = queryRAGStream({ question: "What is a Pod?" })
 * for await (const event of stream) {
 *   if (event.type === 'metadata') {
 *     console.log('Sources:', event.sources)
 *   } else if (event.type === 'chunk') {
 *     console.log('Chunk:', event.content)
 *   }
 * }
 * ```
 */
export async function* queryRAGStream(request: QueryRequest): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_BASE_URL}/api/query/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('Response body is not readable');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE format: "data: {...}\n\n"
      const lines = buffer.split('\n\n');

      // Keep the last incomplete line in buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6)) as StreamEvent;
          yield data;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Hook-friendly version that returns a callback function
 *
 * @example
 * ```tsx
 * const [answer, setAnswer] = useState('')
 * const [sources, setSources] = useState<Source[]>([])
 *
 * const handleStream = async () => {
 *   for await (const event of queryRAGStream({ question: "What is a Pod?" })) {
 *     if (event.type === 'metadata') {
 *       setSources(event.sources)
 *     } else if (event.type === 'chunk') {
 *       setAnswer(prev => prev + event.content)
 *     }
 *   }
 * }
 * ```
 */
export const useStreamQuery = () => {
  return async (
    request: QueryRequest,
    callbacks: {
      onMetadata?: (metadata: StreamMetadata) => void;
      onChunk?: (chunk: StreamChunk) => void;
      onDone?: () => void;
      onError?: (error: StreamError) => void;
    },
  ) => {
    try {
      for await (const event of queryRAGStream(request)) {
        switch (event.type) {
          case 'metadata':
            callbacks.onMetadata?.(event);
            break;
          case 'chunk':
            callbacks.onChunk?.(event);
            break;
          case 'done':
            callbacks.onDone?.();
            break;
          case 'error':
            callbacks.onError?.(event);
            break;
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      throw error;
    }
  };
};
