/**
 * Shared type definitions for the RAG application
 */

/**
 * Source document returned from the RAG pipeline
 */
export interface Source {
  rank?: number;
  source: string;
  snippet: string;
  score?: number;
}

/**
 * Message in the chat conversation
 */
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  /** Present on agent-mode messages */
  agentMeta?: {
    trace: import('@/types/agent').AgentTraceEntry[];
    toolCallsUsed: number;
    citationValidation: import('@/types/agent').CitationValidation | null;
    isStreaming: boolean;
  };
}

/**
 * Query request to the RAG API
 */
export interface QueryRequest {
  question: string;
  top_n?: number;
}

/**
 * Query response from the RAG API (non-streaming)
 */
export interface QueryResponse {
  question: string;
  answer: string;
  sources: Source[];
}

/**
 * Stream event types
 */
export interface StreamMetadata {
  type: 'metadata';
  sources: Source[];
  question: string;
}

export interface StreamChunk {
  type: 'chunk';
  content: string;
}

export interface StreamDone {
  type: 'done';
}

export interface StreamError {
  type: 'error';
  message: string;
}

export type StreamEvent = StreamMetadata | StreamChunk | StreamDone | StreamError;
