export type AgentTraceThought = { type: 'thought'; content: string };
export type AgentTraceToolCall = {
  type: 'tool_call';
  id: string;
  name: string;
  args: Record<string, unknown>;
};
export type AgentTraceToolResult = {
  type: 'tool_result';
  tool_call_id: string;
  name: string;
  content: unknown;
};
export type AgentTraceAnswer = { type: 'answer'; content: string };
export type AgentTraceEntry =
  | AgentTraceThought
  | AgentTraceToolCall
  | AgentTraceToolResult
  | AgentTraceAnswer;

export type AgentSource = {
  rank: number;
  source: string;
  snippet: string;
  score?: number;
  law_id?: string;
  article?: string;
};

export type CitationValidation = {
  sub_answer_citations: string[];
  final_answer_citations: string[];
  preserved: string[];
  dropped: string[];
  invented: string[];
};

export type AgentResponse = {
  question: string;
  answer: string;
  sources: AgentSource[];
  trace: AgentTraceEntry[];
  tool_calls_used: number;
  citation_validation: CitationValidation | null;
};

export type AgentStreamEvent =
  | { type: 'metadata'; question: string }
  | { type: 'thought'; content: string }
  | { type: 'tool_call'; id: string; name: string; args: Record<string, unknown> }
  | { type: 'tool_result'; tool_call_id: string; name: string; content: unknown }
  | { type: 'chunk'; content: string }
  | {
      type: 'done';
      sources: AgentSource[];
      tool_calls_used: number;
      citation_validation: CitationValidation | null;
    }
  | { type: 'error'; message: string };
