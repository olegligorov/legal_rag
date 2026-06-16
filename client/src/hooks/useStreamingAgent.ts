import { useState, useCallback } from 'react';
import { agentStream } from '@/services/agentApi';
import type { AgentTraceEntry, AgentSource, CitationValidation } from '@/types/agent';

export type AgentStreamState = {
  answer: string;
  sources: AgentSource[];
  trace: AgentTraceEntry[];
  toolCallsUsed: number;
  citationValidation: CitationValidation | null;
  isStreaming: boolean;
  error: string | null;
  /** Live token buffer for the current LLM turn (cleared on thought/tool_call/tool_result/done) */
  currentLiveText: string;
};

const INITIAL_STATE: AgentStreamState = {
  answer: '',
  sources: [],
  trace: [],
  toolCallsUsed: 0,
  citationValidation: null,
  isStreaming: false,
  error: null,
  currentLiveText: '',
};

export function useStreamingAgent() {
  const [state, setState] = useState<AgentStreamState>(INITIAL_STATE);

  const run = useCallback(async (question: string) => {
    setState({ ...INITIAL_STATE, isStreaming: true });

    try {
      for await (const event of agentStream(question)) {
        switch (event.type) {
          case 'metadata':
            // nothing to do — question is already known
            break;

          case 'chunk':
            setState((prev) => ({
              ...prev,
              currentLiveText: prev.currentLiveText + event.content,
            }));
            break;

          case 'thought':
            // currentLiveText was a live preview of this thought — finalize it
            setState((prev) => ({
              ...prev,
              trace: [...prev.trace, { type: 'thought', content: event.content }],
              currentLiveText: '',
            }));
            break;

          case 'tool_call':
            setState((prev) => ({
              ...prev,
              trace: [
                ...prev.trace,
                { type: 'tool_call', id: event.id, name: event.name, args: event.args },
              ],
              currentLiveText: '',
            }));
            break;

          case 'tool_result':
            setState((prev) => ({
              ...prev,
              trace: [
                ...prev.trace,
                {
                  type: 'tool_result',
                  tool_call_id: event.tool_call_id,
                  name: event.name,
                  content: event.content,
                },
              ],
              currentLiveText: '',
            }));
            break;

          case 'done':
            setState((prev) => ({
              ...prev,
              // remaining currentLiveText is the final answer
              answer: prev.currentLiveText,
              currentLiveText: '',
              sources: event.sources,
              toolCallsUsed: event.tool_calls_used,
              citationValidation: event.citation_validation,
              isStreaming: false,
            }));
            break;

          case 'error':
            setState((prev) => ({
              ...prev,
              error: event.message,
              isStreaming: false,
              currentLiveText: '',
            }));
            break;
        }
      }
    } catch (err) {
      setState((prev) => ({
        ...prev,
        error: err instanceof Error ? err.message : 'Unknown error',
        isStreaming: false,
        currentLiveText: '',
      }));
    }
  }, []);

  const reset = useCallback(() => {
    setState(INITIAL_STATE);
  }, []);

  return { state, run, reset };
}
