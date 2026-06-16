import { useState, useCallback } from 'react';
import { queryRAGStream } from '@/services/api';
import { agentStream } from '@/services/agentApi';
import type { Message, Source } from '@/types';
import type { AgentSource, AgentTraceEntry, CitationValidation } from '@/types/agent';

export type ChatMode = 'quick' | 'agent';

function toSource(s: AgentSource): Source {
  return { rank: s.rank, source: s.source, snippet: s.snippet, score: s.score };
}

export function useChat(mode: ChatMode = 'quick') {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = useCallback(
    async (question: string) => {
      const userMessage: Message = {
        id: Date.now().toString(),
        role: 'user',
        content: question,
      };

      setMessages((prev) => [...prev, userMessage]);
      setInput('');
      setIsLoading(true);

      const assistantMessageId = (Date.now() + 1).toString();

      if (mode === 'agent') {
        await sendAgentMessage(question, assistantMessageId);
      } else {
        await sendQuickMessage(question, assistantMessageId);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [mode],
  );

  async function sendQuickMessage(question: string, assistantMessageId: string) {
    let assistantContent = '';
    let sources: Source[] = [];

    try {
      for await (const event of queryRAGStream({ question })) {
        switch (event.type) {
          case 'metadata':
            sources = event.sources;
            setMessages((prev) => [
              ...prev,
              { id: assistantMessageId, role: 'assistant', content: '', sources },
            ]);
            break;

          case 'chunk':
            assistantContent += event.content;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId ? { ...m, content: assistantContent } : m,
              ),
            );
            break;

          case 'done':
            setIsLoading(false);
            break;

          case 'error':
            setIsLoading(false);
            setMessages((prev) => upsertError(prev, assistantMessageId, event.message));
            break;
        }
      }
    } catch (error) {
      setIsLoading(false);
      setMessages((prev) =>
        upsertError(prev, assistantMessageId, 'Failed to process query. Please try again.'),
      );
      console.error('Query error:', error);
    }
  }

  async function sendAgentMessage(question: string, assistantMessageId: string) {
    // Create placeholder immediately
    setMessages((prev) => [
      ...prev,
      {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        sources: [],
        agentMeta: {
          trace: [],
          toolCallsUsed: 0,
          citationValidation: null,
          isStreaming: true,
        },
      },
    ]);

    let liveText = '';
    const trace: AgentTraceEntry[] = [];

    const update = (patch: Partial<Message>) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantMessageId ? { ...m, ...patch } : m)),
      );

    const updateMeta = (
      metaPatch: Partial<{
        trace: AgentTraceEntry[];
        toolCallsUsed: number;
        citationValidation: CitationValidation | null;
        isStreaming: boolean;
      }>,
    ) =>
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessageId && m.agentMeta
            ? { ...m, agentMeta: { ...m.agentMeta, ...metaPatch } }
            : m,
        ),
      );

    try {
      for await (const event of agentStream(question)) {
        switch (event.type) {
          case 'metadata':
            break;

          case 'chunk':
            liveText += event.content;
            update({ content: liveText });
            break;

          case 'thought':
            trace.push({ type: 'thought', content: event.content });
            liveText = '';
            update({ content: '' });
            updateMeta({ trace: [...trace] });
            break;

          case 'tool_call':
            trace.push({ type: 'tool_call', id: event.id, name: event.name, args: event.args });
            liveText = '';
            update({ content: '' });
            updateMeta({ trace: [...trace] });
            break;

          case 'tool_result':
            trace.push({
              type: 'tool_result',
              tool_call_id: event.tool_call_id,
              name: event.name,
              content: event.content,
            });
            liveText = '';
            updateMeta({ trace: [...trace] });
            break;

          case 'done': {
            const sources = event.sources.map(toSource);
            update({ content: liveText, sources });
            updateMeta({
              trace: [...trace],
              toolCallsUsed: event.tool_calls_used,
              citationValidation: event.citation_validation,
              isStreaming: false,
            });
            setIsLoading(false);
            break;
          }

          case 'error':
            updateMeta({ isStreaming: false });
            setMessages((prev) => upsertError(prev, assistantMessageId, event.message));
            setIsLoading(false);
            break;
        }
      }
    } catch (error) {
      updateMeta({ isStreaming: false });
      setMessages((prev) =>
        upsertError(prev, assistantMessageId, 'Failed to process query. Please try again.'),
      );
      setIsLoading(false);
      console.error('Agent error:', error);
    }
  }

  return { messages, input, setInput, isLoading, sendMessage };
}

function upsertError(messages: Message[], id: string, text: string): Message[] {
  const exists = messages.find((m) => m.id === id);
  const errorMsg = `Error: ${text}`;
  if (exists) {
    return messages.map((m) => (m.id === id ? { ...m, content: errorMsg } : m));
  }
  return [...messages, { id, role: 'assistant', content: errorMsg }];
}
