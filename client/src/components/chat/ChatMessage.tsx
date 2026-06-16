import { Message, MessageContent, MessageResponse } from '@/components/ai-elements/message';
import { Sources, SourcesTrigger, SourcesContent, Source as SourceElement } from '@/components/ai-elements/sources';
import { AgentTraceAccordion } from './AgentTraceAccordion';
import type { UIMessage } from 'ai';
import type { Source } from '@/types';
import type { AgentTraceEntry, CitationValidation } from '@/types/agent';

interface AgentMeta {
  trace: AgentTraceEntry[];
  toolCallsUsed: number;
  citationValidation: CitationValidation | null;
  isStreaming: boolean;
}

interface ChatMessageProps {
  role: UIMessage['role'];
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
  agentMeta?: AgentMeta;
  /** True only for the very first render of this message (controls accordion default-open). */
  isNew?: boolean;
}

export function ChatMessage({
  role,
  content,
  sources,
  isStreaming,
  agentMeta,
  isNew,
}: ChatMessageProps) {
  return (
    <Message from={role} className="mb-4">
      <MessageContent>
        {agentMeta && (
          <AgentTraceAccordion
            trace={agentMeta.trace}
            toolCallsUsed={agentMeta.toolCallsUsed}
            isStreaming={agentMeta.isStreaming}
            defaultOpen={isNew ?? false}
          />
        )}
        {sources && sources.length > 0 && (
          <Sources>
            <SourcesTrigger count={sources.length} />
            <SourcesContent>
              {sources.map((source, index) => (
                <SourceElement key={index} title={source.source}>
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-xs">{source.source}</span>
                      {source.score !== undefined && source.score !== null && (
                        <span className="text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                          {source.score.toFixed(2)}
                        </span>
                      )}
                    </div>
                    <span className="text-muted-foreground text-xs line-clamp-2">
                      {source.snippet}
                    </span>
                  </div>
                </SourceElement>
              ))}
            </SourcesContent>
          </Sources>
        )}
        <MessageResponse>{content}</MessageResponse>
        {isStreaming && (
          <span className="inline-block w-0.5 h-4 ml-1 bg-primary animate-pulse align-middle" />
        )}
      </MessageContent>
    </Message>
  );
}
