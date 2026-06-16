import { useState, type ReactNode } from 'react';
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';
import { ChevronDownIcon, WrenchIcon, BrainIcon } from 'lucide-react';
import type { AgentTraceEntry, AgentSource } from '@/types/agent';

interface AgentTraceAccordionProps {
  trace: AgentTraceEntry[];
  toolCallsUsed: number;
  isStreaming: boolean;
  /** Open by default on first render (new message). */
  defaultOpen?: boolean;
}

export function AgentTraceAccordion({
  trace,
  toolCallsUsed,
  isStreaming,
  defaultOpen = false,
}: AgentTraceAccordionProps) {
  const [open, setOpen] = useState(defaultOpen);

  if (trace.length === 0 && !isStreaming) return null;

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="not-prose mb-3">
      <CollapsibleTrigger
        className={cn(
          'flex w-full items-center gap-2 text-muted-foreground text-xs transition-colors hover:text-foreground',
        )}
      >
        <BrainIcon className="size-3.5" />
        <span className="font-medium">
          Reasoning steps
          {toolCallsUsed > 0 ? ` (${toolCallsUsed} tool call${toolCallsUsed !== 1 ? 's' : ''})` : ''}
          {isStreaming && trace.length === 0 ? ' …' : ''}
        </span>
        <ChevronDownIcon
          className={cn('size-3.5 transition-transform', open ? 'rotate-180' : 'rotate-0')}
        />
      </CollapsibleTrigger>

      <CollapsibleContent className="mt-2 space-y-2 text-xs data-[state=closed]:animate-out data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0">
        {trace.map((entry, i) => (
          <TraceEntry key={i} entry={entry} />
        ))}
        {isStreaming && (
          <div className="text-muted-foreground italic animate-pulse pl-1">thinking…</div>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}

function TraceEntry({ entry }: { entry: AgentTraceEntry }) {
  switch (entry.type) {
    case 'thought':
      return (
        <div className="pl-2 border-l border-border text-muted-foreground italic whitespace-pre-wrap">
          {entry.content}
        </div>
      );
    case 'tool_call':
      return <ToolCallEntry name={entry.name} args={entry.args} />;
    case 'tool_result':
      return <ToolResultEntry name={entry.name} content={entry.content} />;
    case 'answer':
      return null;
  }
}

function ToolCallEntry({
  name,
  args,
}: {
  name: string;
  args: Record<string, unknown>;
}) {
  return (
    <div className="rounded-md border border-border bg-muted/30 p-2 space-y-1.5">
      <div className="flex items-center gap-1.5">
        <WrenchIcon className="size-3 text-muted-foreground" />
        <span className="font-mono font-medium text-foreground">{name}</span>
      </div>
      <pre className="overflow-x-auto rounded bg-muted/50 px-2 py-1 text-[11px] text-muted-foreground">
        {JSON.stringify(args, null, 2)}
      </pre>
    </div>
  );
}

type RagToolResult = {
  question?: string;
  answer?: string;
  sources?: AgentSource[];
};

type BatchToolResult = {
  results?: Array<{ question?: string; answer?: string; sources?: AgentSource[] }>;
  total?: number;
  successful?: number;
  failed?: number;
};

function isRagResult(content: unknown): content is RagToolResult {
  return (
    typeof content === 'object' &&
    content !== null &&
    ('answer' in content || 'question' in content)
  );
}

function isBatchResult(content: unknown): content is BatchToolResult {
  return typeof content === 'object' && content !== null && 'results' in content;
}

function ToolResultEntry({ name, content }: { name: string; content: unknown }) {
  const [open, setOpen] = useState(false);

  let preview: ReactNode = null;

  if (isRagResult(content)) {
    preview = <RagResultPreview result={content} />;
  } else if (isBatchResult(content)) {
    preview = <BatchResultPreview result={content} />;
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-md border border-border bg-muted/20">
      <CollapsibleTrigger className="flex w-full items-center justify-between gap-2 px-2 py-1.5 text-muted-foreground hover:text-foreground">
        <span className="font-mono text-[11px]">{name} result</span>
        <ChevronDownIcon
          className={cn('size-3 shrink-0 transition-transform', open ? 'rotate-180' : 'rotate-0')}
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="px-2 pb-2 space-y-1.5 data-[state=closed]:animate-out data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0">
        {preview ?? (
          <pre className="overflow-x-auto rounded bg-muted/50 px-2 py-1 text-[11px] text-muted-foreground">
            {JSON.stringify(content, null, 2)}
          </pre>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}

function RagResultPreview({ result }: { result: RagToolResult }) {
  return (
    <div className="space-y-1.5">
      {result.answer && (
        <p className="text-[11px] text-muted-foreground line-clamp-3">{result.answer}</p>
      )}
      {result.sources && result.sources.length > 0 && (
        <SourceList sources={result.sources} />
      )}
    </div>
  );
}

function BatchResultPreview({ result }: { result: BatchToolResult }) {
  return (
    <div className="space-y-2">
      {result.total !== undefined && (
        <p className="text-[11px] text-muted-foreground">
          {result.successful}/{result.total} queries succeeded
        </p>
      )}
      {result.results?.map((r, i) => (
        <div key={i} className="space-y-1 border-t border-border pt-1.5 first:border-0 first:pt-0">
          {r.question && (
            <p className="text-[11px] font-medium text-foreground/80">{r.question}</p>
          )}
          {r.answer && (
            <p className="text-[11px] text-muted-foreground line-clamp-2">{r.answer}</p>
          )}
          {r.sources && r.sources.length > 0 && <SourceList sources={r.sources} />}
        </div>
      ))}
    </div>
  );
}

function SourceList({ sources }: { sources: AgentSource[] }) {
  return (
    <ul className="space-y-0.5">
      {sources.map((s, i) => (
        <li key={i} className="flex items-baseline gap-1.5 text-[11px] text-muted-foreground">
          <span className="shrink-0 font-mono text-[10px] tabular-nums">{s.rank}.</span>
          {s.article && <span className="font-medium text-foreground/70">{s.article}</span>}
          {s.law_id && <span className="opacity-70">{s.law_id}</span>}
          {s.score !== undefined && (
            <span className="ml-auto shrink-0 opacity-50">{s.score.toFixed(2)}</span>
          )}
        </li>
      ))}
    </ul>
  );
}
