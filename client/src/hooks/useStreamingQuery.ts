import { useQuery } from '@tanstack/react-query';
import { queryRAGStream } from '@/services/api';
import type { Source } from '@/types';

interface UseStreamingQueryOptions {
  onMetadata?: (sources: Source[]) => void;
  onChunk?: (chunk: string) => void;
  onDone?: () => void;
  onError?: (error: string) => void;
}

export function useStreamingQuery(question: string | null, options: UseStreamingQueryOptions) {
  return useQuery({
    queryKey: ['rag-stream', question],
    queryFn: async ({ signal }) => {
      if (!question) return null;

      let fullAnswer = '';
      const sources: Source[] = [];

      for await (const event of queryRAGStream({ question })) {
        if (signal?.aborted) {
          throw new Error('Query aborted');
        }

        switch (event.type) {
          case 'metadata':
            sources.push(...event.sources);
            options.onMetadata?.(event.sources);
            break;

          case 'chunk':
            fullAnswer += event.content;
            options.onChunk?.(event.content);
            break;

          case 'done':
            options.onDone?.();
            break;

          case 'error':
            options.onError?.(event.message);
            throw new Error(event.message);
        }
      }

      return { answer: fullAnswer, sources };
    },
    enabled: !!question,
    staleTime: 0,
    gcTime: 0,
    retry: false,
  });
}
