import type { AgentStreamEvent } from '@/types/agent';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function* agentStream(question: string): AsyncGenerator<AgentStreamEvent> {
  const response = await fetch(`${API_BASE_URL}/api/agent/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
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

      const lines = buffer.split('\n\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          yield JSON.parse(line.slice(6)) as AgentStreamEvent;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
