import { useState, useCallback } from 'react';
import { queryRAGStream } from '@/services/api';
import type { Message, Source } from '@/types';

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = useCallback(async (question: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: question,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const assistantMessageId = (Date.now() + 1).toString();
    let assistantContent = '';
    let sources: Source[] = [];

    try {
      for await (const event of queryRAGStream({ question })) {
        switch (event.type) {
          case 'metadata':
            sources = event.sources;
            // Create the assistant message immediately with sources
            setMessages((prev) => [
              ...prev,
              {
                id: assistantMessageId,
                role: 'assistant' as const,
                content: '',
                sources: sources,
              },
            ]);
            break;

          case 'chunk':
            assistantContent += event.content;
            setMessages((prev) => {
              return prev.map((m) =>
                m.id === assistantMessageId ? { ...m, content: assistantContent } : m,
              );
            });
            break;

          case 'done':
            setIsLoading(false);
            break;

          case 'error':
            console.error('Stream error:', event.message);
            setIsLoading(false);
            setMessages((prev) => {
              const existing = prev.find((m) => m.id === assistantMessageId);
              if (existing) {
                return prev.map((m) =>
                  m.id === assistantMessageId ? { ...m, content: 'Error: ' + event.message } : m,
                );
              }
              return [
                ...prev,
                {
                  id: assistantMessageId,
                  role: 'assistant',
                  content: 'Error: ' + event.message,
                },
              ];
            });
            break;
        }
      }
    } catch (error) {
      console.error('Query error:', error);
      setIsLoading(false);
      setMessages((prev) => {
        const existing = prev.find((m) => m.id === assistantMessageId);
        if (existing) {
          return prev.map((m) =>
            m.id === assistantMessageId
              ? { ...m, content: 'Failed to process query. Please try again.' }
              : m,
          );
        }
        return [
          ...prev,
          {
            id: assistantMessageId,
            role: 'assistant',
            content: 'Failed to process query. Please try again.',
          },
        ];
      });
    }
  }, []);

  return {
    messages,
    input,
    setInput,
    isLoading,
    sendMessage,
  };
}
