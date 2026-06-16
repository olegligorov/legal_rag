import { useRef, useEffect } from 'react';
import { ChatMessage } from './ChatMessage';
import type { Message } from '@/types';

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
}

export function MessageList({ messages, isLoading }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const lastMessageId = useRef<string | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const newestAssistantId =
    messages.length > 0 && messages[messages.length - 1].role === 'assistant'
      ? messages[messages.length - 1].id
      : null;

  // Track whether the current newest assistant message is brand new this render
  const isNew = newestAssistantId !== null && newestAssistantId !== lastMessageId.current;
  if (newestAssistantId !== null) {
    lastMessageId.current = newestAssistantId;
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="container max-w-6xl mx-auto px-4 py-6">
        <div className="space-y-6">
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              role={message.role}
              content={message.content}
              sources={message.sources}
              agentMeta={message.agentMeta}
              isNew={message.id === newestAssistantId && isNew}
              isStreaming={
                message.role === 'assistant' &&
                message.id === messages[messages.length - 1]?.id &&
                isLoading &&
                !message.agentMeta
              }
            />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>
    </div>
  );
}
