import { useRef, useEffect } from 'react';
import { ChatMessage } from './ChatMessage';
import type { Message } from '@/types';

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
}

export function MessageList({ messages, isLoading }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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
              isStreaming={
                message.role === 'assistant' &&
                message.id === messages[messages.length - 1]?.id &&
                isLoading
              }
            />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>
    </div>
  );
}
