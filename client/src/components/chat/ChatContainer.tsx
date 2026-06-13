import { ChatInput } from './ChatInput';

interface ChatContainerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

export function ChatContainer({ value, onChange, onSubmit, isLoading }: ChatContainerProps) {
  return (
    <div className="border-t border-border bg-background/80 backdrop-blur-sm">
      <div className="container max-w-3xl mx-auto px-4 py-4">
        <ChatInput value={value} onChange={onChange} onSubmit={onSubmit} isLoading={isLoading} />
        <p className="text-xs text-muted-foreground text-center mt-2">
          Themis може да допуска грешки. Проверявайте важна информация.
        </p>
      </div>
    </div>
  );
}
