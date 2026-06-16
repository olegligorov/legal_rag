import { ChatInput } from './ChatInput';
import { Switch } from '@/components/ui/switch';
import type { ChatMode } from '@/hooks/useChat';

interface ChatContainerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
}

export function ChatContainer({
  value,
  onChange,
  onSubmit,
  isLoading,
  mode,
  onModeChange,
}: ChatContainerProps) {
  return (
    <div className="border-t border-border bg-background/80 backdrop-blur-sm">
      <div className="container max-w-3xl mx-auto px-4 py-4">
        <ChatInput value={value} onChange={onChange} onSubmit={onSubmit} isLoading={isLoading} />
        <div className="mt-2 flex items-center justify-between">
          <p className="text-xs text-muted-foreground">
            Themis може да допуска грешки. Проверявайте важна информация.
          </p>
          <label className="flex items-center gap-1.5 cursor-pointer select-none">
            <span className="text-xs text-muted-foreground">Quick</span>
            <Switch
              size="sm"
              checked={mode === 'agent'}
              onCheckedChange={(checked) => onModeChange(checked ? 'agent' : 'quick')}
              aria-label="Toggle deep reasoning mode"
            />
            <span
              className={
                mode === 'agent' ? 'text-xs font-medium text-foreground' : 'text-xs text-muted-foreground'
              }
            >
              Deep reasoning
            </span>
          </label>
        </div>
      </div>
    </div>
  );
}
