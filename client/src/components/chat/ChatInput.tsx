import {
  PromptInput,
  PromptInputBody,
  PromptInputTextarea,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTools,
  PromptInputButton,
  usePromptInputAttachments,
} from '@/components/ai-elements/prompt-input';
import {
  Attachments,
  Attachment,
  AttachmentPreview,
  AttachmentRemove,
} from '@/components/ai-elements/attachments';
import { ImageIcon } from 'lucide-react';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isLoading?: boolean;
  placeholder?: string;
}

function AttachmentsList() {
  const attachments = usePromptInputAttachments();

  if (attachments.files.length === 0) return null;

  return (
    <Attachments variant="inline">
      {attachments.files.map((file) => (
        <Attachment key={file.id} data={file} onRemove={() => attachments.remove(file.id)}>
          <AttachmentPreview />
          <AttachmentRemove />
        </Attachment>
      ))}
    </Attachments>
  );
}

export function ChatInput({
  value,
  onChange,
  onSubmit,
  isLoading = false,
  placeholder = 'Ask a question about your knowledge base...',
}: ChatInputProps) {
  const handleSubmit = (message: { text: string; files: any[] }) => {
    if ((message.text.trim() || message.files.length > 0) && !isLoading) {
      onChange(message.text);
      onSubmit();
    }
  };

  return (
    <PromptInput onSubmit={handleSubmit} accept="image/*" multiple maxFiles={5}>
      <PromptInputBody>
        <AttachmentsList />
        <PromptInputTextarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={isLoading}
        />
      </PromptInputBody>
      <PromptInputFooter>
        <PromptInputTools>
          <AttachButton />
        </PromptInputTools>
        <PromptInputSubmit
          status={isLoading ? 'streaming' : undefined}
          disabled={!value.trim() || isLoading}
        />
      </PromptInputFooter>
    </PromptInput>
  );
}

function AttachButton() {
  const attachments = usePromptInputAttachments();

  return (
    <PromptInputButton onClick={attachments.openFileDialog} aria-label="Attach image">
      <ImageIcon className="h-4 w-4" />
    </PromptInputButton>
  );
}
