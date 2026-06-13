import { EmptyState } from './EmptyState';
import { SuggestedQuestions } from './SuggestedQuestions';

interface WelcomeScreenProps {
  questions: string[];
  onSelectQuestion: (question: string) => void;
}

export function WelcomeScreen({ questions, onSelectQuestion }: WelcomeScreenProps) {
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="container max-w-6xl mx-auto px-4 py-6">
        <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-8">
          <EmptyState />
          <SuggestedQuestions questions={questions} onSelect={onSelectQuestion} />
        </div>
      </div>
    </div>
  );
}
