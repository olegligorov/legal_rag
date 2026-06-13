import { Button } from '@/components/ui/button';
import { MessageSquare } from 'lucide-react';

interface SuggestedQuestionsProps {
  questions: string[];
  onSelect: (question: string) => void;
}

export function SuggestedQuestions({ questions, onSelect }: SuggestedQuestionsProps) {
  return (
    <div className="w-full max-w-3xl mx-auto px-4">
      <h3 className="text-sm font-medium text-muted-foreground mb-3 text-center">
        Suggested questions
      </h3>
      <div className="grid gap-3 sm:grid-cols-2">
        {questions.map((question, index) => (
          <Button
            key={index}
            variant="outline"
            className="rounded-xl h-auto p-4 justify-start text-left whitespace-normal hover:bg-secondary/80 hover:border-primary/50 transition-all group bg-transparent"
            onClick={() => onSelect(question)}
          >
            <MessageSquare className="h-4 w-4 mr-3 shrink-0 text-primary group-hover:scale-110 transition-transform" />
            <span className="text-sm text-foreground">{question}</span>
          </Button>
        ))}
      </div>
    </div>
  );
}
