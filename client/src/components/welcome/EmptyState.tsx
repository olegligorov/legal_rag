import { Sparkles } from 'lucide-react';

export function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center text-center py-16 px-4">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 mb-6">
        <Sparkles className="h-8 w-8 text-primary" />
      </div>
      <h2 className="text-2xl font-semibold text-foreground mb-2 text-balance">
        Задайте въпрос за българското законодателство
      </h2>
      <p className="text-muted-foreground max-w-md text-balance leading-relaxed">
        Търся в Кодекса на труда, Закона за защита на потребителите и Закона за задълженията и
        договорите. Отговарям с цитати от конкретни членове.
      </p>
    </div>
  );
}
