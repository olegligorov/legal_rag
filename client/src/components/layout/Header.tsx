import { Button } from '@/components/ui/button';
import { Database, Github, Settings } from 'lucide-react';

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/80 backdrop-blur-sm">
      <div className="container flex h-16 items-center justify-between px-4 max-w-full">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
            <Button
              onClick={() => (window.location.href = '/')}
              variant="ghost"
              size="icon"
              className="h-9 w-9 text-muted-foreground hover:text-foreground"
            >
              <Database className="h-5 w-5 text-primary-foreground" />
            </Button>
          </div>
          <div className="flex flex-col">
            <h1 className="text-lg font-semibold text-foreground leading-none">Themis</h1>
            <p className="text-xs text-muted-foreground leading-none mt-1">RAG-Powered Assistant</p>
          </div>
        </div>
        {/* <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9 text-muted-foreground hover:text-foreground"
            onClick={() => window.open('https://github.tools.sap/I750415/rag', '_blank')}
          >
            <Github className="h-5 w-5" />
            <span className="sr-only">GitHub</span>
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9 text-muted-foreground hover:text-foreground"
          >
            <Settings className="h-5 w-5" />
            <span className="sr-only">Settings</span>
          </Button>
        </div> */}
      </div>
    </header>
  );
}
