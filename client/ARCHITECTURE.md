# Frontend Architecture

A clean, modular React + Vite application with TypeScript and TanStack Query.

## Project Structure

```
src/
├── components/
│   ├── layout/              # Layout components
│   │   ├── Header.tsx       # App header with branding
│   │   ├── Layout.tsx       # Main layout wrapper
│   │   └── index.ts         # Barrel export
│   │
│   ├── chat/                # Chat-related components
│   │   ├── ChatMessage.tsx  # Individual message bubble
│   │   ├── ChatInput.tsx    # Message input with auto-resize
│   │   ├── ChatContainer.tsx # Input area wrapper
│   │   ├── MessageList.tsx  # Message list with auto-scroll
│   │   └── index.ts         # Barrel export
│   │
│   ├── welcome/             # Welcome screen components
│   │   ├── EmptyState.tsx   # Empty state with icon
│   │   ├── SuggestedQuestions.tsx # Question cards
│   │   ├── WelcomeScreen.tsx # Combined welcome view
│   │   └── index.ts         # Barrel export
│   │
│   ├── ui/                  # shadcn/ui components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   └── ...
│   │
│   ├── theme-provider.tsx   # Dark/light theme provider
│   └── index.ts             # Main barrel export
│
├── hooks/                   # Custom React hooks
│   ├── useChat.ts           # Chat state management
│   └── useStreamingQuery.ts # TanStack Query streaming wrapper
│
├── services/                # API and external services
│   └── api.ts               # Axios client + streaming
│
├── lib/                     # Utilities
│   └── utils.ts             # Helper functions (cn, etc.)
│
├── App.tsx                  # Main app component (45 lines)
├── main.tsx                 # Entry point
└── index.css                # Global styles + Tailwind
```

## Component Hierarchy

```
App
└── Layout
    ├── Header
    └── main (content area)
        ├── WelcomeScreen (when no messages)
        │   ├── EmptyState
        │   └── SuggestedQuestions
        │
        ├── MessageList (when has messages)
        │   └── ChatMessage (multiple)
        │
        └── ChatContainer
            └── ChatInput
```

## Key Design Principles

### 1. Modular Components

- Each component has a single responsibility
- Components are grouped by feature (layout, chat, welcome)
- Barrel exports (`index.ts`) for clean imports

### 2. Clean App.tsx

The main App component is only 45 lines:

```typescript
function App() {
  const { messages, input, setInput, isLoading, sendMessage } = useChat()

  const handleSubmit = () => {
    if (input.trim() && !isLoading) {
      sendMessage(input)
    }
  }

  const hasMessages = messages.length > 0

  return (
    <ThemeProvider defaultTheme="dark" storageKey="askbase-theme">
      <Layout>
        {hasMessages ? (
          <MessageList messages={messages} isLoading={isLoading} />
        ) : (
          <WelcomeScreen
            questions={SUGGESTED_QUESTIONS}
            onSelectQuestion={sendMessage}
          />
        )}
        <ChatContainer
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          isLoading={isLoading}
        />
      </Layout>
    </ThemeProvider>
  )
}
```

### 3. Separation of Concerns

#### Layout Components

- **Layout**: Provides the main app structure (header + content area)
- **Header**: Displays branding and navigation

#### Chat Components

- **ChatMessage**: Renders a single message with optional sources
- **ChatInput**: Auto-resizing textarea with submit button
- **ChatContainer**: Wraps input with disclaimer text
- **MessageList**: Manages message rendering and auto-scrolling

#### Welcome Components

- **EmptyState**: Shows welcome message with icon
- **SuggestedQuestions**: Displays clickable question cards
- **WelcomeScreen**: Combines EmptyState + SuggestedQuestions

### 4. Custom Hooks

#### useChat

Manages all chat state and streaming logic:

- Messages array
- Input state
- Loading state
- sendMessage function with streaming

#### useStreamingQuery (optional)

TanStack Query wrapper for streaming queries with callbacks.

## Data Flow

```
User Input
    ↓
ChatInput → onChange/onSubmit
    ↓
App → handleSubmit
    ↓
useChat → sendMessage
    ↓
api.ts → queryRAGStream()
    ↓ (SSE events)
Backend API
    ↓ (streaming response)
useChat → state updates
    ↓ (messages, isLoading)
MessageList → ChatMessage
```

## Streaming Architecture

### 1. Sources Appear First

When a query is sent:

1. User message is added immediately
2. Metadata event arrives with sources
3. Assistant message is created with sources (empty content)
4. Content streams in and updates the message
5. Done event marks completion

### 2. Real-time Updates

The useChat hook uses React state updates to progressively render:

- Sources appear immediately when metadata arrives
- Text streams character by character
- Auto-scroll keeps latest message visible

## Adding New Features

### Adding a New Component

1. Create component in appropriate folder:

```typescript
// src/components/chat/MessageActions.tsx
export function MessageActions({ messageId }: { messageId: string }) {
  return (
    <div className="flex gap-2">
      {/* actions */}
    </div>
  )
}
```

2. Export from folder's index.ts:

```typescript
// src/components/chat/index.ts
export { MessageActions } from './MessageActions';
```

3. Use in parent component:

```typescript
import { MessageActions } from '@/components/chat';
```

### Adding a New Hook

1. Create hook file:

```typescript
// src/hooks/useMessageHistory.ts
export function useMessageHistory() {
  const [history, setHistory] = useState<string[]>([]);
  // implementation
  return { history, addToHistory };
}
```

2. Use in components:

```typescript
import { useMessageHistory } from '@/hooks/useMessageHistory';
```

### Extending the API

1. Add types to `src/services/api.ts`:

```typescript
export interface FeedbackRequest {
  messageId: string;
  rating: number;
}
```

2. Create API function:

```typescript
export const submitFeedback = async (feedback: FeedbackRequest) => {
  const response = await apiClient.post('/api/feedback', feedback);
  return response.data;
};
```

3. Use in components via hooks.

## Benefits of This Architecture

### 1. Easy to Navigate

- Clear folder structure
- Components grouped by feature
- Barrel exports for clean imports

### 2. Easy to Test

- Small, focused components
- Hooks can be tested in isolation
- Clear separation of concerns

### 3. Easy to Extend

- Add new components without touching existing code
- Modular structure supports growth
- TanStack Query ready for complex data fetching

### 4. Easy to Maintain

- App.tsx is minimal orchestration
- Each component has single responsibility
- TypeScript ensures type safety

## Development

### Start Dev Server

```bash
npm run dev
```

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Environment Variables

Create `.env` file:

```
VITE_API_URL=http://localhost:8000
```

## Technology Stack

- **React 19**: Latest React with new features
- **Vite 6**: Fast build tool and dev server
- **TypeScript**: Strict typing with verbatimModuleSyntax
- **TanStack Query v5**: Server state management
- **Tailwind CSS v4**: Utility-first styling
- **shadcn/ui**: Accessible component library
- **Axios**: HTTP client
- **lucide-react**: Icon library
