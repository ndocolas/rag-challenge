# Task 10 — Frontend setup

## Objetivo

Projeto frontend Vite + React + TS + Tailwind + shadcn/ui rodando. Pipe de SSE
validado renderizando eventos em raw JSON. Nenhum componente bonito ainda — só
provar que o stream chega e é parseado corretamente.

## Pré-requisitos

- Task 03 ou superior (backend `/chat` funcional).
- Node 20+ e npm/pnpm instalado.

## Subtarefas

### 1. Inicializar Vite + React + TS

Na raiz do repo:

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

### 2. Tailwind CSS

```bash
npm install -D tailwindcss@latest postcss autoprefixer
npx tailwindcss init -p
```

`tailwind.config.js`:
```js
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
};
```

`src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

### 3. shadcn/ui

```bash
npx shadcn@latest init
# escolher: default style, Slate base color, CSS variables yes
npx shadcn@latest add button input card scroll-area badge separator toast
```

(componentes vão para `src/components/ui/`)

### 4. Estrutura de pastas

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/             # shadcn components (gerados)
│   │   └── DebugStream.tsx # MVP: renderiza eventos raw
│   ├── hooks/
│   │   └── useChatStream.ts
│   ├── lib/
│   │   ├── api.ts          # cliente SSE
│   │   └── utils.ts        # (shadcn cn helper)
│   ├── types/
│   │   └── chat.ts         # tipagem dos eventos SSE
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── .env.example            # VITE_API_URL=http://localhost:8000
├── index.html
├── tsconfig.json
├── tailwind.config.js
├── vite.config.ts
└── package.json
```

### 5. `src/types/chat.ts`

```typescript
export type StreamEventType =
  | "trace_id"
  | "token"
  | "tool_call"
  | "tool_result"
  | "sources"
  | "done"
  | "error";

export interface StreamEvent {
  event: StreamEventType;
  data: any;
}

export interface Citation {
  bula_id: string;
  med_name: string;
  med_variant?: string | null;
  section_canonical: string;
  section_label: string;
  source_page?: number | null;
  snippet: string;
}

export interface ToolCallEvent {
  name: string;
  args: Record<string, any>;
}

export interface ToolResultEvent {
  name: string;
  preview?: string;
  error?: string;
  latency_ms?: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  toolCalls?: { name: string; args: Record<string, any>; result?: string; error?: string }[];
  citations?: Citation[];
  traceId?: string;
}
```

### 6. `src/lib/api.ts` — cliente SSE

EventSource não suporta POST com body — usar fetch + ReadableStream + parser SSE.

```typescript
import type { StreamEvent } from "@/types/chat";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface ChatRequest {
  session_id: string;
  message: string;
}

export async function* streamChat(
  req: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent, void, unknown> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE: blocos separados por \n\n; cada bloco tem linhas `event:` e `data:`
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";  // último pode estar incompleto

    for (const block of blocks) {
      if (!block.trim()) continue;
      const lines = block.split("\n");
      let eventType: string | null = null;
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (eventType && data) {
        try {
          yield { event: eventType as any, data: JSON.parse(data) };
        } catch {
          yield { event: eventType as any, data };
        }
      }
    }
  }
}
```

### 7. `src/hooks/useChatStream.ts`

```typescript
import { useState, useCallback, useRef } from "react";
import { streamChat } from "@/lib/api";
import type { StreamEvent } from "@/types/chat";

export function useChatStream() {
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback(async (sessionId: string, message: string) => {
    setEvents([]);
    setError(null);
    setIsStreaming(true);
    abortRef.current = new AbortController();

    try {
      for await (const ev of streamChat(
        { session_id: sessionId, message },
        abortRef.current.signal,
      )) {
        setEvents((prev) => [...prev, ev]);
        if (ev.event === "done" || ev.event === "error") {
          setIsStreaming(false);
        }
      }
    } catch (e) {
      setError(String(e));
      setIsStreaming(false);
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
  }, []);

  return { events, isStreaming, error, send, cancel };
}
```

### 8. `src/components/DebugStream.tsx`

```tsx
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChatStream } from "@/hooks/useChatStream";

const SESSION_KEY = "panvel:session_id";

export function DebugStream() {
  const [input, setInput] = useState("");
  const { events, isStreaming, error, send } = useChatStream();
  const sessionId =
    localStorage.getItem(SESSION_KEY) ??
    (() => {
      const id = crypto.randomUUID();
      localStorage.setItem(SESSION_KEY, id);
      return id;
    })();

  return (
    <div className="flex h-screen flex-col gap-4 p-4">
      <div className="flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="pergunta..."
          disabled={isStreaming}
        />
        <Button
          onClick={() => {
            if (input.trim()) send(sessionId, input);
            setInput("");
          }}
          disabled={isStreaming || !input.trim()}
        >
          {isStreaming ? "..." : "Enviar"}
        </Button>
      </div>

      {error && <div className="text-red-500">erro: {error}</div>}

      <ScrollArea className="flex-1 rounded-md border p-4">
        <pre className="text-xs">
          {events.map((ev, i) => (
            <div key={i} className="mb-2">
              <span className="font-bold text-blue-600">{ev.event}</span>
              {": "}
              <code>{JSON.stringify(ev.data)}</code>
            </div>
          ))}
        </pre>
      </ScrollArea>

      <div className="text-xs text-gray-500">
        session: {sessionId.slice(0, 8)}
      </div>
    </div>
  );
}
```

### 9. `src/App.tsx`

```tsx
import { DebugStream } from "@/components/DebugStream";

function App() {
  return <DebugStream />;
}

export default App;
```

### 10. `vite.config.ts` — alias `@`

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
});
```

`tsconfig.json` adicionar paths:
```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": { "@/*": ["./src/*"] }
  }
}
```

### 11. `frontend/.env.example`

```env
VITE_API_URL=http://localhost:8000
```

## Verificação

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
# abre http://localhost:5173

# digita "oi" e clica enviar
# esperado: lista crescente de eventos `token` com data como objeto
# {text:"O"} {text:"l"} {text:"á"} ... terminando em `done`

# digita "contraindicações ritalina" (com Qdrant indexado da Task 06)
# esperado: vê eventos sources, depois tokens, depois done
```

## Gotchas

- **EventSource NÃO serve**: ele só suporta GET. POST com body precisa de fetch
  manual + parser SSE (feito acima).
- CORS: backend já permite `localhost:5173` em ALLOWED_ORIGINS (Task 01).
- TextDecoder com `stream: true` é importante para não cortar UTF-8 multi-byte.
- Buffer SSE: bloco final pode chegar incompleto entre `read()`s — manter resíduo
  no `buffer` até próximo `\n\n`.
- shadcn cli: instala em `src/components/ui/`. NÃO editar esses arquivos (são
  vendored, eject-friendly).
- AbortController evita memory leak quando usuário sai da página durante stream.
- `crypto.randomUUID()` disponível em browsers modernos; fallback `Math.random` se
  precisar suportar contextos não-HTTPS (não é o caso em prod).
