# Task 11 — Frontend integração

## Objetivo

Substituir o `DebugStream` por UI completa de chat: bolhas de mensagem, streaming
de texto token-a-token, badges de tool calls com args, painel lateral de sources
(citations), suporte multi-turno via Redis. Frontend roda em docker-compose.

## Pré-requisitos

- Task 10 (frontend setup, pipe SSE validado).
- Backend completo até Task 07 (eventos `trace_id`, `tool_call`, `tool_result`,
  `sources`, `token`, `done`, `error`).

## Subtarefas

### 1. Modelo de estado de chat

`src/store/chatStore.ts` (opcional Zustand) — ou apenas `useState` no hook.

```typescript
export interface ToolCallView {
  id: string;
  name: string;
  args: Record<string, any>;
  status: "calling" | "done" | "error";
  preview?: string;
  error?: string;
  latencyMs?: number;
}

export interface ChatTurnView {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCalls?: ToolCallView[];
  citations?: Citation[];
  traceId?: string;
  isStreaming?: boolean;
}
```

### 2. Atualizar `useChatStream` para construir turns

```typescript
import { useReducer, useCallback, useRef } from "react";
import { streamChat } from "@/lib/api";
import type { Citation, ChatTurnView, ToolCallView } from "@/types/chat";

type State = {
  turns: ChatTurnView[];
  isStreaming: boolean;
  error: string | null;
};

type Action =
  | { type: "USER_MSG"; id: string; content: string }
  | { type: "START_ASSISTANT"; id: string }
  | { type: "TRACE_ID"; traceId: string }
  | { type: "TOKEN"; text: string }
  | { type: "TOOL_CALL"; tc: ToolCallView }
  | { type: "TOOL_RESULT"; name: string; preview?: string; error?: string; latencyMs?: number }
  | { type: "SOURCES"; citations: Citation[] }
  | { type: "DONE" }
  | { type: "ERROR"; message: string };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "USER_MSG":
      return {
        ...state,
        turns: [...state.turns, { id: action.id, role: "user", content: action.content }],
      };
    case "START_ASSISTANT":
      return {
        ...state,
        isStreaming: true,
        turns: [...state.turns, { id: action.id, role: "assistant", content: "", isStreaming: true }],
      };
    case "TRACE_ID":
      return updateLast(state, (t) => ({ ...t, traceId: action.traceId }));
    case "TOKEN":
      return updateLast(state, (t) => ({ ...t, content: t.content + action.text }));
    case "TOOL_CALL":
      return updateLast(state, (t) => ({
        ...t,
        toolCalls: [...(t.toolCalls ?? []), action.tc],
      }));
    case "TOOL_RESULT":
      return updateLast(state, (t) => ({
        ...t,
        toolCalls: t.toolCalls?.map((tc) =>
          tc.name === action.name && tc.status === "calling"
            ? { ...tc, status: action.error ? "error" : "done",
                preview: action.preview, error: action.error,
                latencyMs: action.latencyMs }
            : tc,
        ),
      }));
    case "SOURCES":
      return updateLast(state, (t) => ({ ...t, citations: action.citations }));
    case "DONE":
      return {
        ...state,
        isStreaming: false,
        turns: state.turns.map((t, i) =>
          i === state.turns.length - 1 ? { ...t, isStreaming: false } : t,
        ),
      };
    case "ERROR":
      return { ...state, isStreaming: false, error: action.message };
  }
}

function updateLast(state: State, fn: (t: ChatTurnView) => ChatTurnView): State {
  return {
    ...state,
    turns: state.turns.map((t, i) => (i === state.turns.length - 1 ? fn(t) : t)),
  };
}

export function useChatStream() {
  const [state, dispatch] = useReducer(reducer, {
    turns: [], isStreaming: false, error: null,
  });
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback(async (sessionId: string, message: string) => {
    dispatch({ type: "USER_MSG", id: crypto.randomUUID(), content: message });
    dispatch({ type: "START_ASSISTANT", id: crypto.randomUUID() });
    abortRef.current = new AbortController();

    try {
      for await (const ev of streamChat(
        { session_id: sessionId, message },
        abortRef.current.signal,
      )) {
        switch (ev.event) {
          case "trace_id":
            dispatch({ type: "TRACE_ID", traceId: ev.data.trace_id });
            break;
          case "token":
            dispatch({ type: "TOKEN", text: ev.data.text });
            break;
          case "tool_call":
            dispatch({
              type: "TOOL_CALL",
              tc: {
                id: crypto.randomUUID(),
                name: ev.data.name,
                args: ev.data.args ?? {},
                status: "calling",
              },
            });
            break;
          case "tool_result":
            dispatch({
              type: "TOOL_RESULT",
              name: ev.data.name,
              preview: ev.data.preview,
              error: ev.data.error,
              latencyMs: ev.data.latency_ms,
            });
            break;
          case "sources":
            dispatch({ type: "SOURCES", citations: ev.data.citations });
            break;
          case "done":
            dispatch({ type: "DONE" });
            break;
          case "error":
            dispatch({ type: "ERROR", message: ev.data.message });
            break;
        }
      }
    } catch (e) {
      dispatch({ type: "ERROR", message: String(e) });
    }
  }, []);

  return { ...state, send };
}
```

### 3. Componentes

`src/components/ChatWindow.tsx` — layout split:

```tsx
import { useState } from "react";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";
import { SourcesPanel } from "./SourcesPanel";
import { useChatStream } from "@/hooks/useChatStream";

const SESSION_KEY = "panvel:session_id";

function getOrCreateSessionId(): string {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

export function ChatWindow() {
  const [sessionId] = useState(getOrCreateSessionId);
  const { turns, isStreaming, error, send } = useChatStream();
  const lastAssistant = [...turns].reverse().find((t) => t.role === "assistant");

  return (
    <div className="grid h-screen grid-cols-1 md:grid-cols-[1fr_360px]">
      <div className="flex flex-col border-r">
        <MessageList turns={turns} />
        <ChatInput
          isStreaming={isStreaming}
          onSend={(msg) => send(sessionId, msg)}
        />
        {error && <div className="border-t bg-red-50 p-2 text-red-700">{error}</div>}
      </div>
      <SourcesPanel citations={lastAssistant?.citations ?? []} />
    </div>
  );
}
```

`src/components/MessageList.tsx`:
```tsx
import { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";
import type { ChatTurnView } from "@/types/chat";
import { ScrollArea } from "@/components/ui/scroll-area";

export function MessageList({ turns }: { turns: ChatTurnView[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), [turns]);

  return (
    <ScrollArea className="flex-1 p-4">
      <div className="mx-auto flex max-w-2xl flex-col gap-4">
        {turns.map((t) => <MessageBubble key={t.id} turn={t} />)}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
```

`src/components/MessageBubble.tsx`:
```tsx
import { StreamingText } from "./StreamingText";
import { ToolCallBadge } from "./ToolCallBadge";
import type { ChatTurnView } from "@/types/chat";

export function MessageBubble({ turn }: { turn: ChatTurnView }) {
  const isUser = turn.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 ${
          isUser ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-900"
        }`}
      >
        {turn.toolCalls?.length ? (
          <div className="mb-2 flex flex-wrap gap-1">
            {turn.toolCalls.map((tc) => <ToolCallBadge key={tc.id} tc={tc} />)}
          </div>
        ) : null}
        <StreamingText text={turn.content} streaming={turn.isStreaming} />
      </div>
    </div>
  );
}
```

`src/components/StreamingText.tsx`:
```tsx
export function StreamingText({ text, streaming }: { text: string; streaming?: boolean }) {
  return (
    <div className="whitespace-pre-wrap break-words">
      {text}
      {streaming && <span className="ml-0.5 inline-block h-4 w-1 animate-pulse bg-current align-middle" />}
    </div>
  );
}
```

`src/components/ToolCallBadge.tsx`:
```tsx
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import type { ToolCallView } from "@/types/chat";

export function ToolCallBadge({ tc }: { tc: ToolCallView }) {
  const [open, setOpen] = useState(false);
  const colorByStatus = {
    calling: "bg-amber-100 text-amber-800",
    done: "bg-green-100 text-green-800",
    error: "bg-red-100 text-red-800",
  }[tc.status];

  return (
    <div className="text-xs">
      <button
        onClick={() => setOpen(!open)}
        className={`rounded-full px-2 py-0.5 font-mono ${colorByStatus}`}
      >
        🔧 {tc.name} {tc.status === "calling" && "..."}
      </button>
      {open && (
        <pre className="mt-1 max-h-40 overflow-auto rounded bg-black/5 p-2 text-[10px]">
{JSON.stringify({ args: tc.args, result: tc.preview, error: tc.error }, null, 2)}
        </pre>
      )}
    </div>
  );
}
```

`src/components/SourcesPanel.tsx`:
```tsx
import { useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { Citation } from "@/types/chat";

export function SourcesPanel({ citations }: { citations: Citation[] }) {
  return (
    <ScrollArea className="hidden h-screen border-l bg-gray-50 md:block">
      <div className="p-4">
        <h3 className="mb-3 text-sm font-semibold text-gray-700">Fontes</h3>
        {citations.length === 0 && (
          <p className="text-xs text-gray-400">Sem fontes neste turno.</p>
        )}
        <div className="flex flex-col gap-3">
          {citations.map((c, i) => <SourceCard key={i} c={c} />)}
        </div>
      </div>
    </ScrollArea>
  );
}

function SourceCard({ c }: { c: Citation }) {
  const [open, setOpen] = useState(false);
  return (
    <button
      onClick={() => setOpen(!open)}
      className="rounded border bg-white p-3 text-left text-xs hover:bg-gray-50"
    >
      <div className="font-semibold">
        {c.med_name}
        {c.med_variant && <span className="text-gray-500"> ({c.med_variant})</span>}
      </div>
      <div className="text-gray-600">{c.section_label}</div>
      <div className="mt-1 text-gray-700">
        {open ? c.snippet : `${c.snippet.slice(0, 80)}...`}
      </div>
    </button>
  );
}
```

`src/components/ChatInput.tsx`:
```tsx
import { useState, KeyboardEvent } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

export function ChatInput({
  isStreaming,
  onSend,
}: {
  isStreaming: boolean;
  onSend: (msg: string) => void;
}) {
  const [val, setVal] = useState("");

  function submit() {
    if (val.trim() && !isStreaming) {
      onSend(val.trim());
      setVal("");
    }
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <div className="border-t bg-white p-3">
      <div className="mx-auto flex max-w-2xl gap-2">
        <Textarea
          value={val}
          onChange={(e) => setVal(e.target.value)}
          onKeyDown={onKey}
          placeholder="Pergunte sobre medicamentos ou filiais..."
          disabled={isStreaming}
          className="min-h-[52px] resize-none"
        />
        <Button onClick={submit} disabled={isStreaming || !val.trim()}>
          Enviar
        </Button>
      </div>
    </div>
  );
}
```

Adicionar `textarea` shadcn: `npx shadcn@latest add textarea`.

### 4. `App.tsx`

```tsx
import { ChatWindow } from "@/components/ChatWindow";

function App() {
  return <ChatWindow />;
}

export default App;
```

### 5. Frontend no docker-compose

Atualizar `docker-compose.yml` adicionando service `frontend`:

```yaml
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: panvel-frontend
    ports:
      - "5173:80"
    environment:
      VITE_API_URL: http://localhost:8000
    depends_on:
      - api
```

`frontend/Dockerfile` (build estático servido por nginx):

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

`frontend/nginx.conf`:
```nginx
server {
  listen 80;
  root /usr/share/nginx/html;
  location / {
    try_files $uri /index.html;
  }
}
```

Nota: como o `VITE_API_URL` é resolvido em build-time, se quiser trocar em runtime
sem rebuild, usar uma estratégia de `window.__ENV__` injetada. MVP: rebuild OK.

## Verificação

```bash
# dev local
cd frontend && npm run dev
# abre http://localhost:5173

# turno tools
# input: "quais lojas 24h em Curitiba com Panvel Clinic?"
# esperado:
# - bolha de usuário aparece à direita
# - bolha de assistant à esquerda
# - badge "🔧 buscar_filiais..." aparece em amber
# - quando tool retorna: badge fica verde
# - tokens stream → texto cita filial 1557
# - painel direito permanece vazio (não houve sources)

# turno RAG
# input: "contraindicações da ritalina"
# esperado:
# - SourcesPanel popula com 4 citations expandíveis
# - resposta cita [Ritalina — Quando não devo usar]
# - cursor pulsante durante stream

# Docker
docker compose up -d
# abre http://localhost:5173 — funciona idêntico
```

## Gotchas

- `useReducer` é mais simples que Zustand pra esse escopo. Subir pra Zustand se
  precisar persistir entre rotas (não é o caso).
- Cuidado com closures stale em `useChatStream` — `useCallback` com deps corretas.
- Side panel responsivo: `hidden md:block` esconde em mobile (Task 12 trata drawer).
- `Textarea` em vez de `Input` permite Shift+Enter quebra linha.
- VITE_API_URL build-time vs runtime: docker compose passa via ARG no build; pra prod
  com swap de URL sem rebuild, gerar `env.js` no entrypoint do nginx.
- Tool call matching no reducer: assumi que `name` é único por turno; se LLM chama
  mesma tool 2x, refinar usando `id` único por call.
- `crypto.randomUUID()` precisa HTTPS ou localhost; já é o caso em dev/prod.
