# Frontend Chat UI — Design Spec

**Date:** 2026-05-20  
**Status:** Approved  
**Scope:** Full chat UI (Tasks 10–11 combined, skipping debug MVP)

---

## Overview

A dark-mode React chat application backed by the existing FastAPI SSE streaming backend. The UI mirrors the ChatGPT layout: collapsible left sidebar listing past sessions, a centered chat column with streaming assistant responses, and inline citation chips at the end of each answer. Two small backend additions enable session listing and history loading.

---

## Stack

- **Frontend:** Vite + React 18 + TypeScript + Tailwind CSS + shadcn/ui
- **Location:** `frontend/` at repo root
- **Dev server:** `http://localhost:5173` (CORS already open on backend)
- **Env:** `VITE_API_URL=http://localhost:8000` (`.env.example`)
- **No external state library** — React Context + `useState`/`useReducer` sufficient

---

## Folder Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   ├── ChatLayout.tsx       # Root flex layout: sidebar + main
│   │   │   ├── Sidebar.tsx          # Collapsible session list panel
│   │   │   └── ChatView.tsx         # Active session: MessageList + ChatInput
│   │   ├── chat/
│   │   │   ├── MessageList.tsx      # Scrollable list, auto-scroll on stream
│   │   │   ├── UserMessage.tsx      # Right-aligned bubble
│   │   │   ├── AssistantMessage.tsx # Left-aligned, markdown rendered
│   │   │   ├── ThinkingIndicator.tsx# Animated badge shown during tool calls
│   │   │   ├── CitationChips.tsx    # Numbered [1][2] pills → popover on click
│   │   │   └── EmptyState.tsx       # Shown when no active session
│   │   └── ui/                      # shadcn generated components
│   ├── hooks/
│   │   ├── useChatStream.ts         # POST /chat SSE fetch + streaming state
│   │   └── useSessionList.ts        # GET /sessions + refresh trigger
│   ├── lib/
│   │   └── api.ts                   # fetch wrapper, SSE parser, error handling
│   ├── types/
│   │   └── chat.ts                  # ChatMessage, Citation, Session, SSEEvent
│   ├── context/
│   │   └── AppContext.tsx           # activeSessionId, sidebarOpen
│   └── App.tsx
├── .env.example
└── package.json
```

---

## Layout

Full-height `flex-row`. Dark background (`bg-zinc-950`).

```
┌──────────┬────────────────────────────────────┐
│          │  [chat header — session title]      │
│ Sidebar  │                                     │
│ (260px)  │  MessageList (scrollable)           │
│          │                                     │
│ [chats   │                                     │
│  list]   │  ─────────────────────────────────  │
│          │  ChatInput (sticky bottom)          │
└──────────┴────────────────────────────────────┘
```

A hamburger button (top-left, always visible) toggles the sidebar. When closed, sidebar slides off-screen (`translate-x-full`) and the chat area expands to full width.

---

## Components

### Sidebar
- "New Chat" button at top — creates a new `crypto.randomUUID()` session, sets it active, triggers session list refresh
- Scrollable list of sessions: title (first user message, ≤60 chars) + relative timestamp
- Active session highlighted (`bg-zinc-800`)
- Fetches from `GET /sessions` on mount and after each `done` event

### ChatView
- Renders `EmptyState` when `activeSessionId` is null
- Loads history via `GET /sessions/{id}/history` when switching sessions
- Passes `session_id` + user message to `useChatStream` on send

### MessageList
- Scrolls to bottom on every new token (via `useEffect` + `ref.scrollIntoView`)
- User messages: right-aligned, `bg-zinc-800` bubble, plain text
- Assistant messages: left-aligned, no bubble border, markdown via `react-markdown`

### ThinkingIndicator
- Shown above the forming assistant response while `currentTool` is set
- Animated pulse dot + localized tool label:
  - `buscar_filiais` → "Buscando filiais..."
  - `detalhes_filial` → "Consultando filial..."
  - `listar_cidades_atendidas` → "Buscando cidades..."
  - `search_bulas` → "Consultando bula..."
  - fallback → "Pesquisando..."
- Disappears after `tool_result` or `done`

### CitationChips
- Rendered after `done` if `citations` is non-empty
- Numbered pills: `[1]` `[2]` ... in `text-xs bg-zinc-700 rounded`
- Click opens a shadcn `Popover` with:
  - Medication name + variant (if any)
  - Section label
  - Snippet (truncated to 200 chars, `...` if longer)
  - Source page if available

### ChatInput
- `textarea` (auto-expands 1→5 rows via `onInput` height adjustment)
- `Enter` → send, `Shift+Enter` → newline
- Send button → stop button (AbortController) while streaming
- Disabled while `isStreaming`

### EmptyState
- Centered in chat area: app name + "Como posso ajudar?"
- 3 suggested questions from `docs/queries-piloto.md` as clickable chips

---

## Data Flow

### Types (`src/types/chat.ts`)

```typescript
type MessageRole = "user" | "assistant";
type MessageStatus = "done" | "streaming" | "error";

interface Citation {
  bula_id: string;
  med_name: string;
  med_variant?: string;
  section_label: string;
  snippet: string;
  source_page?: number;
}

interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  citations?: Citation[];
  currentTool?: string;     // set during streaming
  status: MessageStatus;
}

interface Session {
  session_id: string;
  title: string;
  created_at: string;       // ISO 8601
}
```

### Global Context (`AppContext`)

```typescript
{ activeSessionId: string | null, setActiveSessionId, sidebarOpen, setSidebarOpen }
```

### Send flow

1. User submits message → `useChatStream.send(message)`
2. Append `UserMessage` (optimistic)
3. Append empty `AssistantMessage` with `status: "streaming"`
4. `POST /chat` with `{session_id, message}` via `fetch` + `AbortController`
5. Parse `ReadableStream` with `TextDecoder({ stream: true })`, split on `\n\n`
6. Per SSE frame:
   - `event: token` → `data` is plain text → append to `content`
   - `event: sources` → `JSON.parse(data).citations` → set on message
   - `event: tool_call` → `JSON.parse(data).name` → set `currentTool`
   - `event: tool_result` → clear `currentTool`
   - `event: done` → set `status: "done"`, trigger session list refresh
   - `event: error` → set `status: "error"`, append error note to content
7. HTTP error before stream (non-2xx) → parse JSON error envelope → show toast

### SSE token parsing (critical)

The backend sends `event: token` with **plain text** in `data:` (not JSON). Multi-line data fields must be joined with `\n`. All other events are `JSON.parse`.

```typescript
function parseSSEFrame(frame: string): { event: string; data: string } | null {
  const lines = frame.split("\n");
  let event = "message";
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("event: ")) event = line.slice(7);
    else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
  }
  if (!dataLines.length) return null;
  return { event, data: dataLines.join("\n") };
}
```

---

## Backend Changes

### New Redis key: `sessions:meta`

A Redis sorted set where score = Unix timestamp (creation time) and member = JSON string `{session_id, title, created_at}`.

Written in `ChatHistoryService` (or directly in the chat route) on first message of a session. If the key already exists for a session, skip the write (title = first message only).

Key: `sessions:meta` (global, no per-session prefix)  
TTL: same as `CHAT_HISTORY_TTL_SECONDS` (30 min), refreshed alongside the history key.

### `GET /sessions`

Returns list of known sessions sorted newest-first.

```python
# Response
[
  {"session_id": "...", "title": "Quais são as...", "created_at": "2026-05-20T..."},
  ...
]
```

Returns `[]` when no sessions exist. New route file: `src/panvel_assistant/routes/sessions.py`.

### `GET /sessions/{session_id}/history`

Calls the existing `RedisChatMessageHistory.aget_messages()` and maps LangChain messages to frontend shape.

```python
# Response
[
  {"id": "...", "role": "user", "content": "...", "citations": null, "status": "done"},
  {"id": "...", "role": "assistant", "content": "...", "citations": [...], "status": "done"},
  ...
]
```

Returns 404 with error envelope if session not found in Redis (expired TTL). The frontend handles this by removing the session from the sidebar.

**Citations in history:** The existing `AssistantService` must persist citations alongside the AI message in Redis when it saves turn history. Currently only the text content is saved. A small addition is needed: store citations as `additional_kwargs["citations"]` on the `AIMessage` before persisting.

---

## Error Handling

| Scenario | Behavior |
|---|---|
| HTTP 429 before stream | Toast: "Muitas mensagens. Aguarde um momento." |
| HTTP 409 before stream | Toast: "Aguarde a resposta anterior terminar." |
| HTTP 422 validation | Toast with backend message |
| Network failure | Toast: "Erro de conexão. Tente novamente." |
| SSE `error` event | Inline note below partial text: "Ocorreu um erro. A resposta pode estar incompleta." |
| Stop button (abort) | Appends "Resposta interrompida." to partial content, sets `status: "done"` |
| 404 on history load | Remove session from sidebar list, show empty state |

Toasts use shadcn `Sonner`, auto-dismiss after 4s.

---

## Key Constraints & Notes

- Do **not** use `EventSource` (GET-only). Use `fetch` POST + `ReadableStream`.
- `session_id` must match `^[A-Za-z0-9_-]{1,128}$` — `crypto.randomUUID()` satisfies this.
- Backend max body: 16 KB. Frontend `textarea` max length: 4000 chars (matches `ChatRequest` model).
- Send button disabled while `isStreaming` — backend rejects concurrent turns with 409.
- `react-markdown` for assistant message rendering (supports bulas content which often contains lists and tables).
- Dark mode only — no light mode toggle needed. Base color: `zinc-950` page, `zinc-900` sidebar, `zinc-800` user bubble.
