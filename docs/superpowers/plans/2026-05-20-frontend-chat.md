# Frontend Chat UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full dark-mode React chat UI with SSE streaming, collapsible session sidebar, and citation chips, backed by two new FastAPI endpoints for session listing and history.

**Architecture:** Backend gains a `sessions:meta` Redis sorted set (+ per-session hash) for session metadata, and two new routes (`GET /sessions`, `GET /sessions/{id}/history`). The `AssistantService` is extended to persist citations in `AIMessage.additional_kwargs`. The frontend is a Vite/React/TS app with a custom SSE fetch hook, React Context for global state, and shadcn/ui components throughout.

**Tech Stack:** Python 3.12 + FastAPI + fakeredis (backend tests) | Vite + React 18 + TypeScript + Tailwind CSS + shadcn/ui + react-markdown + sonner + date-fns (frontend)

**Spec:** `docs/superpowers/specs/2026-05-20-frontend-chat-design.md`

---

## File Map

**Backend — modified:**
- `src/panvel_assistant/services/chat_history_service.py` — add `save_session_meta()`, `list_sessions()` to `RedisHistoryStore`
- `src/panvel_assistant/assistant/assistant_service.py` — track citations locally, persist in `additional_kwargs`, call `save_session_meta()`
- `src/panvel_assistant/main.py` — register sessions router

**Backend — created:**
- `src/panvel_assistant/routes/sessions.py` — `GET /sessions` + `GET /sessions/{session_id}/history`
- `tests/unit/test_sessions_service.py` — unit tests for new `RedisHistoryStore` methods
- `tests/integration/test_sessions_routes.py` — integration tests for both new routes

**Frontend — created (all under `frontend/`):**
- `src/types/chat.ts` — `ChatMessage`, `Citation`, `Session`, `SSEFrame` types
- `src/lib/api.ts` — fetch wrapper, SSE async generator, `parseSSEFrame`, error helpers
- `src/hooks/useChatStream.ts` — manages messages array + streaming state machine
- `src/hooks/useSessionList.ts` — fetches and refreshes session list
- `src/context/AppContext.tsx` — `activeSessionId`, `sidebarOpen`
- `src/components/layout/ChatLayout.tsx` — root flex layout
- `src/components/layout/Sidebar.tsx` — collapsible session list panel
- `src/components/layout/ChatView.tsx` — active session view (wires hooks + components)
- `src/components/chat/MessageList.tsx` — scrollable list with auto-scroll
- `src/components/chat/UserMessage.tsx` — right-aligned bubble
- `src/components/chat/AssistantMessage.tsx` — markdown + ThinkingIndicator + CitationChips
- `src/components/chat/ThinkingIndicator.tsx` — animated tool-in-progress badge
- `src/components/chat/CitationChips.tsx` — numbered pills + popover
- `src/components/chat/EmptyState.tsx` — empty session landing
- `src/components/chat/ChatInput.tsx` — auto-resize textarea + send/stop button
- `src/App.tsx` — mounts context + layout

---

## Task 1: Backend — Session metadata storage

**Files:**
- Modify: `src/panvel_assistant/services/chat_history_service.py`
- Create: `tests/unit/test_sessions_service.py`

- [ ] **Step 1: Write failing tests for `save_session_meta` and `list_sessions`**

```python
# tests/unit/test_sessions_service.py
"""Unit tests for RedisHistoryStore session-metadata methods."""
from __future__ import annotations

import pytest
from fakeredis import aioredis as fake_aioredis

from panvel_assistant.services.chat_history_service import RedisHistoryStore


@pytest.fixture
async def store():
    client = fake_aioredis.FakeRedis(decode_responses=True)
    s = RedisHistoryStore()
    s._client = client
    return s


@pytest.mark.asyncio
async def test_save_session_meta_stores_entry(store):
    await store.save_session_meta("sess-1", "Quais são as contraindicações?")
    sessions = await store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "sess-1"
    assert sessions[0]["title"] == "Quais são as contraindicações?"
    assert "created_at" in sessions[0]


@pytest.mark.asyncio
async def test_save_session_meta_idempotent(store):
    await store.save_session_meta("sess-1", "Primeira mensagem")
    await store.save_session_meta("sess-1", "Segunda — não deve sobrescrever")
    sessions = await store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["title"] == "Primeira mensagem"


@pytest.mark.asyncio
async def test_list_sessions_returns_newest_first(store):
    await store.save_session_meta("sess-a", "Mensagem A")
    await store.save_session_meta("sess-b", "Mensagem B")
    sessions = await store.list_sessions()
    assert sessions[0]["session_id"] == "sess-b"
    assert sessions[1]["session_id"] == "sess-a"


@pytest.mark.asyncio
async def test_list_sessions_empty(store):
    sessions = await store.list_sessions()
    assert sessions == []


@pytest.mark.asyncio
async def test_save_session_meta_truncates_title(store):
    long_title = "x" * 100
    await store.save_session_meta("sess-1", long_title)
    sessions = await store.list_sessions()
    assert len(sessions[0]["title"]) == 60
```

- [ ] **Step 2: Run to confirm failures**

```bash
cd /Users/nicolasfonseca/Desktop/rag-challenge
uv run pytest tests/unit/test_sessions_service.py -v
```

Expected: all 5 tests FAIL with `AttributeError: 'RedisHistoryStore' object has no attribute 'save_session_meta'`

- [ ] **Step 3: Implement `save_session_meta` and `list_sessions` in `RedisHistoryStore`**

Add these two methods to `RedisHistoryStore` (after the `client` property, before `rate_limit_check`). Also add `import time` and `from datetime import UTC, datetime` at the top of the file if not already present:

```python
# In RedisHistoryStore class, after the `client` property:

_SESSIONS_INDEX_KEY = "sessions:meta"

async def save_session_meta(self, session_id: str, title: str) -> None:
    """Persist session metadata on first turn only (idempotent via NX flag).

    Storage layout:
    - Sorted set ``sessions:meta``: score=Unix timestamp, member=session_id
    - Hash ``sessions:meta:{session_id}``: title, created_at, session_id fields
    Both keys share the chat history TTL so stale sessions auto-expire.
    """
    flag_key = f"sessions:meta:{session_id}"
    is_new = await self.client.set(flag_key, "1", nx=True, ex=self._settings.CHAT_HISTORY_TTL_SECONDS)  # type: ignore[misc]
    if not is_new:
        return
    now = datetime.now(UTC)
    async with self.client.pipeline(transaction=False) as pipe:
        pipe.zadd(self._SESSIONS_INDEX_KEY, {session_id: now.timestamp()})
        pipe.hset(
            f"sessions:data:{session_id}",
            mapping={
                "session_id": session_id,
                "title": title[:60],
                "created_at": now.isoformat(),
            },
        )
        pipe.expire(f"sessions:data:{session_id}", self._settings.CHAT_HISTORY_TTL_SECONDS)
        await pipe.execute()

async def list_sessions(self) -> list[dict]:
    """Return all known sessions sorted newest-first.

    Sessions whose data hash has expired (TTL elapsed) are skipped silently —
    the frontend handles 404 on history load by removing the stale entry.
    """
    session_ids: list[str] = await self.client.zrevrange(self._SESSIONS_INDEX_KEY, 0, -1)  # type: ignore[misc]
    result: list[dict] = []
    for sid in session_ids:
        data = await self.client.hgetall(f"sessions:data:{sid}")  # type: ignore[misc]
        if data:
            result.append(data)
    return result
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/unit/test_sessions_service.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/panvel_assistant/services/chat_history_service.py tests/unit/test_sessions_service.py
git commit -m "feat(backend): add session metadata storage to RedisHistoryStore"
```

---

## Task 2: Backend — Citations persistence in `AssistantService`

**Files:**
- Modify: `src/panvel_assistant/assistant/assistant_service.py`
- Modify: `tests/unit/test_assistant_service.py` (add citation assertions)

- [ ] **Step 1: Check the existing test to understand the test fixture**

```bash
uv run pytest tests/unit/test_assistant_service.py -v --collect-only
```

Look at how `AssistantService` tests mock `history` and what fixtures exist.

- [ ] **Step 2: Write a failing test asserting citations end up in `additional_kwargs`**

Open `tests/unit/test_assistant_service.py` and add this test at the end of the file:

```python
@pytest.mark.asyncio
async def test_handle_turn_persists_citations_in_additional_kwargs(monkeypatch):
    """Citations from a sources event must be saved in AIMessage.additional_kwargs."""
    from langchain_core.messages import AIMessage, HumanMessage
    from panvel_assistant.assistant.assistant_service import AssistantService
    from panvel_assistant.models.chat import ChatRequest

    captured_messages: list = []

    class FakeHistory:
        async def aget_messages(self):
            return []
        async def aadd_messages(self, messages):
            captured_messages.extend(messages)

    class FakeStore:
        def get_session_history(self, _):
            return FakeHistory()
        def register_pending(self, task, *, session_id=None):
            import asyncio
            asyncio.get_event_loop().run_until_complete(task)
        async def save_session_meta(self, session_id, title):
            pass

    fake_citations = [{"bula_id": "b1", "med_name": "Ritalina", "snippet": "..."}]

    async def fake_stream_with_tools(messages):
        yield {"type": "token", "text": "Resposta."}
        yield {"type": "sources", "citations": fake_citations}
        yield {"type": "done", "tokens_in": 10, "tokens_out": 5}

    svc = AssistantService.__new__(AssistantService)
    svc._history = FakeStore()
    monkeypatch.setattr(svc, "stream_with_tools", fake_stream_with_tools)

    req = ChatRequest(session_id="s1", message="Ritalina contraindicações")
    frames = [f async for f in svc.handle_turn(req)]

    ai_msgs = [m for m in captured_messages if isinstance(m, AIMessage)]
    assert ai_msgs, "no AIMessage persisted"
    assert ai_msgs[0].additional_kwargs.get("citations") == fake_citations
```

- [ ] **Step 3: Run to confirm failure**

```bash
uv run pytest tests/unit/test_assistant_service.py::test_handle_turn_persists_citations_in_additional_kwargs -v
```

Expected: FAIL — assertion error (citations not in `additional_kwargs`)

- [ ] **Step 4: Implement citation tracking in `handle_turn`**

In `src/panvel_assistant/assistant/assistant_service.py`, in the `handle_turn` method:

**Line ~378** — alongside `text_chunks: list[str] = []`, add:
```python
_collected_citations: list[dict] = []
```

**Lines ~403–407** — in the `sources` handler, add capture before the `yield`:
```python
elif event_type == "sources":
    _collected_citations = event["citations"]   # <-- add this line
    trace_service.set_citations(event["citations"])
    yield encode_event(
        "sources", {"citations": event["citations"]}
    )
```

**Line ~496** — change the `AIMessage` construction from:
```python
assistant_msg = AIMessage(content="".join(text_chunks))
```
to:
```python
assistant_msg = AIMessage(
    content="".join(text_chunks),
    additional_kwargs={"citations": _collected_citations} if _collected_citations else {},
)
```

**Line ~502** (after `_spawn_persist` call) — call `save_session_meta` fire-and-forget:
```python
self._spawn_persist(
    history,
    [user_msg, assistant_msg],
    step="persist",
    session_id=req.session_id,
)
# Fire-and-forget: store session metadata (idempotent, only writes on first turn)
_title = req.message[:60]
try:
    loop = asyncio.get_running_loop()
    loop.create_task(
        self._history.save_session_meta(req.session_id, _title),
        name=f"save_meta:{req.session_id}",
    )
except RuntimeError:
    pass
yield encode_event("done", {"session_id": req.session_id})
```

- [ ] **Step 5: Run the new test and full unit suite**

```bash
uv run pytest tests/unit/test_assistant_service.py -v
```

Expected: all tests PASS including the new citation test

- [ ] **Step 6: Commit**

```bash
git add src/panvel_assistant/assistant/assistant_service.py tests/unit/test_assistant_service.py
git commit -m "feat(backend): persist citations and session metadata on chat turn"
```

---

## Task 3: Backend — Sessions routes

**Files:**
- Create: `src/panvel_assistant/routes/sessions.py`
- Modify: `src/panvel_assistant/main.py`
- Create: `tests/integration/test_sessions_routes.py`

- [ ] **Step 1: Write failing integration tests**

```python
# tests/integration/test_sessions_routes.py
"""Integration tests for GET /sessions and GET /sessions/{session_id}/history."""
from __future__ import annotations

import pytest
import httpx
from asgi_lifespan import LifespanManager
from fakeredis import aioredis as fake_aioredis
from langchain_core.messages import AIMessage, HumanMessage

from panvel_assistant.main import app
from panvel_assistant.services.chat_history_service import (
    RedisChatMessageHistory,
    history_store,
)


@pytest.fixture
def fake_redis():
    client = fake_aioredis.FakeRedis(decode_responses=True)
    original = history_store._client
    history_store._client = client

    original_connect = history_store.connect
    original_disconnect = history_store.disconnect

    async def noop_connect(): history_store._client = client
    async def noop_disconnect(): history_store._client = None

    history_store.connect = noop_connect   # type: ignore[method-assign]
    history_store.disconnect = noop_disconnect  # type: ignore[method-assign]
    try:
        yield client
    finally:
        history_store._client = original
        history_store.connect = original_connect  # type: ignore[method-assign]
        history_store.disconnect = original_disconnect  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_list_sessions_empty(fake_redis):
    async with LifespanManager(app):
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.get("/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_sessions_returns_metadata(fake_redis):
    await history_store.save_session_meta("sess-x", "Posologia do Advil")
    async with LifespanManager(app):
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.get("/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["session_id"] == "sess-x"
    assert data[0]["title"] == "Posologia do Advil"


@pytest.mark.asyncio
async def test_get_history_not_found(fake_redis):
    async with LifespanManager(app):
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.get("/sessions/nonexistent/history")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "session_not_found"


@pytest.mark.asyncio
async def test_get_history_returns_messages(fake_redis):
    sess = RedisChatMessageHistory("hist-1", history_store.client, ttl_seconds=300)
    await sess.aadd_messages([
        HumanMessage(content="Qual a posologia?"),
        AIMessage(
            content="A posologia é X.",
            additional_kwargs={"citations": [{"bula_id": "b1", "med_name": "Advil", "snippet": "..."}]},
        ),
    ])
    async with LifespanManager(app):
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.get("/sessions/hist-1/history")
    assert resp.status_code == 200
    msgs = resp.json()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Qual a posologia?"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["citations"][0]["med_name"] == "Advil"
```

- [ ] **Step 2: Run to confirm failures (route doesn't exist yet)**

```bash
uv run pytest tests/integration/test_sessions_routes.py -v
```

Expected: all 4 tests FAIL with 404 (route not registered)

- [ ] **Step 3: Create `src/panvel_assistant/routes/sessions.py`**

```python
"""Sessions routes — list known sessions and load message history."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from panvel_assistant.services.chat_history_service import (
    RedisHistoryStore,
    get_history_store,
)
from panvel_assistant.utils.handle_errors import handle_errors

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("")
@handle_errors
async def list_sessions(store: RedisHistoryStore = Depends(get_history_store)):
    """Return all known sessions ordered newest-first."""
    return await store.list_sessions()


@router.get("/{session_id}/history")
@handle_errors
async def get_session_history(
    session_id: str,
    store: RedisHistoryStore = Depends(get_history_store),
):
    """Return the full message history for a session as frontend ChatMessage objects."""
    history = store.get_session_history(session_id)
    messages = await history.aget_messages()
    if not messages:
        raise HTTPException(
            status_code=404,
            detail={"code": "session_not_found", "message": f"session '{session_id}' not found or expired"},
        )
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({
                "id": str(uuid4()),
                "role": "user",
                "content": msg.content,
                "citations": None,
                "status": "done",
            })
        elif isinstance(msg, AIMessage):
            citations = msg.additional_kwargs.get("citations") or None
            result.append({
                "id": str(uuid4()),
                "role": "assistant",
                "content": msg.content,
                "citations": citations,
                "status": "done",
            })
    return result
```

- [ ] **Step 4: Register the router in `main.py`**

Add to the imports block:
```python
from panvel_assistant.routes.sessions import router as sessions_router
```

Add to the `include_router` calls (after the existing three):
```python
app.include_router(sessions_router)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
uv run pytest tests/integration/test_sessions_routes.py -v
```

Expected: 4 tests PASS

- [ ] **Step 6: Run the full test suite to check for regressions**

```bash
uv run pytest tests/ -v --ignore=tests/integration/test_ingestion.py --ignore=tests/integration/test_retrieval.py --ignore=tests/integration/test_chat_with_rag.py -x
```

Expected: all tests PASS (ingestion/retrieval/rag tests require live Qdrant — skip them)

- [ ] **Step 7: Commit**

```bash
git add src/panvel_assistant/routes/sessions.py src/panvel_assistant/main.py tests/integration/test_sessions_routes.py
git commit -m "feat(backend): add GET /sessions and GET /sessions/{id}/history routes"
```

---

## Task 4: Frontend scaffold

**Files:**
- Create: `frontend/` (entire Vite project)
- Create: `frontend/.env.example`

- [ ] **Step 1: Scaffold the Vite project**

```bash
cd /Users/nicolasfonseca/Desktop/rag-challenge
npm create vite@latest frontend -- --template react-ts
```

- [ ] **Step 2: Install production dependencies**

```bash
cd frontend
npm install
npm install react-markdown date-fns sonner
```

- [ ] **Step 3: Install and configure Tailwind**

```bash
npm install -D tailwindcss @tailwindcss/typography postcss autoprefixer
npx tailwindcss init -p
```

Update `tailwind.config.js`:

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: { extend: {} },
  plugins: [require("@tailwindcss/typography")],
};
```

Replace `src/index.css` with:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  color-scheme: dark;
}

body {
  @apply bg-zinc-950 text-zinc-100 h-screen overflow-hidden;
}

* {
  @apply border-zinc-800;
}
```

- [ ] **Step 4: Add `dark` class to `<html>` in `index.html`**

In `index.html`, change `<html lang="en">` to `<html lang="pt-BR" class="dark">`.

- [ ] **Step 5: Init shadcn/ui**

```bash
npx shadcn@latest init
```

When prompted:
- Style: `Default`
- Base color: `Zinc`
- CSS variables: `Yes`

Then install required components:

```bash
npx shadcn@latest add button input textarea badge separator toast popover scroll-area
```

- [ ] **Step 6: Create `.env.example`**

```bash
# frontend/.env.example
VITE_API_URL=http://localhost:8000
```

Copy it to `.env`:

```bash
cp .env.example .env
```

- [ ] **Step 7: Verify the dev server starts**

```bash
npm run dev
```

Open `http://localhost:5173` — should show the default Vite React page with no errors.

- [ ] **Step 8: Remove default boilerplate**

Replace `src/App.tsx` with:

```tsx
export default function App() {
  return <div className="flex h-screen bg-zinc-950 text-zinc-100">Loading…</div>;
}
```

Replace `src/App.css` with an empty file (or delete it and remove the import from `App.tsx`).

- [ ] **Step 9: Commit**

```bash
cd /Users/nicolasfonseca/Desktop/rag-challenge
git add frontend/
git commit -m "feat(frontend): scaffold Vite + React + Tailwind + shadcn/ui"
```

---

## Task 5: Types + API client

**Files:**
- Create: `frontend/src/types/chat.ts`
- Create: `frontend/src/lib/api.ts`

- [ ] **Step 1: Create `src/types/chat.ts`**

```typescript
// frontend/src/types/chat.ts

export type MessageRole = "user" | "assistant";
export type MessageStatus = "done" | "streaming" | "error";
export type SSEEventType =
  | "trace_id"
  | "token"
  | "tool_call"
  | "tool_result"
  | "sources"
  | "done"
  | "error";

export interface Citation {
  bula_id: string;
  med_name: string;
  med_variant?: string;
  section_canonical?: string;
  section_label: string;
  source_page?: number;
  snippet: string;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  citations?: Citation[];
  currentTool?: string;
  status: MessageStatus;
}

export interface Session {
  session_id: string;
  title: string;
  created_at: string;
}

export interface SSEFrame {
  event: SSEEventType;
  data: string;
}

export interface ApiErrorEnvelope {
  error: {
    code: string;
    message: string;
    status_code: number;
    trace_id?: string;
  };
}
```

- [ ] **Step 2: Create `src/lib/api.ts`**

```typescript
// frontend/src/lib/api.ts
import type { ChatMessage, Session, SSEFrame } from "../types/chat";

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

// --- SSE parser ---

export function parseSSEFrame(raw: string): SSEFrame | null {
  const lines = raw.split("\n");
  let event = "message";
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("event: ")) event = line.slice(7).trim();
    else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
  }
  if (!dataLines.length) return null;
  return { event: event as SSEFrame["event"], data: dataLines.join("\n") };
}

// --- Error helpers ---

export function httpErrorMessage(status: number): string {
  if (status === 429) return "Muitas mensagens. Aguarde um momento.";
  if (status === 409) return "Aguarde a resposta anterior terminar.";
  if (status === 422) return "Mensagem inválida. Verifique o conteúdo.";
  if (status === 413) return "Mensagem muito longa.";
  return "Erro de conexão. Tente novamente.";
}

// --- Sessions API ---

export async function fetchSessions(): Promise<Session[]> {
  try {
    const res = await fetch(`${API_URL}/sessions`);
    if (!res.ok) return [];
    return res.json() as Promise<Session[]>;
  } catch {
    return [];
  }
}

export async function fetchHistory(sessionId: string): Promise<ChatMessage[]> {
  const res = await fetch(`${API_URL}/sessions/${encodeURIComponent(sessionId)}/history`);
  if (res.status === 404) throw Object.assign(new Error("not_found"), { status: 404 });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<ChatMessage[]>;
}

// --- SSE streaming ---

export async function* streamChat(
  sessionId: string,
  message: string,
  signal: AbortSignal,
): AsyncGenerator<SSEFrame> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal,
  });

  if (!res.ok) {
    // Pre-stream HTTP error: parse JSON error envelope
    const body = await res.json().catch(() => ({}));
    const msg = (body as { error?: { message?: string } }).error?.message
      ?? httpErrorMessage(res.status);
    throw Object.assign(new Error(msg), { status: res.status, body });
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: false });
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      if (!part.trim()) continue;
      const frame = parseSSEFrame(part);
      if (frame) yield frame;
    }
  }
}
```

- [ ] **Step 3: Write a unit test for `parseSSEFrame`**

Create `frontend/src/lib/__tests__/api.test.ts`:

```typescript
// frontend/src/lib/__tests__/api.test.ts
import { describe, expect, it } from "vitest";
import { parseSSEFrame } from "../api";

describe("parseSSEFrame", () => {
  it("parses a token frame (plain text data)", () => {
    const raw = "event: token\ndata: Olá, como";
    const frame = parseSSEFrame(raw);
    expect(frame).toEqual({ event: "token", data: "Olá, como" });
  });

  it("joins multi-line data fields with newline", () => {
    const raw = "event: token\ndata: linha 1\ndata: linha 2";
    const frame = parseSSEFrame(raw);
    expect(frame).toEqual({ event: "token", data: "linha 1\nlinha 2" });
  });

  it("parses a JSON-carrying event", () => {
    const payload = JSON.stringify({ session_id: "s1" });
    const raw = `event: done\ndata: ${payload}`;
    const frame = parseSSEFrame(raw);
    expect(frame?.event).toBe("done");
    expect(JSON.parse(frame!.data)).toEqual({ session_id: "s1" });
  });

  it("returns null for empty frame", () => {
    expect(parseSSEFrame("")).toBeNull();
    expect(parseSSEFrame("event: done")).toBeNull();
  });
});
```

Install vitest if not present (Vite template includes it):

```bash
cd frontend && npm run test -- --run src/lib/__tests__/api.test.ts
```

Expected: 4 tests PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/nicolasfonseca/Desktop/rag-challenge
git add frontend/src/types/ frontend/src/lib/
git commit -m "feat(frontend): add types, API client, and SSE parser"
```

---

## Task 6: Hooks — `useChatStream` and `useSessionList`

**Files:**
- Create: `frontend/src/hooks/useChatStream.ts`
- Create: `frontend/src/hooks/useSessionList.ts`

- [ ] **Step 1: Create `src/hooks/useChatStream.ts`**

```typescript
// frontend/src/hooks/useChatStream.ts
import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import { fetchHistory, httpErrorMessage, streamChat } from "../lib/api";
import type { ChatMessage } from "../types/chat";

const TOOL_LABELS: Record<string, string> = {
  buscar_filiais: "Buscando filiais...",
  detalhes_filial: "Consultando filial...",
  listar_cidades_atendidas: "Buscando cidades...",
  search_bulas: "Consultando bula...",
};

function toolLabel(name: string): string {
  return TOOL_LABELS[name] ?? "Pesquisando...";
}

function applyFrame(
  messages: ChatMessage[],
  assistantId: string,
  updater: (m: ChatMessage) => ChatMessage,
): ChatMessage[] {
  return messages.map((m) => (m.id === assistantId ? updater(m) : m));
}

export function useChatStream() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback(async (sessionId: string, text: string): Promise<void> => {
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      status: "done",
    };
    const assistantId = crypto.randomUUID();
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      status: "streaming",
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      for await (const frame of streamChat(sessionId, text, controller.signal)) {
        if (frame.event === "token") {
          setMessages((prev) =>
            applyFrame(prev, assistantId, (m) => ({
              ...m,
              content: m.content + frame.data,
            })),
          );
        } else if (frame.event === "tool_call") {
          const { name } = JSON.parse(frame.data) as { name: string };
          setMessages((prev) =>
            applyFrame(prev, assistantId, (m) => ({
              ...m,
              currentTool: toolLabel(name),
            })),
          );
        } else if (frame.event === "tool_result") {
          setMessages((prev) =>
            applyFrame(prev, assistantId, (m) => ({
              ...m,
              currentTool: undefined,
            })),
          );
        } else if (frame.event === "sources") {
          const { citations } = JSON.parse(frame.data);
          setMessages((prev) =>
            applyFrame(prev, assistantId, (m) => ({ ...m, citations })),
          );
        } else if (frame.event === "done") {
          setMessages((prev) =>
            applyFrame(prev, assistantId, (m) => ({
              ...m,
              status: "done",
              currentTool: undefined,
            })),
          );
        } else if (frame.event === "error") {
          const { message } = JSON.parse(frame.data) as { message: string };
          setMessages((prev) =>
            applyFrame(prev, assistantId, (m) => ({
              ...m,
              status: "error",
              content: m.content + `\n\n_${message}_`,
              currentTool: undefined,
            })),
          );
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") {
        setMessages((prev) =>
          applyFrame(prev, assistantId, (m) => ({
            ...m,
            status: "done",
            content: m.content
              ? m.content + "\n\n_Resposta interrompida._"
              : "_Resposta interrompida._",
            currentTool: undefined,
          })),
        );
      } else {
        const errObj = err as { status?: number; body?: unknown };
        const msg =
          errObj.status !== undefined
            ? httpErrorMessage(errObj.status)
            : "Erro de conexão. Tente novamente.";
        toast.error(msg);
        setMessages((prev) => prev.filter((m) => m.id !== assistantId));
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  }, []);

  const stop = useCallback((): void => {
    abortRef.current?.abort();
  }, []);

  const loadHistory = useCallback(async (sessionId: string): Promise<boolean> => {
    try {
      const history = await fetchHistory(sessionId);
      setMessages(history);
      return true;
    } catch (err: unknown) {
      const errObj = err as { status?: number };
      if (errObj.status === 404) return false;
      toast.error("Erro ao carregar histórico.");
      return false;
    }
  }, []);

  const reset = useCallback((): void => {
    setMessages([]);
  }, []);

  return { messages, isStreaming, send, stop, loadHistory, reset };
}
```

- [ ] **Step 2: Create `src/hooks/useSessionList.ts`**

```typescript
// frontend/src/hooks/useSessionList.ts
import { useCallback, useEffect, useState } from "react";
import { fetchSessions } from "../lib/api";
import type { Session } from "../types/chat";

export function useSessionList() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async (): Promise<void> => {
    const data = await fetchSessions();
    setSessions(data);
    setLoading(false);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { sessions, loading, refresh };
}
```

- [ ] **Step 3: Commit**

```bash
cd /Users/nicolasfonseca/Desktop/rag-challenge
git add frontend/src/hooks/
git commit -m "feat(frontend): add useChatStream and useSessionList hooks"
```

---

## Task 7: App Context + Layout + Sidebar

**Files:**
- Create: `frontend/src/context/AppContext.tsx`
- Create: `frontend/src/components/layout/ChatLayout.tsx`
- Create: `frontend/src/components/layout/Sidebar.tsx`

- [ ] **Step 1: Create `src/context/AppContext.tsx`**

```tsx
// frontend/src/context/AppContext.tsx
import { createContext, useCallback, useContext, useState } from "react";
import type { ReactNode } from "react";

interface AppContextValue {
  activeSessionId: string | null;
  setActiveSessionId: (id: string | null) => void;
  sidebarOpen: boolean;
  toggleSidebar: () => void;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const toggleSidebar = useCallback(() => setSidebarOpen((v) => !v), []);

  return (
    <AppContext.Provider value={{ activeSessionId, setActiveSessionId, sidebarOpen, toggleSidebar }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppContext must be used within AppProvider");
  return ctx;
}
```

- [ ] **Step 2: Create `src/components/layout/ChatLayout.tsx`**

```tsx
// frontend/src/components/layout/ChatLayout.tsx
import { Toaster } from "sonner";
import { useAppContext } from "../../context/AppContext";
import { ChatView } from "./ChatView";
import { Sidebar } from "./Sidebar";

export function ChatLayout() {
  const { sidebarOpen } = useAppContext();

  return (
    <div className="flex h-screen bg-zinc-950 overflow-hidden">
      <div
        className={`transition-all duration-200 ease-in-out shrink-0 ${
          sidebarOpen ? "w-64" : "w-0"
        } overflow-hidden`}
      >
        <Sidebar />
      </div>
      <main className="flex-1 flex flex-col min-w-0">
        <ChatView />
      </main>
      <Toaster theme="dark" position="top-right" />
    </div>
  );
}
```

- [ ] **Step 3: Create `src/components/layout/Sidebar.tsx`**

```tsx
// frontend/src/components/layout/Sidebar.tsx
import { formatDistanceToNow } from "date-fns";
import { ptBR } from "date-fns/locale";
import { PenSquare } from "lucide-react";
import { useAppContext } from "../../context/AppContext";
import { useSessionList } from "../../hooks/useSessionList";
import type { Session } from "../../types/chat";

function SessionItem({
  session,
  active,
  onClick,
}: {
  session: Session;
  active: boolean;
  onClick: () => void;
}) {
  const relativeTime = formatDistanceToNow(new Date(session.created_at), {
    addSuffix: true,
    locale: ptBR,
  });

  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors ${
        active
          ? "bg-zinc-700 text-zinc-100"
          : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
      }`}
    >
      <p className="truncate font-medium text-zinc-200">{session.title || "Nova conversa"}</p>
      <p className="text-xs text-zinc-500 mt-0.5">{relativeTime}</p>
    </button>
  );
}

export function Sidebar() {
  const { activeSessionId, setActiveSessionId } = useAppContext();
  const { sessions } = useSessionList();

  function handleNewChat() {
    setActiveSessionId(crypto.randomUUID());
  }

  return (
    <div className="h-full bg-zinc-900 flex flex-col w-64 border-r border-zinc-800">
      <div className="p-3 border-b border-zinc-800">
        <button
          onClick={handleNewChat}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 transition-colors"
        >
          <PenSquare size={14} />
          Nova conversa
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
        {sessions.map((s: Session) => (
          <SessionItem
            key={s.session_id}
            session={s}
            active={s.session_id === activeSessionId}
            onClick={() => setActiveSessionId(s.session_id)}
          />
        ))}
      </div>
    </div>
  );
}
```

Install the `lucide-react` package (shadcn uses it internally, but install explicitly):

```bash
cd frontend && npm install lucide-react date-fns
```

- [ ] **Step 4: Commit**

```bash
cd /Users/nicolasfonseca/Desktop/rag-challenge
git add frontend/src/context/ frontend/src/components/layout/ChatLayout.tsx frontend/src/components/layout/Sidebar.tsx
git commit -m "feat(frontend): add AppContext, ChatLayout, and Sidebar"
```

---

## Task 8: Message components

**Files:**
- Create: `frontend/src/components/chat/UserMessage.tsx`
- Create: `frontend/src/components/chat/ThinkingIndicator.tsx`
- Create: `frontend/src/components/chat/CitationChips.tsx`
- Create: `frontend/src/components/chat/AssistantMessage.tsx`
- Create: `frontend/src/components/chat/EmptyState.tsx`
- Create: `frontend/src/components/chat/MessageList.tsx`

- [ ] **Step 1: Create `UserMessage.tsx`**

```tsx
// frontend/src/components/chat/UserMessage.tsx
export function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex justify-end px-4">
      <div className="max-w-[70%] bg-zinc-800 text-zinc-100 rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap break-words">
        {content}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create `ThinkingIndicator.tsx`**

```tsx
// frontend/src/components/chat/ThinkingIndicator.tsx
export function ThinkingIndicator({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs text-zinc-500 mb-1">
      <span className="flex gap-0.5">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-1 h-1 rounded-full bg-zinc-500 animate-bounce"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </span>
      {label}
    </div>
  );
}
```

- [ ] **Step 3: Create `CitationChips.tsx`**

```tsx
// frontend/src/components/chat/CitationChips.tsx
import { useState } from "react";
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import type { Citation } from "../../types/chat";

function CitationPopover({ citation, index }: { citation: Citation; index: number }) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button className="inline-flex items-center justify-center w-5 h-5 text-[10px] font-medium rounded bg-zinc-700 text-zinc-300 hover:bg-zinc-600 hover:text-zinc-100 transition-colors cursor-pointer">
          {index + 1}
        </button>
      </PopoverTrigger>
      <PopoverContent
        side="top"
        className="w-72 bg-zinc-800 border-zinc-700 text-zinc-200 p-3 text-xs"
      >
        <p className="font-semibold text-zinc-100 truncate">
          {citation.med_name}
          {citation.med_variant ? ` — ${citation.med_variant}` : ""}
        </p>
        <p className="text-zinc-400 mt-0.5 mb-2">{citation.section_label}</p>
        <p className="text-zinc-300 leading-relaxed line-clamp-5">
          {citation.snippet.length > 200
            ? citation.snippet.slice(0, 200) + "…"
            : citation.snippet}
        </p>
        {citation.source_page && (
          <p className="text-zinc-500 mt-1">Página {citation.source_page}</p>
        )}
      </PopoverContent>
    </Popover>
  );
}

export function CitationChips({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <div className="flex flex-wrap gap-1 mt-2">
      {citations.map((c, i) => (
        <CitationPopover key={`${c.bula_id}-${i}`} citation={c} index={i} />
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Create `AssistantMessage.tsx`**

```tsx
// frontend/src/components/chat/AssistantMessage.tsx
import ReactMarkdown from "react-markdown";
import type { ChatMessage } from "../../types/chat";
import { CitationChips } from "./CitationChips";
import { ThinkingIndicator } from "./ThinkingIndicator";

export function AssistantMessage({ message }: { message: ChatMessage }) {
  const { content, citations, currentTool, status } = message;

  return (
    <div className="flex justify-start px-4">
      <div className="max-w-[80%]">
        {currentTool && <ThinkingIndicator label={currentTool} />}
        {content && (
          <div className="prose prose-invert prose-sm max-w-none text-zinc-200 leading-relaxed">
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
        )}
        {!content && status === "streaming" && !currentTool && (
          <span className="inline-block w-2 h-4 bg-zinc-400 animate-pulse rounded-sm" />
        )}
        {status === "done" && citations && citations.length > 0 && (
          <CitationChips citations={citations} />
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Create `EmptyState.tsx`**

```tsx
// frontend/src/components/chat/EmptyState.tsx
import { useAppContext } from "../../context/AppContext";

const SUGGESTED = [
  "Quais são as contraindicações do Ritalina?",
  "Qual a posologia do Advil para adultos?",
  "Pode tomar Rivotril com álcool?",
];

export function EmptyState() {
  const { setActiveSessionId } = useAppContext();

  function handleSuggestion() {
    setActiveSessionId(crypto.randomUUID());
  }

  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 px-4 text-center">
      <div>
        <h1 className="text-2xl font-semibold text-zinc-100">Assistente Panvel</h1>
        <p className="text-zinc-500 mt-1 text-sm">Como posso ajudar?</p>
      </div>
      <div className="flex flex-col gap-2 w-full max-w-sm">
        {SUGGESTED.map((q) => (
          <button
            key={q}
            onClick={handleSuggestion}
            className="text-left px-4 py-2.5 rounded-lg border border-zinc-800 text-sm text-zinc-400 hover:border-zinc-600 hover:text-zinc-200 transition-colors"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Create `MessageList.tsx`**

```tsx
// frontend/src/components/chat/MessageList.tsx
import { useEffect, useRef } from "react";
import type { ChatMessage } from "../../types/chat";
import { AssistantMessage } from "./AssistantMessage";
import { UserMessage } from "./UserMessage";

export function MessageList({ messages }: { messages: ChatMessage[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto py-6 space-y-4">
      {messages.map((m) =>
        m.role === "user" ? (
          <UserMessage key={m.id} content={m.content} />
        ) : (
          <AssistantMessage key={m.id} message={m} />
        ),
      )}
      <div ref={bottomRef} />
    </div>
  );
}
```

- [ ] **Step 7: Commit**

```bash
cd /Users/nicolasfonseca/Desktop/rag-challenge
git add frontend/src/components/chat/
git commit -m "feat(frontend): add message components (UserMessage, AssistantMessage, ThinkingIndicator, CitationChips, EmptyState, MessageList)"
```

---

## Task 9: ChatInput + ChatView + App wiring

**Files:**
- Create: `frontend/src/components/chat/ChatInput.tsx`
- Create: `frontend/src/components/layout/ChatView.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/main.tsx`

- [ ] **Step 1: Create `src/components/chat/ChatInput.tsx`**

```tsx
// frontend/src/components/chat/ChatInput.tsx
import { ArrowUp, Square } from "lucide-react";
import { type KeyboardEvent, useCallback, useRef } from "react";

interface ChatInputProps {
  isStreaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}

export function ChatInput({ isStreaming, onSend, onStop }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleInput = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }, []);

  const handleSend = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    const text = el.value.trim();
    if (!text || isStreaming) return;
    el.value = "";
    el.style.height = "auto";
    onSend(text);
  }, [isStreaming, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div className="p-4 border-t border-zinc-800 bg-zinc-950">
      <div className="flex items-end gap-2 max-w-3xl mx-auto bg-zinc-900 border border-zinc-700 rounded-xl px-3 py-2 focus-within:border-zinc-500 transition-colors">
        <textarea
          ref={textareaRef}
          onInput={handleInput}
          onKeyDown={handleKeyDown}
          placeholder="Mensagem..."
          rows={1}
          className="flex-1 resize-none bg-transparent text-sm text-zinc-100 placeholder:text-zinc-500 outline-none leading-relaxed min-h-[24px] max-h-[140px] py-0.5"
        />
        <button
          onClick={isStreaming ? onStop : handleSend}
          className={`shrink-0 flex items-center justify-center w-7 h-7 rounded-lg transition-colors ${
            isStreaming
              ? "bg-zinc-600 hover:bg-zinc-500 text-zinc-200"
              : "bg-zinc-200 hover:bg-white text-zinc-900 disabled:opacity-40"
          }`}
        >
          {isStreaming ? <Square size={12} /> : <ArrowUp size={14} />}
        </button>
      </div>
      <p className="text-center text-[10px] text-zinc-600 mt-2">
        As respostas são geradas por IA e podem conter erros. Consulte sempre um profissional de saúde.
      </p>
    </div>
  );
}
```

- [ ] **Step 2: Create `src/components/layout/ChatView.tsx`**

```tsx
// frontend/src/components/layout/ChatView.tsx
import { Menu } from "lucide-react";
import { useEffect } from "react";
import { toast } from "sonner";
import { useAppContext } from "../../context/AppContext";
import { useChatStream } from "../../hooks/useChatStream";
import { useSessionList } from "../../hooks/useSessionList";
import { ChatInput } from "../chat/ChatInput";
import { EmptyState } from "../chat/EmptyState";
import { MessageList } from "../chat/MessageList";

export function ChatView() {
  const { activeSessionId, toggleSidebar } = useAppContext();
  const { messages, isStreaming, send, stop, loadHistory, reset } = useChatStream();
  const { refresh: refreshSessions } = useSessionList();

  // Load history when switching sessions
  useEffect(() => {
    if (!activeSessionId) {
      reset();
      return;
    }
    void (async () => {
      reset();
      const ok = await loadHistory(activeSessionId);
      if (!ok) {
        // Session 404 — treat as a fresh new chat (history expired)
        reset();
      }
    })();
  }, [activeSessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  async function handleSend(text: string) {
    if (!activeSessionId) return;
    await send(activeSessionId, text);
    void refreshSessions();
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-zinc-800 shrink-0">
        <button
          onClick={toggleSidebar}
          className="text-zinc-500 hover:text-zinc-200 transition-colors p-1 rounded"
          aria-label="Toggle sidebar"
        >
          <Menu size={18} />
        </button>
      </div>

      {/* Body */}
      {!activeSessionId ? (
        <EmptyState />
      ) : (
        <>
          <MessageList messages={messages} />
          <ChatInput
            isStreaming={isStreaming}
            onSend={handleSend}
            onStop={stop}
          />
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Update `src/App.tsx`**

```tsx
// frontend/src/App.tsx
import { AppProvider } from "./context/AppContext";
import { ChatLayout } from "./components/layout/ChatLayout";

export default function App() {
  return (
    <AppProvider>
      <ChatLayout />
    </AppProvider>
  );
}
```

- [ ] **Step 4: Update `src/main.tsx`** to remove the default `StrictMode` import of the old CSS

```tsx
// frontend/src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

- [ ] **Step 5: Commit**

```bash
cd /Users/nicolasfonseca/Desktop/rag-challenge
git add frontend/src/
git commit -m "feat(frontend): wire ChatInput, ChatView, and App — full UI complete"
```

---

## Task 10: End-to-end verification

- [ ] **Step 1: Start the backend**

```bash
cd /Users/nicolasfonseca/Desktop/rag-challenge
docker compose up -d redis qdrant
uv run uvicorn panvel_assistant.main:app --reload --port 8000
```

- [ ] **Step 2: Start the frontend**

```bash
cd frontend && npm run dev
```

Open `http://localhost:5173`.

- [ ] **Step 3: Verify empty state**

Expected: dark page with "Assistente Panvel", subtitle, and 3 suggested question chips. No errors in browser console.

- [ ] **Step 4: Verify new chat + streaming**

Click a suggested question chip (or "Nova conversa", then type "oi" and press Enter).

Expected:
- User message appears immediately on the right
- Assistant message appears on the left, tokens stream in character by character
- Send button becomes a Stop button (square icon) while streaming
- After `done`, the input unlocks

- [ ] **Step 5: Verify tool call indicator**

Send a pharmacy question: "Tem filiais em Porto Alegre?"

Expected: "Buscando filiais..." badge appears above the streaming text, then disappears when the response completes.

- [ ] **Step 6: Verify citations**

Send a RAG question: "Quais são as contraindicações do Ritalina?"

Expected: numbered chips `[1]` `[2]` appear below the assistant text after streaming ends. Clicking a chip opens a popover with the medication name, section, and snippet.

- [ ] **Step 7: Verify sidebar + history**

Send a second message in the same session. Open `http://localhost:8000/sessions` in a new tab — should see the session in the JSON response.

Refresh the frontend. The session should appear in the sidebar. Click another session (or create one). Click back to the first session — history should reload (both messages visible).

- [ ] **Step 8: Verify stop button**

During a long response, click Stop. Expected: partial response kept, "_Resposta interrompida._" appended.

- [ ] **Step 9: Verify sidebar collapse**

Click the hamburger (Menu) icon. Sidebar should slide away smoothly and the chat column expands to full width. Click again to restore.

- [ ] **Step 10: Final commit with any fixes applied during verification**

```bash
cd /Users/nicolasfonseca/Desktop/rag-challenge
git add -A
git commit -m "fix(frontend): verification fixes after e2e testing"
```

---

## Self-Review

**Spec coverage check:**
- Dark mode only → `index.css` color-scheme + `class="dark"` on `<html>` ✓
- Skip debug MVP → full UI from Task 4 ✓
- Collapsible sidebar (hamburger) → `ChatLayout` + `toggleSidebar` + `sidebarOpen` ✓
- Chat list from backend → `GET /sessions` + `useSessionList` ✓
- Load history on click → `loadHistory` in `ChatView` useEffect ✓
- SSE streaming → `streamChat` generator + `useChatStream` ✓
- Citations inline chips → `CitationChips` + `CitationPopover` ✓
- Tool call transient indicators → `ThinkingIndicator` + `currentTool` state cleared on `tool_result`/`done` ✓
- Error handling: 429/409/422/network → `httpErrorMessage` + toast ✓
- Mid-stream error → inline `_${message}_` appended to content ✓
- Stop button → `AbortController` ✓
- Session TTL/404 → `loadHistory` returns `false` + `reset()` ✓
- Citations in history → `AIMessage.additional_kwargs["citations"]` + history route serializes them ✓
- `save_session_meta` called after successful persist → fire-and-forget `create_task` in `handle_turn` ✓

**Type consistency check:**
- `Citation` fields match backend `Citation` Pydantic model (`bula_id`, `med_name`, `med_variant?`, `section_label`, `source_page?`, `snippet`) ✓
- `Session` fields match `save_session_meta`/`list_sessions` output (`session_id`, `title`, `created_at`) ✓
- `ChatMessage` `id`, `role`, `content`, `citations`, `status` match history route response shape ✓
- `fetchHistory` returns `ChatMessage[]` which `useChatStream.loadHistory` sets as `messages` ✓
