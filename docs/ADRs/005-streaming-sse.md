# ADR 005: Streaming via SSE

**Status:** Accepted
**Date:** 2026-05-20

## Context

The assistant generates long responses (dosage, contraindications) and executes intermediate tools. Without streaming, the client would wait several seconds in blank before seeing any response — unacceptable UX.

The client is a browser (React/Vite) and needs to consume the stream via JavaScript.

Evaluated alternatives:

| Alternative | Reason for rejection |
|---|---|
| WebSocket | Bidirectional, but adds handshake and connection state; unnecessary overhead for a unidirectional server → client flow |
| HTTP/2 Server Push | Irregular support in browsers and proxies; semantically inadequate for response streaming |
| Long polling | Extra latency per round-trip per chunk; poor UX |
| gRPC streaming | No native browser support without proxy (grpc-web); disproportionate complexity |

## Decision

**SSE (`text/event-stream`)** via FastAPI `StreamingResponse`:

- Each event is a `data: <json>\n\n` frame
- Event types typed in the SSE `event` field: `token`, `tool_call`, `tool_result`, `sources`, `done`, `error`
- Client uses **fetch + ReadableStream** (not `EventSource`) because `EventSource` doesn't support `POST` with body
- Headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no` (disables nginx buffering)

## Consequences

**Positive:**
- Simple protocol over HTTP/1.1: passes through proxies and load balancers without special configuration
- Typed events allow client to render each phase (token streaming, tool indicator, sources) independently
- FastAPI has native `StreamingResponse` support; no extra library on backend
- No persistent connection state: automatic client reconnect if needed

**Negative / trade-offs:**
- Unidirectional: client cannot cancel stream mid-flight via SSE (would require a second HTTP request)
- Native browser `EventSource` doesn't work with POST — client needs custom logic with `fetch`
- Manual reconnection required: `EventSource` has automatic retry, `fetch` + `ReadableStream` does not
