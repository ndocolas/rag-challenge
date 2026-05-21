# ADR 005: Streaming via SSE

**Status:** Aceito
**Data:** 2026-05-20

## Contexto

O assistente gera respostas longas (posologia, contraindicações) e executa tools intermediárias. Sem streaming, o cliente esperaria vários segundos em branco antes de ver qualquer resposta — UX inaceitável.

O cliente é um browser (React/Vite) e precisa consumir o stream via JavaScript.

Alternativas avaliadas:

| Alternativa | Motivo de descarte |
|---|---|
| WebSocket | Bidirecional, mas adiciona handshake e estado de conexão; overhead desnecessário para um fluxo unidirecional servidor → cliente |
| HTTP/2 Server Push | Suporte irregular em browsers e proxies; semanticamente inadequado para stream de resposta |
| Long polling | Latência extra por round-trip a cada chunk; péssima UX |
| gRPC streaming | Sem suporte nativo em browsers sem proxy (grpc-web); complexidade desproporcional |

## Decisão

**SSE (`text/event-stream`)** via `StreamingResponse` do FastAPI:

- Cada evento é um frame `data: <json>\n\n`
- Tipos de evento tipados no campo `event` do SSE: `token`, `tool_call`, `tool_result`, `sources`, `done`, `error`
- Cliente usa **fetch + ReadableStream** (não `EventSource`) porque `EventSource` não suporta `POST` com body
- Headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no` (desativa buffer do nginx)

## Consequências

**Positivas:**
- Protocolo simples sobre HTTP/1.1: passa por proxies e load balancers sem configuração especial
- Eventos tipados permitem ao cliente renderizar cada fase (token streaming, indicador de tool, fontes) de forma independente
- FastAPI tem suporte nativo a `StreamingResponse`; sem biblioteca extra no backend
- Sem estado de conexão persistente: reconexão automática pelo cliente se necessário

**Negativas / trade-offs:**
- Unidirecional: cliente não pode cancelar o stream mid-flight via SSE (seria necessário uma segunda requisição HTTP)
- `EventSource` nativo do browser não funciona com POST — o cliente precisa de lógica custom com `fetch`
- Reconexão manual necessária: `EventSource` tem retry automático, `fetch` + `ReadableStream` não