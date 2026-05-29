# ADR 004: LangChain without LangGraph

**Status:** Accepted
**Date:** 2026-05-20

## Context

The assistant needs a tool calling loop with the following properties:

- Token-by-token streaming during generation (SSE to client)
- Parallel execution of multiple tool calls in the same turn
- Infinite loop guard (e.g., repeated tool call or cycle between tools)
- Traceability: each turn needs an observable trace_id in LangSmith
- Code debuggable by a small team without a graph framework learning curve

Evaluated alternatives:

| Alternative | Reason for rejection |
|---|---|
| LangGraph | Explicit state machine required; relevant learning curve; overhead for linear use case |
| OpenAI SDK directly | More control, but more boilerplate; manual tracing; provider lock-in |
| AgentExecutor (classic LangChain) | Deprecated; no granular tool call streaming support |

## Decision

**Pure LangChain** with manual loop in `AssistantService`:

- `llm.bind_tools(tools)` to register the 5 tools in the model
- LangChain `astream` for token-by-token streaming via `astream_events`
- Manual loop: detects `tool_calls` in response, executes in parallel with `asyncio.gather`, appends results as `ToolMessage`, restarts stream
- Safety guard: deduplication by tool call fingerprint (name + args) aborts duplicate loop; `MAX_TOOL_ITERATIONS=4` limits depth

## Consequences

**Positive:**
- Direct and linear code: easy to debug and test with mocks
- LangSmith traces automatically via LangChain callbacks (no manual instrumentation)
- LLM provider swap = modify only `builders.py` (`ChatGoogleGenerativeAI` → any `BaseChatModel`)
- Granular SSE event control: each step (token, tool_call, tool_result, sources) emitted explicitly

**Negative / trade-offs:**
- Manual loop must be maintained if LangChain's tool calling contract changes
- No explicit state machine: more complex flows (e.g., conditional branches) would require refactoring
- `MAX_TOOL_ITERATIONS=4` is a fixed limit; edge cases with many legitimate tools would need adjustment
