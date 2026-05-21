# ADR 004: LangChain sem LangGraph

**Status:** Aceito
**Data:** 2026-05-20

## Contexto

O assistente precisa de um loop de tool calling com as seguintes propriedades:

- Streaming token-a-token durante a geração (SSE para o cliente)
- Execução paralela de múltiplas tool calls em um mesmo turno
- Guard contra loop infinito (ex: tool call repetida ou ciclo entre tools)
- Rastreabilidade: cada turno precisa de um trace_id observável no LangSmith
- Código debugável por um time pequeno sem curva de aprendizado de frameworks de grafo

Alternativas avaliadas:

| Alternativa | Motivo de descarte |
|---|---|
| LangGraph | State machine explícita necessária; curva de aprendizado relevante; overhead para caso de uso linear |
| OpenAI SDK direto | Mais controle, mas mais código boilerplate; tracing manual; lock-in de provider |
| AgentExecutor (LangChain clássico) | Deprecated; sem suporte a streaming granular de tool calls |

## Decisão

**LangChain puro** com loop manual em `AssistantService`:

- `llm.bind_tools(tools)` para registrar as 5 tools no modelo
- `astream` do LangChain para streaming token-a-token via `astream_events`
- Loop manual: detecta `tool_calls` na resposta, executa em paralelo com `asyncio.gather`, anexa resultados como `ToolMessage`, reinicia o stream
- Guard de segurança: deduplicação por fingerprint de tool call (nome + args) aborta loop duplicado; `MAX_TOOL_ITERATIONS=4` limita profundidade

## Consequências

**Positivas:**
- Código direto e linear: fácil de debugar e testar com mocks
- LangSmith traça automaticamente via callbacks LangChain (sem instrumentação manual)
- Troca de provider LLM = apenas alterar `builders.py` (interface `ChatGoogleGenerativeAI` → qualquer `BaseChatModel`)
- Controle granular dos eventos SSE: cada etapa (token, tool_call, tool_result, sources) é emitida explicitamente

**Negativas / trade-offs:**
- Loop manual precisa ser mantido se o contrato de tool calling do LangChain mudar
- Sem state machine explícita: fluxos mais complexos (ex: ramificações condicionais) exigiriam refatoração
- `MAX_TOOL_ITERATIONS=4` é um limite fixo; casos edge com muitas tools legítimas precisariam de ajuste