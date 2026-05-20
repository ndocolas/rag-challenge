# ADR 001: LLM provider — Google Gemini

**Status:** Aceito
**Data:** 2026-05-20

## Contexto

O assistente precisa de um modelo de linguagem que suporte simultaneamente:

- Tool calling nativo (para invocar ferramentas de filiais e RAG)
- Streaming token-a-token (SSE)
- Geração de embeddings (para ingestão e retrieval)
- Boa qualidade em português brasileiro
- Custo acessível para um MVP com 20 bulas e ~100 req/dia

Alternativas avaliadas:

| Provedor | Motivo de descarte |
|---|---|
| OpenAI GPT-4o | Custo elevado; embeddings separados (`text-embedding-3`) aumentam complexidade |
| Anthropic Claude | Não oferece embeddings próprios; necessitaria segundo provedor |
| AWS Bedrock | Setup pesado (IAM, VPC, credenciais temporárias); latência extra |
| Modelos locais (Ollama) | Qualidade insuficiente em PT-BR para tool calling + RAG |

## Decisão

Usar **Google Gemini** como provedor único:

- Chat: `gemini-2.0-flash` (`temperature=0.2`, streaming habilitado)
- Embeddings: `gemini-embedding-001` (3072 dimensões, task types `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`)
- Integração via **LangChain** (`ChatGoogleGenerativeAI` + `GoogleGenerativeAIEmbeddings`)

## Consequências

**Positivas:**
- Free tier generoso cobre o volume do MVP sem custo
- Cliente unificado via LangChain: trocar de provedor exige alterar apenas `builders.py`
- LangSmith traça chamadas Gemini automaticamente (via LangChain callbacks)
- Embeddings 3072-dim suficientes para corpus de 20 bulas; BM25 complementa o recall

**Negativas / trade-offs:**
- Dependência de um único provedor externo (disponibilidade e rate limits do Google)
- Embeddings 3072-dim vs 1536-dim do OpenAI: vetor maior, mas corpus pequeno não é gargalo
- `gemini-2.0-flash` não é o modelo mais capaz da família; aceitável para o escopo
