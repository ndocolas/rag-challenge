# Task 06 — RAG Retrieval + integração no chat

## Objetivo

Recuperar chunks relevantes do Qdrant (hybrid dense+BM25 com RRF) e injetar como
contexto no chat_service. Emitir evento SSE `sources` com Citations. Respostas
farmacológicas passam a citar bula+seção.

## Pré-requisitos

- Task 05 (ingestão concluída, collection populada)
- Task 04 (chat com tools)

## Subtarefas

### 1. `services/rag_service.py`

```python
import re
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FusionQuery, Fusion, Prefetch, SparseVector,
)
from fastembed import SparseTextEmbedding

from panvel_assistant.models.bula import BulaChunk, BulaMetadata
from panvel_assistant.models.chat import Citation
from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)


# Intent classifier rudimentar: regex em palavras-chave → section preferencial
INTENT_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(posologia|como (usar|tomar)|dose|dosagem)\b", re.I), "IAP_6_POSOLOGIA"),
    (re.compile(r"\b(contraindica|não devo|nao devo)\b", re.I), "IAP_3_CONTRAINDICACOES"),
    (re.compile(r"\b(reação|reacao|efeito colateral|males)\b", re.I), "IAP_8_REACOES_ADVERSAS"),
    (re.compile(r"\b(interação|interacao|interage)\b", re.I), "IT_INTERACOES_MEDICAMENTOSAS"),
    (re.compile(r"\b(superdose|overdose|usar a mais)\b", re.I), "IAP_9_SUPERDOSE"),
    (re.compile(r"\b(armazena|guardar|conservar)\b", re.I), "IAP_5_ARMAZENAMENTO"),
    (re.compile(r"\b(esqueci|esquecer)\b", re.I), "IAP_7_ESQUECIMENTO_DOSE"),
    (re.compile(r"\b(para que serve|indicação|indicacao|indicado)\b", re.I), "IAP_1_INDICACOES"),
    (re.compile(r"\b(como funciona|mecanismo|farmacologia)\b", re.I), "IAP_2_MECANISMO"),
]

SECTION_LABEL = {
    "IAP_1_INDICACOES": "Para que é indicado",
    "IAP_2_MECANISMO": "Como funciona",
    "IAP_3_CONTRAINDICACOES": "Quando não devo usar",
    "IAP_4_PRECAUCOES_ADVERTENCIAS": "O que devo saber antes de usar",
    "IAP_5_ARMAZENAMENTO": "Como guardar",
    "IAP_6_POSOLOGIA": "Como devo usar",
    "IAP_7_ESQUECIMENTO_DOSE": "Esquecimento de dose",
    "IAP_8_REACOES_ADVERSAS": "Reações adversas",
    "IAP_9_SUPERDOSE": "Superdose",
    "IT_CARACTERISTICAS_FARMACOLOGICAS": "Características farmacológicas",
    "IT_INTERACOES_MEDICAMENTOSAS": "Interações medicamentosas",
    "IT_REACOES_ADVERSAS_TECNICAS": "Reações adversas (técnico)",
    "IDENT_APRESENTACOES": "Apresentações",
    "IDENT_COMPOSICAO": "Composição",
    "IDENT_VIA_USO": "Via/Uso",
    "DIZERES_LEGAIS": "Dizeres legais",
    "UNCLASSIFIED": "—",
}


def detect_intent_section(query: str) -> str | None:
    for pat, section in INTENT_HINTS:
        if pat.search(query):
            return section
    return None


class RAGService:
    def __init__(self) -> None:
        self._embedder = GoogleGenerativeAIEmbeddings(
            model=f"models/{settings.GEMINI_EMBED_MODEL}",
            google_api_key=settings.GOOGLE_API_KEY,
            task_type="RETRIEVAL_QUERY",
        )
        self._sparse = SparseTextEmbedding("Qdrant/bm25")
        self._qdrant = AsyncQdrantClient(url=settings.QDRANT_URL)

    async def retrieve(self, query: str, k: int = 6) -> list[BulaChunk]:
        dense = await self._embedder.aembed_query(query)
        sparse = next(iter(self._sparse.query_embed([query])))

        hint_section = detect_intent_section(query)

        # hybrid via prefetch + fusion RRF
        results = await self._qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            prefetch=[
                Prefetch(query=dense, using="dense", limit=k * 3),
                Prefetch(
                    query=SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                    using="bm25",
                    limit=k * 3,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k * 2,  # mais que k para permitir dedup
            with_payload=True,
        )

        # dedup por (bula_id, section_canonical): mantém só primeiro
        seen: set[tuple[str, str]] = set()
        chunks: list[BulaChunk] = []
        for p in results.points:
            payload = p.payload
            key = (payload["bula_id"], payload["section_canonical"])
            if key in seen:
                continue
            seen.add(key)
            md = BulaMetadata(**{k: payload[k] for k in BulaMetadata.model_fields})
            chunks.append(BulaChunk(
                chunk_id=payload["chunk_id"],
                text=payload["text"],
                metadata=md,
                score=p.score,
            ))
            if len(chunks) >= k:
                break

        # se intent claro, prioriza chunks daquela seção no topo
        if hint_section:
            chunks.sort(key=lambda c: (0 if c.metadata.section_canonical == hint_section else 1, -1 * (c.score or 0)))

        logger.info(
            "retrieval concluído",
            extra={"step": "retrieval", "query_len": len(query), "k": k,
                   "hint": hint_section, "returned": len(chunks)},
        )
        return chunks

    @staticmethod
    def format_citations(chunks: list[BulaChunk]) -> list[Citation]:
        out: list[Citation] = []
        for c in chunks:
            md = c.metadata
            out.append(Citation(
                bula_id=md.bula_id,
                med_name=md.med_name,
                med_variant=md.med_variant,
                section_canonical=md.section_canonical,
                section_label=SECTION_LABEL.get(md.section_canonical, md.section_canonical),
                source_page=md.source_page,
                snippet=c.text[:200].strip(),
            ))
        return out

    @staticmethod
    def format_context_for_llm(chunks: list[BulaChunk]) -> str:
        """Texto pronto pra injetar como SystemMessage."""
        parts = ["Contexto recuperado das bulas (use para responder, citando fonte):\n"]
        for c in chunks:
            md = c.metadata
            label = SECTION_LABEL.get(md.section_canonical, md.section_canonical)
            tag = f"[{md.med_name}{' '+md.med_variant if md.med_variant else ''} — {label}]"
            parts.append(f"\n{tag}\n{c.text}\n")
        return "\n".join(parts)


rag_service = RAGService()
```

### 2. Atualizar `chat_service.py`

```python
# pseudocódigo dos passos novos dentro de handle_turn:

async def handle_turn(self, req: ChatRequest):
    history = await chat_history_service.load(req.session_id)
    user_msg = ChatMessage(role="user", content=req.message)

    # NOVO: retrieval sempre (MVP heurístico). Pode-se condicionar a "tem nome de
    # medicamento na mensagem", mas simplicidade > otimização inicial.
    chunks = await rag_service.retrieve(req.message, k=4)
    citations = rag_service.format_citations(chunks)
    context_msg = SystemMessage(content=rag_service.format_context_for_llm(chunks))

    # emite evento sources ANTES do primeiro token
    yield encode_event("sources", {"citations": [c.model_dump() for c in citations]})

    # monta mensagens com contexto + system + histórico + usuário
    lc_messages = [SystemMessage(content=SYSTEM_PROMPT_MVP), context_msg]
    lc_messages += _to_lc(history + [user_msg])[1:]  # skip o system prompt duplicado

    # restante do loop (tool calling + streaming) idêntico à Task 04
    ...
```

### 3. Atualizar `prompts.py`

Adicionar regras de citação:

```python
SYSTEM_PROMPT_MVP = """\
... (anterior) ...

Quando o usuário pergunta sobre medicamentos:
1. Use APENAS o contexto recuperado das bulas para responder fatos farmacológicos.
2. SEMPRE cite a fonte no formato [Medicamento — Seção], ex: [Ritalina — Posologia].
3. Se o contexto não tem a resposta, diga: "Não encontrei essa informação nas bulas
   disponíveis." — NÃO invente.
4. Sempre inclua o disclaimer: "Esta informação não substitui orientação médica."

Quando o usuário pergunta sobre filiais, use as tools (não invente dados).
"""
```

### 4. Testes

`tests/unit/test_intent_classifier.py`:
- "posologia da ritalina" → IAP_6_POSOLOGIA
- "quais reações" → IAP_8_REACOES_ADVERSAS
- "tem 24h em Curitiba" → None (intent não-farmacológico)

`tests/integration/test_retrieval.py` (Qdrant + corpus indexado):
- query "contraindicações da ritalina" → top-3 contém pelo menos 1 chunk de
  bula 927100, section IAP_3_CONTRAINDICACOES
- query "posologia pantoprazol" → top-3 contém chunks de bula 805950, section IAP_6
- dedup: nenhum (bula_id, section_canonical) repetido nos resultados

`tests/integration/test_chat_with_rag.py`:
- POST /chat "contraindicações da ritalina"
- valida evento `sources` antes de `token`
- valida que resposta contém substring `[Ritalina —`
- valida disclaimer médico presente

## Verificação

Pré-condição: Qdrant rodando com corpus indexado (Task 05 executada).

```bash
# pergunta farmacológica
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"r1","message":"Quais as contraindicações da Ritalina?"}'

# esperado:
# event: sources    {citations: [{med_name:"Ritalina", section:"IAP_3_CONTRAINDICACOES", ...}, ...]}
# event: token      ... (resposta cita [Ritalina — Quando não devo usar])
# event: token      ... ("Esta informação não substitui orientação médica.")
# event: done

# multi-turno: contexto resolvido por anáfora
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"r1","message":"e quais os males?"}'

# esperado: retrieval ainda recupera bula ritalina (histórico + msg atual), resposta
# cita [Ritalina — Reações adversas]

# fora do corpus
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"r2","message":"qual a posologia da dipirona?"}'

# esperado: chunks pouco relevantes → LLM responde "não encontrei nas bulas disponíveis"
```

## Gotchas

- `aembed_query` vs `embed_query`: usar async em hot path.
- Fusion RRF do Qdrant requer cliente ≥1.10 e collection criada com sparse_vectors_config.
- Dedup é crítico: sem ele, mesma bula+seção aparece N vezes nos top-k.
- Intent classifier é heurístico — falsos positivos OK porque é só hint de ordenação,
  não filtro hard. Falsos negativos não machucam (fallback para fusion score).
- Histórico para retrieval: MVP usa só a mensagem atual. Pode-se concatenar últimas 2
  turnos para resolver anáfora ("e quais os males?"), mas explode tokens. Avaliar.
- SystemMessage com contexto pode ficar grande (4 chunks × ~800 tokens = ~3200 tokens).
  Gemini Flash suporta 1M context, sem problema.
- Quando NÃO recuperar nada relevante (todos os scores baixos), considerar fallback:
  "Não encontrei nas bulas disponíveis" como instrução pro LLM. Threshold por score
  pode ser adicionado depois.
- Citação pelo LLM é melhor-effort: pode esquecer formato. Mitigar com few-shots no
  system prompt.
