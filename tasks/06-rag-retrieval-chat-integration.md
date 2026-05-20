# Task 06 — RAG Retrieval + integração no chat (agentic tool)

## Objetivo

Expor o retrieval do Qdrant como **tool** `buscar_bulas` que o agent invoca quando
detecta intent farmacológico. A tool faz hybrid search (dense Gemini +
BM25 sparse com fusion RRF), aceita filtros estruturais (`med_name`,
`patient_facing`, `section_hint`) e retorna chunks tipados que o agent cita.

Diferente da abordagem "RAG sempre-ligado" do design original, esta versão
**não modifica `chat_service.handle_turn`**: o LLM decide quando recuperar via
tool calling, igual ao padrão das tools de filiais. Isso economiza tokens em
small-talk e perguntas não-farmacológicas, e permite múltiplas recuperações
na mesma turn (reformulação de query).

## Pré-requisitos

- Task 05 (ingestão concluída, collection populada com payload contendo
  `is_full_section`, `section_char_len`, `patient_facing`, `med_name`)
- Task 04 (chat com tools — `build_tools()` em `agent_tools.py`)

## Subtarefas

### 1. `services/rag_service.py`

Núcleo de retrieval. Stateless, reusável pela tool em `agent_tools.py`.

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, MatchValue,
    FusionQuery, Fusion, Prefetch, SparseVector,
)
from fastembed import SparseTextEmbedding

from panvel_assistant.models.bula import BulaChunk, BulaMetadata, SectionCanonical
from panvel_assistant.models.chat import Citation
from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)


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


class RAGService:
    def __init__(self) -> None:
        self._embedder = GoogleGenerativeAIEmbeddings(
            model=f"models/{settings.GEMINI_EMBED_MODEL}",
            google_api_key=settings.GOOGLE_API_KEY,
            task_type="RETRIEVAL_QUERY",  # diferente do ingest (RETRIEVAL_DOCUMENT)
        )
        self._sparse = SparseTextEmbedding("Qdrant/bm25")
        self._qdrant = AsyncQdrantClient(url=settings.QDRANT_URL)

    def _build_filter(
        self,
        med_name: str | None,
        section_hint: SectionCanonical | None,
        patient_facing_only: bool,
    ) -> Filter | None:
        must: list[FieldCondition] = []
        if med_name:
            # match exato no payload; agent costuma extrair o nome canonico
            must.append(FieldCondition(key="med_name", match=MatchValue(value=med_name)))
        if section_hint:
            must.append(FieldCondition(key="section_canonical",
                                       match=MatchValue(value=section_hint)))
        if patient_facing_only:
            must.append(FieldCondition(key="patient_facing", match=MatchValue(value=True)))
        return Filter(must=must) if must else None

    async def retrieve(
        self,
        query: str,
        k: int = 4,
        med_name: str | None = None,
        section_hint: SectionCanonical | None = None,
        patient_facing_only: bool = True,
    ) -> list[BulaChunk]:
        """Hybrid search (dense + BM25 com RRF) + filtros estruturais.

        Tipicamente chamado pela tool `buscar_bulas` em ``agent_tools.py``.
        """
        dense = await self._embedder.aembed_query(query)
        sparse = next(iter(self._sparse.query_embed([query])))
        qfilter = self._build_filter(med_name, section_hint, patient_facing_only)

        results = await self._qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            prefetch=[
                Prefetch(query=dense, using="dense", limit=k * 4, filter=qfilter),
                Prefetch(
                    query=SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                    using="bm25",
                    limit=k * 4,
                    filter=qfilter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k * 3,
            with_payload=True,
        )

        # Dedup inteligente:
        # - chunk_id duplicado -> sempre descarta (n score-noise)
        # - is_full_section=True duplicado para mesma (bula_id, section_canonical)
        #   -> mantem so o melhor (mesma se\u00e7\u00e3o inteira aparece 1x)
        # - is_full_section=False (sub-chunks) -> NAO deduplica por se\u00e7\u00e3o,
        #   permitindo que 2 trechos diferentes da mesma se\u00e7\u00e3o longa sejam usados.
        seen_chunk_ids: set[str] = set()
        seen_full_sections: set[tuple[str, str]] = set()
        chunks: list[BulaChunk] = []
        for p in results.points:
            payload = p.payload
            chunk_id = payload["chunk_id"]
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            if payload.get("is_full_section"):
                key = (payload["bula_id"], payload["section_canonical"])
                if key in seen_full_sections:
                    continue
                seen_full_sections.add(key)
            md = BulaMetadata(**{k: payload[k] for k in BulaMetadata.model_fields
                                 if k in payload})
            chunks.append(BulaChunk(
                chunk_id=chunk_id,
                text=payload["text"],
                metadata=md,
                score=p.score,
            ))
            if len(chunks) >= k:
                break

        logger.info(
            "retrieval concluido",
            extra={
                "step": "retrieval",
                "query_len": len(query),
                "k": k,
                "med_name": med_name,
                "section_hint": section_hint,
                "patient_facing_only": patient_facing_only,
                "returned": len(chunks),
            },
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
    def format_tool_payload(chunks: list[BulaChunk]) -> dict:
        """Saida JSON-serializable para a tool retornar ao LLM."""
        items = []
        for c in chunks:
            md = c.metadata
            label = SECTION_LABEL.get(md.section_canonical, md.section_canonical)
            items.append({
                "chunk_id": c.chunk_id,
                "med_name": md.med_name,
                "med_variant": md.med_variant,
                "section_canonical": md.section_canonical,
                "section_label": label,
                "is_full_section": md.is_full_section,
                "text": c.text,
                "score": c.score,
            })
        return {"matches": items, "total": len(items)}


rag_service = RAGService()
```

### 2. Nova tool `buscar_bulas` em `assistant/agent_tools.py`

`build_tools()` ganha um 4º tool, com `rag_service` injetado igual ao
`filiais_service`. Input schema fica em `models/tool.py`.

```python
# models/tool.py (novos schemas)
from typing import Literal
from pydantic import BaseModel, Field

SectionHint = Literal[
    "IAP_1_INDICACOES", "IAP_2_MECANISMO", "IAP_3_CONTRAINDICACOES",
    "IAP_4_PRECAUCOES_ADVERTENCIAS", "IAP_5_ARMAZENAMENTO",
    "IAP_6_POSOLOGIA", "IAP_7_ESQUECIMENTO_DOSE",
    "IAP_8_REACOES_ADVERSAS", "IAP_9_SUPERDOSE",
    "IT_INTERACOES_MEDICAMENTOSAS", "IT_REACOES_ADVERSAS_TECNICAS",
    "IT_CARACTERISTICAS_FARMACOLOGICAS",
]


class BuscarBulasInput(BaseModel):
    query: str = Field(..., description="Pergunta/termos sobre o medicamento.")
    med_name: str | None = Field(
        None, description="Nome canonico do medicamento (ex: 'Ritalina'). "
                         "Filtra por igualdade no payload; use quando souber.")
    section_hint: SectionHint | None = Field(
        None, description="Hint de secao Anvisa quando intent for claro "
                         "(posologia, contraindicacao, reacoes, etc).")
    patient_facing_only: bool = Field(
        True, description="True (default) restringe a secoes IAP (linguagem ao "
                         "paciente). Use False so para perguntas tecnicas.")
    k: int = Field(4, ge=1, le=10, description="Numero de chunks retornados.")
```

```python
# agent_tools.py (adicao em build_tools)
def build_tools(
    filiais_service: FiliaisService,
    rag_service: RAGService,
) -> list:
    ...  # tools existentes

    @tool("buscar_bulas", args_schema=BuscarBulasInput)
    async def buscar_bulas(
        query: str,
        med_name: str | None = None,
        section_hint: str | None = None,
        patient_facing_only: bool = True,
        k: int = 4,
    ) -> str:
        """Recupera trechos de bulas Anvisa relevantes para uma pergunta
        farmacologica.

        Use quando o usuario perguntar sobre indicacao, posologia, contraindicacao,
        reacoes adversas, interacoes, armazenamento, mecanismo, etc., de algum
        medicamento. Passe `med_name` sempre que conseguir identificar o nome do
        medicamento (canonico, ex.: 'Ritalina', 'Losartana') para ganhar precisao.

        Retorna lista de matches com texto, secao canonica, label legivel e flag
        is_full_section (indicando se eh a secao completa ou um trecho).
        """
        started = time.perf_counter()
        try:
            chunks = await rag_service.retrieve(
                query=query, k=k, med_name=med_name,
                section_hint=section_hint, patient_facing_only=patient_facing_only,
            )
            payload = RAGService.format_tool_payload(chunks)
            return json.dumps(payload, ensure_ascii=False)
        finally:
            logger.info("tool buscar_bulas", extra={
                "step": "tool", "tool": "buscar_bulas",
                "latency_ms": (time.perf_counter() - started) * 1000,
                "k": k, "med_name": med_name,
            })

    return [buscar_filiais, detalhes_filial, listar_cidades_atendidas, buscar_bulas]
```

### 2b. Emissao do evento SSE `sources`

`chat_service.handle_turn` **nao** muda fluxo principal. Para emitir
`sources` precisamos interceptar quando a tool `buscar_bulas` retorna:

- Opcao A (escolhida): apos cada `ToolMessage` no loop de tool-calling, se a
  tool foi `buscar_bulas`, parsear o JSON de retorno e emitir
  `event: sources {citations: [...]}` antes dos tokens subsequentes.
- Helper `RAGService.format_citations` continua existindo mas eh consumido
  pelo chat_service a partir do payload da tool (re-parse `matches`).

```python
# pseudocodigo no loop existente da Task 04:
if tool_msg.name == "buscar_bulas":
    matches = json.loads(tool_msg.content).get("matches", [])
    citations = [_match_to_citation(m) for m in matches]
    yield encode_event("sources", {"citations": [c.model_dump() for c in citations]})
```

Nenhuma `SystemMessage` extra eh injetada: o LLM ja recebe o `ToolMessage`
com os chunks, do jeito padrao de tool calling.

### 3. Atualizar `prompts.py`

Adicionar regras de uso da tool + citação:

```python
SYSTEM_PROMPT_MVP = """\
... (anterior) ...

Quando o usuário pergunta sobre medicamentos (indicação, posologia, contraindicação,
reações, interações, mecanismo, armazenamento, etc.):
1. Chame a tool `buscar_bulas` para recuperar trechos das bulas Anvisa.
   - Passe `med_name` SEMPRE que o nome do medicamento estiver claro
     (extraia de mensagens anteriores se preciso).
   - Use `section_hint` quando a intenção for óbvia
     (ex.: "posologia" → IAP_6_POSOLOGIA, "reações" → IAP_8_REACOES_ADVERSAS).
2. Responda usando APENAS os `matches` retornados pela tool.
3. SEMPRE cite a fonte no formato [Medicamento — Seção],
   ex: [Ritalina — Como devo usar].
4. Se `matches` vier vazio ou irrelevante, diga:
   "Não encontrei essa informação nas bulas disponíveis." — NÃO invente.
5. Sempre inclua o disclaimer: "Esta informação não substitui orientação médica."
6. Para pacientes, prefira manter `patient_facing_only=true` (default). Use false
   só quando o usuário declarar ser profissional de saúde ou pedir info técnica.

Quando o usuário pergunta sobre filiais, use as tools de filiais (não invente dados).

Para perguntas mistas (medicamento + onde comprar), pode chamar `buscar_bulas` e
`buscar_filiais` na mesma turn.
"""
```

### 4. Testes

`tests/unit/test_rag_filter.py`:
- `_build_filter(med_name="Ritalina")` produz Filter com 1 must (med_name)
- `_build_filter(patient_facing_only=True)` adiciona condição patient_facing
- `_build_filter(None, None, False)` retorna None (sem filtro)
- combinação dos três → 3 conditions em `must`

`tests/integration/test_retrieval.py` (Qdrant + corpus indexado):
- query "contraindicações" + `med_name="Ritalina"` → top-3 todos com
  med_name=Ritalina e section_canonical=IAP_3_CONTRAINDICACOES
- query "posologia" + `med_name="Pantoprazol"` + `section_hint="IAP_6_POSOLOGIA"`
  → bula 805950 retorna chunks IAP_6
- `patient_facing_only=True` (default) → nenhum chunk com section IT_* retorna
- Dedup: 2 sub-chunks da mesma seção longa (is_full_section=False) PODEM aparecer;
  2 chunks com is_full_section=True da mesma (bula,seção) NÃO aparecem.

`tests/integration/test_tool_buscar_bulas.py`:
- Invoca a tool diretamente → retorna JSON com `matches` e `total`
- Cada item tem campos chunk_id, med_name, section_canonical, is_full_section
- Quando `med_name` não existe no corpus → matches vazio (não erro)

`tests/integration/test_chat_with_rag.py`:
- POST /chat "contraindicações da Ritalina"
- LLM **chama** a tool `buscar_bulas` (verificável via tool_calls no trace)
- evento `sources` é emitido após o ToolMessage e antes dos tokens finais
- resposta contém substring `[Ritalina —`
- disclaimer médico presente
- POST /chat "oi tudo bem?" → tool **não é chamada**, nenhum evento `sources`

## Verificação

Pré-condição: Qdrant rodando com corpus indexado (Task 05 executada).

```bash
# pergunta farmacológica
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"r1","message":"Quais as contraindicações da Ritalina?"}'

# esperado:
# event: tool_call  {tool: "buscar_bulas", args: {query:"contraindicacoes",
#                                                 med_name:"Ritalina",
#                                                 section_hint:"IAP_3_CONTRAINDICACOES"}}
# event: sources    {citations: [{med_name:"Ritalina",
#                                 section:"IAP_3_CONTRAINDICACOES", ...}, ...]}
# event: token      ... (resposta cita [Ritalina — Quando não devo usar])
# event: token      ... ("Esta informação não substitui orientação médica.")
# event: done

# multi-turno: contexto resolvido pelo proprio agent
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"r1","message":"e quais os males?"}'

# esperado: agent chama buscar_bulas(query="reacoes adversas", med_name="Ritalina",
# section_hint="IAP_8_REACOES_ADVERSAS"), resposta cita [Ritalina — Reações adversas]

# small-talk: NAO chama a tool
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"r3","message":"oi, tudo bem?"}'

# esperado: nenhum tool_call, nenhum evento sources, resposta direta

# fora do corpus
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"r2","message":"qual a posologia da dipirona?"}'

# esperado: agent chama buscar_bulas, matches vazio, LLM responde
# "Nao encontrei essa informacao nas bulas disponiveis."
```

## Gotchas

- `aembed_query` vs `embed_query`: usar async em hot path.
- Fusion RRF do Qdrant requer cliente ≥1.10 e collection criada com
  sparse_vectors_config (Task 05 já garante).
- Dedup hybrid: por `chunk_id` sempre; por `(bula_id, section_canonical)` SÓ
  para chunks com `is_full_section=True`. Caso contrário sub-chunks da mesma
  seção longa são incorretamente descartados.
- Filtros Qdrant aplicam-se DENTRO dos `Prefetch`, não só no top-level — senão
  o RRF mistura candidatos filtrados e não-filtrados.
- `med_name` filter exige match exato no payload. Garanta consistência com o
  que `parse_filename` produz no ingest (case-sensitive). Se necessário,
  documentar nomes canônicos no system prompt para o LLM acertar.
- Resolução de anáfora ("e quais os males?"): o agent vê o histórico via tool
  calling — não precisamos concatenar turns manualmente. O LLM passa
  `med_name="Ritalina"` na chamada da tool mesmo sem o usuário repetir.
- Token budget: 4 chunks × seção inteira (até ~3500 chars) ≈ 14k chars
  (~3500 tokens) por chamada. Gemini Flash suporta 1M, sem problema.
- Quando `matches` vier vazio, o LLM deve dizer "não encontrei". Reforçado no
  system prompt; podemos adicionar threshold por score depois.
- LLM esquecer formato de citação: mitigar com few-shots ou pós-processamento
  validando regex `\[<Med> — <Section>\]` na resposta.
- Tool não decidida pelo LLM: se em testes o LLM ignorar `buscar_bulas` para
  perguntas farmacológicas, reforçar o system prompt com exemplos negativos
  ("NÃO responda sobre medicamento sem chamar buscar_bulas primeiro").
