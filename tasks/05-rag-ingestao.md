# Task 05 — RAG Ingestão (chunking + embeddings + indexação Qdrant)

## Objetivo

Pipeline offline que processa as 20 bulas PDF e indexa chunks em Qdrant com metadados
ricos. Execução via CLI (`scripts/ingest_bulas.py`). **Sem mudanças no chat ainda** —
a Task 06 integra retrieval ao chat_service.

## Pré-requisitos

- Task 01 (utils, settings, logger)
- Task 02 (models BulaChunk, BulaMetadata)
- Qdrant rodando localmente (docker run -p 6333:6333 qdrant/qdrant)

## Dependências novas

```toml
"pdfplumber>=0.11",
"qdrant-client>=1.12",
"fastembed>=0.4",          # para BM25 sparse vectors
"langchain-text-splitters>=0.3",
```

## Subtarefas

### 1. `utils/pdf.py` — extração + cache

```python
import hashlib
from pathlib import Path

import pdfplumber

from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)


def extract_text(pdf_path: Path, force: bool = False) -> str:
    """Extrai texto com pdfplumber. Cacheia em data/cache/<stem>.txt."""
    settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = settings.CACHE_DIR / f"{pdf_path.stem}.txt"
    if cache_file.exists() and not force:
        return cache_file.read_text(encoding="utf-8")

    with pdfplumber.open(pdf_path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    text = "\n\n[PAGE]\n\n".join(pages)  # marca separadores de página
    cache_file.write_text(text, encoding="utf-8")
    logger.info("pdf extraído", extra={"path": str(pdf_path), "chars": len(text)})
    return text


def file_hash(pdf_path: Path) -> str:
    return hashlib.md5(pdf_path.read_bytes()).hexdigest()
```

### 2. `assistant/sectionizer.py` — parser Anvisa

```python
import re
from dataclasses import dataclass

from panvel_assistant.models.bula import SectionCanonical


@dataclass(frozen=True)
class HeaderPattern:
    canonical: SectionCanonical
    patterns: list[re.Pattern]
    patient_facing: bool


# Cada item: (canonical, [regex aliases], patient_facing)
HEADERS: list[HeaderPattern] = [
    HeaderPattern("IDENT_APRESENTACOES",
        [re.compile(r"^\s*APRESENTA[ÇC][ÕO]ES\s*$", re.I | re.M)], False),
    HeaderPattern("IDENT_COMPOSICAO",
        [re.compile(r"^\s*COMPOSI[ÇC][ÃA]O\s*$", re.I | re.M)], False),
    HeaderPattern("IDENT_VIA_USO",
        [re.compile(r"^\s*(USO|VIA)\s+(ORAL|ADULTO|PEDI[ÁA]TRICO)\s*$", re.I | re.M)],
        False),

    HeaderPattern("IAP_1_INDICACOES",
        [re.compile(r"^\s*1\.\s*PARA QUE ESTE MEDICAMENTO É INDICADO\?\s*$", re.I | re.M)],
        True),
    HeaderPattern("IAP_2_MECANISMO",
        [re.compile(r"^\s*2\.\s*COMO ESTE MEDICAMENTO FUNCIONA\?\s*$", re.I | re.M)],
        True),
    HeaderPattern("IAP_3_CONTRAINDICACOES",
        [re.compile(r"^\s*3\.\s*QUANDO N[ÃA]O DEVO USAR ESTE MEDICAMENTO\?\s*$", re.I | re.M)],
        True),
    HeaderPattern("IAP_4_PRECAUCOES_ADVERTENCIAS",
        [re.compile(r"^\s*4\.\s*O QUE DEVO SABER ANTES DE USAR ESTE MEDICAMENTO\?\s*$", re.I | re.M)],
        True),
    HeaderPattern("IAP_5_ARMAZENAMENTO",
        [re.compile(r"^\s*5\.\s*ONDE.{0,5}COMO E POR QUANTO TEMPO POSSO GUARDAR.*?\?\s*$",
                    re.I | re.M)], True),
    HeaderPattern("IAP_6_POSOLOGIA",
        [re.compile(r"^\s*6\.\s*COMO DEVO USAR ESTE MEDICAMENTO\?\s*$", re.I | re.M)],
        True),
    HeaderPattern("IAP_7_ESQUECIMENTO_DOSE",
        [re.compile(r"^\s*7\.\s*O QUE DEVO FAZER QUANDO EU ME ESQUECER.*?\?\s*$",
                    re.I | re.M)], True),
    HeaderPattern("IAP_8_REACOES_ADVERSAS",
        [re.compile(r"^\s*8\.\s*QUAIS OS MALES QUE ESTE MEDICAMENTO PODE.*?\?\s*$",
                    re.I | re.M)], True),
    HeaderPattern("IAP_9_SUPERDOSE",
        [re.compile(r"^\s*9\.\s*O QUE FAZER SE ALGU[ÉE]M USAR.*?\?\s*$", re.I | re.M)],
        True),

    HeaderPattern("IT_CARACTERISTICAS_FARMACOLOGICAS",
        [re.compile(r"^\s*3\.\s*CARACTER[ÍI]STICAS FARMACOL[ÓO]GICAS\s*$", re.I | re.M)],
        False),
    HeaderPattern("IT_INTERACOES_MEDICAMENTOSAS",
        [re.compile(r"^\s*6\.\s*INTERA[ÇC][ÕO]ES MEDICAMENTOSAS\s*$", re.I | re.M)],
        False),
    HeaderPattern("IT_REACOES_ADVERSAS_TECNICAS",
        [re.compile(r"^\s*9\.\s*REA[ÇC][ÕO]ES ADVERSAS\s*$", re.I | re.M)], False),

    HeaderPattern("DIZERES_LEGAIS",
        [re.compile(r"^\s*DIZERES LEGAIS\s*$", re.I | re.M)], False),
]


@dataclass
class Section:
    canonical: SectionCanonical
    raw_header: str
    start: int
    end: int
    content: str
    patient_facing: bool
    occurrence: int  # 0 = primeira aparição; >0 = multi-produto


def sectionize(text: str) -> list[Section]:
    """Detecta seções canônicas no texto. Lida com multi-produto (Ritalina IR/LA)."""
    # 1) acha todos os matches com posição
    matches: list[tuple[int, HeaderPattern, str]] = []
    for hp in HEADERS:
        for pat in hp.patterns:
            for m in pat.finditer(text):
                matches.append((m.start(), hp, m.group(0).strip()))

    matches.sort(key=lambda x: x[0])
    if not matches:
        return [Section(
            canonical="UNCLASSIFIED", raw_header="", start=0, end=len(text),
            content=text, patient_facing=False, occurrence=0,
        )]

    # 2) constrói seções: cada match até início do próximo
    sections: list[Section] = []
    occurrence_counter: dict[str, int] = {}
    # trunca após DIZERES_LEGAIS para evitar histórico regulatório
    truncate_at = len(text)
    for pos, hp, _ in matches:
        if hp.canonical == "DIZERES_LEGAIS":
            # mantém DIZERES_LEGAIS mas trunca após próximo break grande
            pass

    for i, (pos, hp, raw) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else truncate_at
        occ = occurrence_counter.get(hp.canonical, 0)
        occurrence_counter[hp.canonical] = occ + 1
        sections.append(Section(
            canonical=hp.canonical,
            raw_header=raw,
            start=pos,
            end=end,
            content=text[pos:end].strip(),
            patient_facing=hp.patient_facing,
            occurrence=occ,
        ))
    return sections


def detect_med_variants(sections: list[Section]) -> dict[int, str | None]:
    """Mapeia occurrence → med_variant_label.
    Heurística: occurrence=0 → None (variant default), occurrence>0 → "variante N".
    Quando há múltiplas ocorrências de IAP_1_INDICACOES, indica multi-produto.
    """
    # IAP_1 indica o início de cada produto-variante
    iap1_count = sum(1 for s in sections if s.canonical == "IAP_1_INDICACOES")
    if iap1_count <= 1:
        return {0: None}
    return {i: f"variante_{i+1}" if i > 0 else None for i in range(iap1_count)}
```

### 3. `services/ingestion_service.py`

```python
import asyncio
import json
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, NamedSparseVector, NamedVector, PointStruct,
    SparseVector, SparseVectorParams, VectorParams,
)
from fastembed import SparseTextEmbedding

from panvel_assistant.assistant.sectionizer import Section, sectionize, detect_med_variants
from panvel_assistant.models.bula import BulaChunk, BulaMetadata
from panvel_assistant.utils.logger import get_logger
from panvel_assistant.utils.pdf import extract_text, file_hash
from panvel_assistant.utils.settings import settings

logger = get_logger(__name__)

CHUNK_SIZE = 800     # ~tokens (chars/4 approx) - usar tiktoken se quiser preciso
CHUNK_OVERLAP = 120
EMBED_DIM = 768
CONCURRENCY = 5      # respeitar rate limit Gemini

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE * 4,    # chars approx
    chunk_overlap=CHUNK_OVERLAP * 4,
    separators=["\n\n", "\n", ". ", " "],
)


def parse_filename(stem: str) -> tuple[str, str]:
    """Ex: '927100_ritalina_metilfenidato' → ('927100', 'Ritalina Metilfenidato')."""
    parts = stem.split("_", 1)
    bula_id = parts[0]
    med_name = parts[1].replace("_", " ").title() if len(parts) > 1 else stem
    return bula_id, med_name


def section_to_chunks(
    section: Section,
    bula_id: str,
    med_name: str,
    med_variant: str | None,
) -> list[BulaChunk]:
    pieces = splitter.split_text(section.content)
    chunks: list[BulaChunk] = []
    for idx, piece in enumerate(pieces):
        chunk_id = f"{bula_id}__{section.canonical}__{section.occurrence}__{idx}"
        chunks.append(BulaChunk(
            chunk_id=chunk_id,
            text=piece,
            metadata=BulaMetadata(
                bula_id=bula_id,
                med_name=med_name,
                med_variant=med_variant,
                section_canonical=section.canonical,
                section_raw_header=section.raw_header,
                chunk_idx=idx,
                source_page=None,  # opcional: estimar pelo [PAGE] marker
                patient_facing=section.patient_facing,
            ),
        ))
    return chunks


class IngestionService:
    def __init__(self) -> None:
        self._embedder = GoogleGenerativeAIEmbeddings(
            model=f"models/{settings.GEMINI_EMBED_MODEL}",
            google_api_key=settings.GOOGLE_API_KEY,
            task_type="RETRIEVAL_DOCUMENT",
        )
        self._sparse = SparseTextEmbedding("Qdrant/bm25")
        self._qdrant = AsyncQdrantClient(url=settings.QDRANT_URL)
        self._sem = asyncio.Semaphore(CONCURRENCY)

    async def ensure_collection(self) -> None:
        collections = await self._qdrant.get_collections()
        names = {c.name for c in collections.collections}
        if settings.QDRANT_COLLECTION in names:
            return
        await self._qdrant.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config={"dense": VectorParams(size=EMBED_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={"bm25": SparseVectorParams()},
        )
        logger.info("collection criada", extra={"name": settings.QDRANT_COLLECTION})

    async def _embed_dense(self, text: str) -> list[float]:
        async with self._sem:
            return await asyncio.to_thread(self._embedder.embed_query, text)

    def _embed_sparse(self, text: str) -> tuple[list[int], list[float]]:
        emb = next(iter(self._sparse.embed([text])))
        return emb.indices.tolist(), emb.values.tolist()

    def _process_pdf(self, pdf_path: Path) -> list[BulaChunk]:
        bula_id, med_name = parse_filename(pdf_path.stem)
        text = extract_text(pdf_path)
        sections = sectionize(text)
        variants = detect_med_variants(sections)
        all_chunks: list[BulaChunk] = []
        for s in sections:
            variant = variants.get(s.occurrence)
            all_chunks.extend(section_to_chunks(s, bula_id, med_name, variant))
        return all_chunks

    async def _upsert(self, chunk: BulaChunk) -> None:
        dense = await self._embed_dense(chunk.text)
        sparse_idx, sparse_val = self._embed_sparse(chunk.text)
        point = PointStruct(
            id=abs(hash(chunk.chunk_id)) % (2**63),  # int id estável
            vector={
                "dense": dense,
                "bm25": SparseVector(indices=sparse_idx, values=sparse_val),
            },
            payload={
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                **chunk.metadata.model_dump(),
            },
        )
        await self._qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION, points=[point]
        )

    async def ingest_corpus(self, corpus_dir: Path | None = None, rebuild: bool = False) -> dict:
        corpus_dir = corpus_dir or settings.BULAS_DIR
        await self.ensure_collection()
        if rebuild:
            await self._qdrant.delete_collection(settings.QDRANT_COLLECTION)
            await self.ensure_collection()

        manifest = {"bulas": {}, "total_chunks": 0}
        pdfs = sorted(corpus_dir.glob("*.pdf"))

        for pdf in pdfs:
            chunks = self._process_pdf(pdf)
            await asyncio.gather(*(self._upsert(c) for c in chunks))
            manifest["bulas"][pdf.name] = {
                "hash": file_hash(pdf),
                "chunks": len(chunks),
            }
            manifest["total_chunks"] += len(chunks)
            logger.info("bula indexada", extra={"file": pdf.name, "chunks": len(chunks)})

        manifest_path = settings.CACHE_DIR / "ingest_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        return manifest


ingestion_service = IngestionService()
```

### 4. `scripts/ingest_bulas.py`

```python
"""CLI para ingerir bulas em Qdrant.

Uso:
    uv run python scripts/ingest_bulas.py
    uv run python scripts/ingest_bulas.py --rebuild
"""
import argparse
import asyncio
import sys
from pathlib import Path

# permite rodar do root sem instalar pacote
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend" / "src"))

from panvel_assistant.services.ingestion_service import ingestion_service  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="apaga collection antes")
    args = parser.parse_args()
    manifest = await ingestion_service.ingest_corpus(rebuild=args.rebuild)
    print(f"OK: {manifest['total_chunks']} chunks indexados")
    for fname, info in manifest["bulas"].items():
        print(f"  {fname:55s} → {info['chunks']:3d} chunks")


if __name__ == "__main__":
    asyncio.run(main())
```

### 5. Testes

`tests/unit/test_sectionizer.py`:
- Texto sintético com headers conhecidos → seções na ordem correta
- Texto sem headers → 1 seção UNCLASSIFIED
- Texto multi-produto (2× IAP_1) → `occurrence` incrementa
- Texto com mistura IAP + IT (mesmos números) → resolução por regex distinto

`tests/integration/test_ingestion.py` (Qdrant docker):
- Ingerir 1 bula real do corpus → collection tem N pontos
- Buscar manualmente → metadata correta
- Idempotência: rodar 2x → não duplica (mesmo chunk_id)

## Verificação

```bash
# Qdrant up
docker run --rm -d -p 6333:6333 --name qdrant-dev qdrant/qdrant

# copiar bulas
cp -r ~/Downloads/"Case IA Generativa - Panvel"/corpus_bulas data/

# rodar ingestão
uv run python scripts/ingest_bulas.py --rebuild

# esperado: 20 bulas processadas, ~150-300 chunks total
# manifest gerado em data/cache/ingest_manifest.json

# query manual no Qdrant
curl -X POST http://localhost:6333/collections/bulas_panvel/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit":3,"with_payload":true}'

# espera: payloads com bula_id, section_canonical, med_name
```

## Gotchas

- `task_type="RETRIEVAL_DOCUMENT"` para embeds de documentos; `RETRIEVAL_QUERY` para
  queries (Task 06). Misturar degrada qualidade.
- Gemini embed rate limit: 1500 RPM para `text-embedding-004` (free tier). Semáforo=5
  com 200 chunks: ~40s. Aumentar se pago.
- Qdrant sparse vectors: requer versão ≥1.10 da imagem.
- BM25 do fastembed gera (indices, values) — formato esperado pelo Qdrant SparseVector.
- IDs Qdrant aceitam int OU uuid; `hash()` em Python varia entre runs com PYTHONHASHSEED;
  preferir `uuid.uuid5(NAMESPACE_OID, chunk_id)` para determinismo.
- pdfplumber pode estourar memória em PDFs grandes; bulas Anvisa são pequenas (<32k chars).
- `RecursiveCharacterTextSplitter` em chars, não tokens — multiplicador `×4` é aproximação.
  Para precisão usar `tiktoken`.
- Trunque após DIZERES_LEGAIS quando aparece (histórico regulatório longo nas 2 bulas
  maiores); detalhe na implementação do sectionizer ficou como TODO no código acima.
- IAP vs IT colidem na numeração (ambos têm "3.", "6.", "9."). Os regex distinguem
  pelo conteúdo (IAP é pergunta com `?`, IT é declarativo).
