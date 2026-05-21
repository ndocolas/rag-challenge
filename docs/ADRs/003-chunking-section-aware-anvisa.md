
# ADR 003: Chunking section-aware Anvisa

**Status:** Aceito
**Data:** 2026-05-20

## Contexto

O corpus é composto por 20 bulas Anvisa no formato RDC 47/2009, que define duas estruturas:

- **IAP (Informações ao Paciente):** 9 perguntas em linguagem acessível (ex: "Para que este medicamento é indicado?", "Como devo usar este medicamento?")
- **IT (Informações Técnicas):** seções clínicas (posologia, contraindicações, interações, farmacocinética)

O chunking genérico por tamanho perde a semântica de seção: um chunk de 400 tokens pode cruzar a fronteira entre posologia e contraindicações, degradando o recall em queries específicas.

Alternativas avaliadas:

| Estratégia | Motivo de descarte |
|---|---|
| Recursive char split puro | Ignora estrutura; chunks cruzam seções; citações imprecisas |
| Semantic chunking (embedding-based) | Custo 2× em embeddings durante ingestão; não determinístico; lento |
| Page-level | Chunks muito grandes (>2000 tokens); contexto difuso; sem citação por seção |

## Decisão

**Section-aware chunking** com 16 chaves canônicas mapeando os headers regex das bulas:

- Prefixo `IAP_*` para seções ao paciente (ex: `IAP_6_POSOLOGIA`, `IAP_8_REACOES_ADVERSAS`)
- Prefixo `IT_*` para seções técnicas (ex: `IT_INTERACOES_MEDICAMENTOSAS`, `IT_FARMACOCINETICA`)
- Seções com conteúdo ≤ 3500 chars → chunk único (`is_full_section=True`)
- Seções longas → recursive split com 1600 tokens, overlap 120, header da seção prefixado em cada sub-chunk
- Seções sem header detectável → chave `UNCLASSIFIED` (fallback 100% de cobertura)
- Bulas multi-produto (ex: Ritalina IR/LA) → campo `med_variant` nos metadados do payload

## Consequências

**Positivas:**
- Citações ricas `(bula_id, section_canonical, page_range)` exibidas no frontend
- Recall alto em queries específicas: `section_hint` filtra diretamente por chave canônica
- Cobertura total: UNCLASSIFIED cobre bulas com extração de texto irregular
- Ingestão determinística e idempotente (IDs UUIDv5 por chunk_id)

**Negativas / trade-offs:**
- Regex de detecção de headers pode falhar em bulas com formatação não-padrão (mitigado pelo fallback)
- 16 chaves canônicas precisam de manutenção se a Anvisa atualizar a RDC 47/2009
- Section hints no código do agente precisam ser mantidos em sincronia com as chaves