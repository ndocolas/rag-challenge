"""Centralized tool registry for the Bulas assistant.

All LangChain tools the agent has access to are defined here. Each tool is a
closure over the injected dependencies returned by :func:`build_tools`,
mirroring the pattern used in ``helper-backend``.
"""

from __future__ import annotations

import json
import time

from langchain_core.tools import tool

from bulas_assistant.models.tool_models import (
    BuscarBulasInput,
    BuscarFiliaisInput,
    BuscarFiliaisOutput,
    DetalhesFilialInput,
    DetalhesFilialOutput,
    ListarCidadesOutput,
    ToolErrorPayload,
)
from bulas_assistant.services.filiais_service import FiliaisService
from bulas_assistant.services.rag_service import RAGService
from bulas_assistant.utils.exceptions import (
    InvalidRequestError,
    ResourceNotFoundError,
)
from bulas_assistant.utils.logger import get_logger

logger = get_logger(__name__)
_logger_extra = {"component.name": "AgentTools", "component.version": "v1"}


def build_tools(
    filiais_service: FiliaisService,
    rag_service: RAGService,
) -> list:
    """Build all agent tools with injected dependencies.

    Args:
        filiais_service: In-memory branch catalog service used by branch tools.
        rag_service: Hybrid retrieval service for the ``buscar_bulas`` tool.

    Returns:
        List of LangChain tools available to the agent.
    """

    @tool("buscar_filiais", args_schema=BuscarFiliaisInput)
    def buscar_filiais(
        cidade: str | None = None,
        servicos: list[str] | None = None,
        tipo_estabelecimento: str | None = None,
        faixa_vida: str | None = None,
        min_metragem: float | None = None,
        limit: int = 10,
    ) -> str:
        """Search branches in Paraná applying optional filters.

        Use this tool when the user wants to find stores that match criteria
        such as: specific city, services (clinic / delivery / estacionamento /
        atendimento_24_horas), type (BAIRRO / CENTRO / SHOPPING / MALL / SUPERMERCADO),
        operating range, or minimum floor area.

        Returns a summarized list. For full details of a specific branch, use
        detalhes_filial(codigo_filial).
        """
        started = time.perf_counter()
        try:
            total, filiais = filiais_service.buscar(
                cidade=cidade,
                servicos=servicos,
                tipo_estabelecimento=tipo_estabelecimento,
                faixa_vida=faixa_vida,
                min_metragem=min_metragem,
                limit=limit,
            )
            return BuscarFiliaisOutput(
                total_match=total, returned=len(filiais), filiais=filiais
            ).model_dump_json()
        except InvalidRequestError as exc:
            if str(exc).startswith("cidade_nao_encontrada:"):
                return ToolErrorPayload(
                    error="cidade_nao_encontrada",
                    message=f"'{cidade}' is not in the Paraná region served by branches.",
                    hint={"cidades_disponiveis": filiais_service.listar_cidades()},
                ).model_dump_json()
            raise
        finally:
            logger.info(
                "tool buscar_filiais",
                extra={
                    **_logger_extra,
                    "step": "tool",
                    "tool": "buscar_filiais",
                    "latency_ms": (time.perf_counter() - started) * 1000,
                },
            )

    @tool("detalhes_filial", args_schema=DetalhesFilialInput)
    def detalhes_filial(codigo_filial: str) -> str:
        """Return the full record of a specific branch by its code.

        Use after the user has identified a branch (via buscar_filiais) and wants
        additional details (floor area, all services, operating range, etc.).
        """
        started = time.perf_counter()
        try:
            f = filiais_service.detalhar(codigo_filial)
            return DetalhesFilialOutput(filial=f).model_dump_json()
        except ResourceNotFoundError:
            return ToolErrorPayload(
                error="codigo_invalido",
                message=f"Branch '{codigo_filial}' does not exist in the registry.",
                hint={"sugestao": "use buscar_filiais to list valid codes"},
            ).model_dump_json()
        finally:
            logger.info(
                "tool detalhes_filial",
                extra={
                    **_logger_extra,
                    "step": "tool",
                    "tool": "detalhes_filial",
                    "latency_ms": (time.perf_counter() - started) * 1000,
                },
            )

    @tool("listar_cidades_atendidas")
    def listar_cidades_atendidas() -> str:
        """List all cities in Paraná where branches exist.

        Use BEFORE buscar_filiais when the user mentions a city that may
        not be covered — this tool confirms the scope.
        """
        started = time.perf_counter()
        cidades = filiais_service.listar_cidades()
        logger.info(
            "tool listar_cidades",
            extra={
                **_logger_extra,
                "step": "tool",
                "tool": "listar_cidades_atendidas",
                "latency_ms": (time.perf_counter() - started) * 1000,
            },
        )
        return ListarCidadesOutput(cidades=cidades, total=len(cidades)).model_dump_json()

    @tool("buscar_bulas", args_schema=BuscarBulasInput)
    async def buscar_bulas(
        query: str,
        med_name: str | None = None,
        med_variant: str | None = None,
        section_hint: str | None = None,
        patient_facing_only: bool = True,
        k: int = 4,
    ) -> str:
        """Retrieve excerpts from indexed Anvisa package inserts via hybrid search.

        MANDATORY PREREQUISITE
            ALWAYS call ``listar_medicamentos_disponiveis`` BEFORE this tool
            to obtain the correct canonical name. Confirm the exact name in the list
            before passing ``med_name`` — users frequently use nicknames
            or partial names.

        WHEN TO USE
            Whenever the question is about medications: indication, dosage,
            dose, contraindication, adverse reaction, interaction, storage,
            mechanism, overdose, missed dose. Call this tool for
            every pharmacological question.

        ARGUMENTS
        - ``query``: question/terms in the user's language. Can be the full user
          question or key terms.
        - ``med_name`` (optional): medication name as the user cited it
          (e.g. "Ritalin", "Losartan"). Applies EXACT MATCH against the
          canonical name in the payload. If the user used a nickname/partial name,
          prefer the canonical name (run ``listar_medicamentos_disponiveis`` if
          unsure).
        - ``section_hint`` (optional): use when intent is clear:
            dosage / dose / how to take    -> IAP_6_POSOLOGIA
            reactions / side effects       -> IAP_8_REACOES_ADVERSAS
            contraindications / do not use -> IAP_3_CONTRAINDICACOES
            interactions                   -> IT_INTERACOES_MEDICAMENTOSAS
            storage / how to store         -> IAP_5_ARMAZENAMENTO
            indication / what it's for     -> IAP_1_INDICACOES
            missed a dose                  -> IAP_7_ESQUECIMENTO_DOSE
            overdose                       -> IAP_9_SUPERDOSE
            mechanism / how it works       -> IAP_2_MECANISMO
        - ``patient_facing_only`` (default true): restricts to IAP_* sections
          (patient-language). Keep true for patient questions; pass false when
          the user declares being a healthcare professional or requests technical
          information.
        - ``k`` (default 4, max 10): number of chunks after dedup.

        RETURN
            Happy path:
              {"matches": [{chunk_id, med_name, section_canonical,
                            section_label, is_full_section, text, ...}, ...],
               "total": N}

            Structured errors (NOT exceptions — the tool returns normal JSON):
              {"error": "medicamento_nao_encontrado",
               "message": "No package insert found for '<med_name>'.",
               "hint": {"medicamentos_disponiveis": [...canonical names...]}}
              -> occurs when you passed ``med_name`` and the filtered search
                 returned 0. RETRY with a name from the hint list, or omit
                 ``med_name`` and rely on the embedding.

              {"error": "nenhum_resultado",
               "message": "No relevant excerpt found in the indexed package inserts.",
               "hint": {"suggestion": "use listar_medicamentos_disponiveis"}}
              -> occurs when ``med_name`` was NOT passed and still nothing came
                 back. Inform the user that the medication is probably not
                 in the indexed corpus.

        HOW TO RESPOND
            Use ONLY text from ``matches[].text``. Cite in the format
            [Medication — section_label], e.g.: [Ritalin — How should I use it].
            Always include the standard medical disclaimer. If an ``error`` or
            empty matches comes back, inform that the requested information was
            not found.
        """
        started = time.perf_counter()
        try:
            chunks = await rag_service.retrieve(
                query=query,
                k=k,
                med_name=med_name,
                med_variant=med_variant,
                section_hint=section_hint,
                patient_facing_only=patient_facing_only,
            )
            if not chunks:
                if med_name:
                    available = await rag_service.list_medicamentos()
                    return ToolErrorPayload(
                        error="medicamento_nao_encontrado",
                        message=f"No package insert found for '{med_name}'.",
                        hint={"medicamentos_disponiveis": available},
                    ).model_dump_json()
                return ToolErrorPayload(
                    error="nenhum_resultado",
                    message=(
                        "No relevant excerpt found in the indexed package inserts "
                        "for this query."
                    ),
                    hint={
                        "suggestion": (
                            "use listar_medicamentos_disponiveis to see "
                            "what is indexed"
                        )
                    },
                ).model_dump_json()
            payload = RAGService.format_tool_payload(chunks)
            return json.dumps(payload, ensure_ascii=False)
        finally:
            logger.info(
                "tool buscar_bulas",
                extra={
                    **_logger_extra,
                    "step": "tool",
                    "tool": "buscar_bulas",
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "k": k,
                    "med_name": med_name,
                    "med_variant": med_variant,
                    "section_hint": section_hint,
                    "patient_facing_only": patient_facing_only,
                },
            )

    @tool("listar_medicamentos_disponiveis")
    async def listar_medicamentos_disponiveis() -> str:
        """List all medications with an indexed package insert.

        ALWAYS call this tool BEFORE ``buscar_bulas`` to confirm the canonical
        medication name. Users frequently use nicknames or partial names
        (e.g. "Ritalin") that differ from the indexed canonical name (e.g.
        "Ritalina Metilfenidato").

        MULTI-VARIANT ENTRIES: when a PDF contains multiple products, each variant
        appears as a separate entry in the format "Base Name — VARIANT NAME".
        Example: "Ritalina Metilfenidato — RITALINA LA".
        For these entries, pass:
          med_name = "Ritalina Metilfenidato"
          med_variant = "RITALINA LA"
        to ``buscar_bulas``.

        Returns ``{"medicamentos": [...], "total": N}``.
        """
        started = time.perf_counter()
        try:
            meds = await rag_service.list_medicamentos()
            return json.dumps(
                {"medicamentos": meds, "total": len(meds)},
                ensure_ascii=False,
            )
        finally:
            logger.info(
                "tool listar_medicamentos",
                extra={
                    **_logger_extra,
                    "step": "tool",
                    "tool": "listar_medicamentos_disponiveis",
                    "latency_ms": (time.perf_counter() - started) * 1000,
                },
            )

    return [
        buscar_filiais,
        detalhes_filial,
        listar_cidades_atendidas,
        buscar_bulas,
        listar_medicamentos_disponiveis,
    ]
