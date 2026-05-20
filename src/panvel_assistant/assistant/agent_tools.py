"""Centralized tool registry for the Panvel assistant.

All LangChain tools the agent has access to are defined here. Each tool is a
closure over the injected dependencies returned by :func:`build_tools`,
mirroring the pattern used in ``helper-backend``.
"""

from __future__ import annotations

import json
import time

from langchain_core.tools import tool

from panvel_assistant.models.tool import (
    BuscarBulasInput,
    BuscarFiliaisInput,
    BuscarFiliaisOutput,
    DetalhesFilialInput,
    DetalhesFilialOutput,
    ListarCidadesOutput,
    ToolErrorPayload,
)
from panvel_assistant.services.filiais_service import FiliaisService
from panvel_assistant.services.rag_service import RAGService
from panvel_assistant.utils.exceptions import (
    InvalidRequestError,
    ResourceNotFoundError,
)
from panvel_assistant.utils.logger import get_logger

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
        """Busca filiais Panvel no Paraná aplicando filtros opcionais.

        Use esta tool quando o usuário quiser encontrar lojas que atendam critérios
        como: cidade específica, serviços (panvel_clinic / delivery / estacionamento /
        atendimento_24_horas), tipo (BAIRRO / CENTRO / SHOPPING / MALL / SUPERMERCADO),
        faixa de operação, ou metragem mínima.

        Retorna lista resumida. Para detalhes completos de uma filial específica, use
        detalhes_filial(codigo_filial).
        """
        started = time.perf_counter()
        try:
            total, filiais = filiais_service.buscar(
                cidade=cidade,
                servicos=servicos,  # type: ignore[arg-type]
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
                    message=f"'{cidade}' não está no Paraná atendido pela Panvel.",
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
        """Retorna o cadastro completo de uma filial específica pelo código.

        Use depois que o usuário identificar uma filial (via buscar_filiais) e quiser
        detalhes adicionais (metragem, todos os serviços, faixa de vida, etc.).
        """
        started = time.perf_counter()
        try:
            f = filiais_service.detalhar(codigo_filial)
            return DetalhesFilialOutput(filial=f).model_dump_json()
        except ResourceNotFoundError:
            return ToolErrorPayload(
                error="codigo_invalido",
                message=f"Filial '{codigo_filial}' não existe no cadastro.",
                hint={"sugestao": "use buscar_filiais para listar códigos válidos"},
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
        """Lista todas as cidades do Paraná onde existem filiais Panvel.

        Use ANTES de buscar_filiais quando o usuário mencionar uma cidade que pode
        não estar coberta — esta tool confirma o escopo.
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
        section_hint: str | None = None,
        patient_facing_only: bool = True,
        k: int = 4,
    ) -> str:
        """Recupera trechos das bulas Anvisa indexadas via hybrid search.

        PRÉ-REQUISITO OBRIGATÓRIO
            SEMPRE chame ``listar_medicamentos_disponiveis`` ANTES desta tool
            para obter o nome canônico correto. Confirme o nome exato na lista
            antes de passar ``med_name`` — o usuário frequentemente usa apelidos
            ou nomes parciais.

        QUANDO USAR
            Sempre que a pergunta for sobre medicamentos: indicação, posologia,
            dose, contraindicação, reação adversa, interação, armazenamento,
            mecanismo, superdose, esquecimento de dose. Chame esta tool para
            toda pergunta farmacológica.

        ARGUMENTOS
        - ``query``: pergunta/termos em português. Pode ser a pergunta inteira
          do usuário ou termos-chave.
        - ``med_name`` (opcional): nome do medicamento como o usuário citou
          (ex.: "Ritalina", "Losartana"). Aplica MATCH EXATO contra o nome
          canônico do payload. Se o usuário usou apelido/nome parcial, prefira
          o nome canônico (rode ``listar_medicamentos_disponiveis`` se estiver
          em dúvida).
        - ``section_hint`` (opcional): use quando o intent for claro:
            posologia / dose / como tomar  -> IAP_6_POSOLOGIA
            reações / efeitos colaterais   -> IAP_8_REACOES_ADVERSAS
            contraindicações / não usar    -> IAP_3_CONTRAINDICACOES
            interações                     -> IT_INTERACOES_MEDICAMENTOSAS
            armazenamento / como guardar   -> IAP_5_ARMAZENAMENTO
            indicação / para que serve     -> IAP_1_INDICACOES
            esqueci uma dose               -> IAP_7_ESQUECIMENTO_DOSE
            superdose / overdose           -> IAP_9_SUPERDOSE
            mecanismo / como funciona      -> IAP_2_MECANISMO
        - ``patient_facing_only`` (default true): restringe a seções IAP_*
          (linguagem ao paciente). Mantenha true para perguntas do paciente;
          passe false quando o usuário declarar ser profissional de saúde ou
          pedir informação técnica.
        - ``k`` (default 4, max 10): nº de chunks após dedup.

        RETORNO
            Caminho feliz:
              {"matches": [{chunk_id, med_name, section_canonical,
                            section_label, is_full_section, text, ...}, ...],
               "total": N}

            Erros estruturados (NÃO são exceção — a tool retorna JSON normal):
              {"error": "medicamento_nao_encontrado",
               "message": "Nenhuma bula encontrada para '<med_name>'.",
               "hint": {"medicamentos_disponiveis": [...nomes canônicos...]}}
              -> ocorre quando você passou ``med_name`` e a busca filtrada
                 retornou 0. RETENTE com um nome da lista do hint, ou omita
                 ``med_name`` e confie no embedding.

              {"error": "nenhum_resultado",
               "message": "Nenhum trecho relevante encontrado nas bulas indexadas.",
               "hint": {"sugestao": "use listar_medicamentos_disponiveis"}}
              -> ocorre quando NÃO passou ``med_name`` e ainda assim não veio
                 nada. Informe ao usuário que o medicamento provavelmente não
                 está no corpus indexado.

        COMO RESPONDER
            Use APENAS texto vindo de ``matches[].text``. Cite no formato
            [Medicamento — section_label], ex.: [Ritalina — Como devo usar].
            Sempre inclua o disclaimer médico padrão. Se vier ``error`` ou
            matches vazio, informe que não encontrou a informação solicitada.
        """
        started = time.perf_counter()
        try:
            chunks = await rag_service.retrieve(
                query=query,
                k=k,
                med_name=med_name,
                section_hint=section_hint,
                patient_facing_only=patient_facing_only,
            )
            if not chunks:
                if med_name:
                    available = await rag_service.list_medicamentos()
                    return ToolErrorPayload(
                        error="medicamento_nao_encontrado",
                        message=f"Nenhuma bula encontrada para '{med_name}'.",
                        hint={"medicamentos_disponiveis": available},
                    ).model_dump_json()
                return ToolErrorPayload(
                    error="nenhum_resultado",
                    message=(
                        "Nenhum trecho relevante encontrado nas bulas indexadas "
                        "para esta consulta."
                    ),
                    hint={
                        "sugestao": (
                            "use listar_medicamentos_disponiveis para ver "
                            "o que esta indexado"
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
                    "section_hint": section_hint,
                    "patient_facing_only": patient_facing_only,
                },
            )

    @tool("listar_medicamentos_disponiveis")
    async def listar_medicamentos_disponiveis() -> str:
        """Lista todos os medicamentos com bula indexada (nomes canônicos).

        SEMPRE chame esta tool ANTES de ``buscar_bulas`` para confirmar o nome
        canônico do medicamento. O usuário frequentemente usa apelidos ou nomes
        parciais (ex.: "Ritalina") que diferem do nome canônico indexado (ex.:
        "Ritalina Metilfenidato"). Use a lista retornada para escolher o nome
        exato antes de passar ``med_name`` a ``buscar_bulas``.

        A lista vem direto do índice Qdrant — sempre alinhada com o corpus atual.

        Retorna ``{"medicamentos": [...], "total": N}``.
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
