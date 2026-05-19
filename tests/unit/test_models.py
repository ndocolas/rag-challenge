"""Unit tests for the project's Pydantic schemas."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from panvel_assistant.models.bula import BulaChunk, BulaMetadata
from panvel_assistant.models.chat import (
    ChatMessage,
    ChatRequest,
    Citation,
    StreamEvent,
)
from panvel_assistant.models.filial import FilialResumo
from panvel_assistant.models.tool import BuscarFiliaisInput, ToolErrorPayload


def test_chat_message_serialization():
    original = ChatMessage(
        role="assistant",
        content="sure, I can help",
        tool_calls=[{"id": "call_1", "name": "buscar_filiais", "args": {"cidade": "CURITIBA"}}],
        name="buscar_filiais",
    )

    raw = original.model_dump_json()
    rebuilt = ChatMessage.model_validate_json(raw)

    assert rebuilt == original
    assert rebuilt.role == "assistant"
    assert rebuilt.tool_calls is not None
    assert rebuilt.tool_calls[0]["name"] == "buscar_filiais"


def test_chat_request_validation_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="")


def test_chat_request_validation_message_too_long():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="x" * 4001)


def test_chat_request_validation_missing_session_id():
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"message": "oi"})


def test_citation_required_fields():
    with pytest.raises(ValidationError):
        Citation.model_validate({"med_name": "Ritalina"})

    cit = Citation(
        bula_id="927100",
        med_name="Ritalina",
        section_canonical="IAP_3_CONTRAINDICACOES",
        section_label="When should I avoid taking this medication?",
        snippet="Contraindicated for patients with...",
    )
    assert cit.bula_id == "927100"
    assert cit.med_variant is None
    assert cit.source_page is None


def test_stream_event_discriminator():
    event_types = ["token", "tool_call", "tool_result", "sources", "done", "error"]
    payloads = ["t", {"name": "x"}, {"ok": True}, [], None, "boom"]

    for et, payload in zip(event_types, payloads, strict=True):
        ev = StreamEvent(event_type=et, payload=payload)  # type: ignore[arg-type]
        assert ev.event_type == et
        assert isinstance(ev.timestamp, datetime)
        assert ev.timestamp.tzinfo is not None
        assert ev.timestamp.utcoffset() == UTC.utcoffset(None)

    with pytest.raises(ValidationError):
        StreamEvent(event_type="bogus", payload=None)  # type: ignore[arg-type]


def test_filial_servico_enum():
    with pytest.raises(ValidationError):
        FilialResumo(
            codigo_filial="042",
            localidade="CURITIBA",
            tipo_estabelecimento="BAIRRO",
            servicos_ativos=["raio_x"],  # type: ignore[list-item]
        )

    ok = FilialResumo(
        codigo_filial="042",
        localidade="CURITIBA",
        tipo_estabelecimento="BAIRRO",
        servicos_ativos=["delivery", "panvel_clinic"],
    )
    assert ok.servicos_ativos == ["delivery", "panvel_clinic"]


def test_tool_input_buscar_filiais_optional_fields():
    empty = BuscarFiliaisInput()
    assert empty.cidade is None
    assert empty.servicos is None
    assert empty.tipo_estabelecimento is None
    assert empty.faixa_vida is None
    assert empty.min_metragem is None
    assert empty.limit == 10

    with pytest.raises(ValidationError):
        BuscarFiliaisInput(limit=0)
    with pytest.raises(ValidationError):
        BuscarFiliaisInput(limit=51)


def test_tool_error_payload_structure():
    minimal = ToolErrorPayload(error="cidade_nao_encontrada", message="City not found.")
    assert minimal.hint is None

    with_hint = ToolErrorPayload(
        error="codigo_invalido",
        message="Code not found.",
        hint={"cidades_disponiveis": ["CURITIBA", "LONDRINA"]},
    )
    assert with_hint.hint == {"cidades_disponiveis": ["CURITIBA", "LONDRINA"]}


def test_bula_metadata_section_canonical_literal():
    with pytest.raises(ValidationError):
        BulaMetadata(
            bula_id="927100",
            med_name="Ritalina",
            section_canonical="FOO",  # type: ignore[arg-type]
            chunk_idx=0,
            patient_facing=True,
        )

    meta = BulaMetadata(
        bula_id="927100",
        med_name="Ritalina",
        section_canonical="IAP_3_CONTRAINDICACOES",
        chunk_idx=0,
        patient_facing=True,
    )
    assert meta.section_canonical == "IAP_3_CONTRAINDICACOES"


def test_bula_chunk_id_format():
    meta = BulaMetadata(
        bula_id="927100",
        med_name="Ritalina",
        section_canonical="IAP_6_POSOLOGIA",
        chunk_idx=0,
        patient_facing=True,
    )
    chunk = BulaChunk(
        chunk_id="927100__IAP_6_POSOLOGIA__0",
        text="Adults: one tablet per day...",
        metadata=meta,
    )
    parts = chunk.chunk_id.split("__")
    assert parts == ["927100", "IAP_6_POSOLOGIA", "0"]
    assert parts[0] == meta.bula_id
    assert parts[1] == meta.section_canonical
    assert int(parts[2]) == meta.chunk_idx
