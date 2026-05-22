"""Admin routes — observability endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from bulas_assistant.services.trace_service import trace_service
from bulas_assistant.utils.handle_errors import handle_errors

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/traces/{trace_id}")
@handle_errors
async def get_trace(trace_id: str):
    """Return the structured trace for a given turn."""
    trace = await trace_service.get(trace_id)
    if not trace:
        raise HTTPException(
            status_code=404, detail=f"trace {trace_id} não encontrado"
        )
    return trace
