"""Decorator that maps domain exceptions to standardized HTTP responses."""

import inspect
from functools import wraps

from fastapi import HTTPException
from pydantic import ValidationError

from panvel_assistant.utils.exceptions import (
    InvalidRequestError,
    LLMProviderError,
    ResourceNotFoundError,
    RetrievalError,
    ToolExecutionError,
)
from panvel_assistant.utils.logger import get_logger

logger = get_logger(__name__)


def handle_errors(func):
    """Wrap a FastAPI route, translating domain errors into HTTPException.

    Works with both sync and async handlers. Provider/external-resource errors
    are logged before being surfaced as 503 to preserve operational context.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            if inspect.isawaitable(res):
                return await res
            return res
        except ResourceNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except InvalidRequestError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors()) from e
        except (LLMProviderError, RetrievalError, ToolExecutionError) as e:
            logger.exception("provider/resource error: %s", e)
            raise HTTPException(status_code=503, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("unexpected error")
            raise HTTPException(status_code=500, detail="Internal server error") from e

    return wrapper
