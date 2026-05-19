"""Domain exception hierarchy for the Panvel Assistant."""


class AppError(Exception):
    """Base class for domain errors."""


class ResourceNotFoundError(AppError):
    """Requested resource was not found (mapped to HTTP 404)."""


class InvalidRequestError(AppError):
    """Request payload/arguments are invalid (mapped to HTTP 400)."""


class LLMProviderError(AppError):
    """External LLM provider failure (mapped to HTTP 503)."""


class RetrievalError(AppError):
    """Retrieval / vector-store pipeline failure (mapped to HTTP 503)."""


class ToolExecutionError(AppError):
    """Failure while executing an assistant tool (mapped to HTTP 503)."""
