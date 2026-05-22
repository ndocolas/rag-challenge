"""Domain exception hierarchy for the Bulas Assistant."""


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


class SessionBusyError(AppError):
    """Another turn is already in flight for the same session (HTTP 409)."""


class RateLimitedError(AppError):
    """Too many requests for the active bucket (HTTP 429).

    Carries optional ``retry_after`` and ``limit`` so the route can lift them
    into response headers (``Retry-After``, ``X-RateLimit-*``).
    """

    def __init__(
        self,
        message: str = "too many requests",
        *,
        retry_after: int = 60,
        limit: int | None = None,
        count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.limit = limit
        self.count = count
