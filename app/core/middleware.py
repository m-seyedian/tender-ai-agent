import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logger import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware journalisant chaque requête HTTP entrante avec un ID unique et sa durée d'exécution."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        logger.info(f"[{request_id}] --> {request.method} {request.url.path}")

        try:
            response = await call_next(request)
        except Exception as exc:
            process_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request_id}] <-- ERREUR NON GÉRÉE sur {request.method} {request.url.path} "
                f"après {process_time_ms:.2f}ms : {exc}"
            )
            raise

        process_time_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time_ms:.2f}ms"

        logger.info(
            f"[{request_id}] <-- {response.status_code} {request.method} {request.url.path} "
            f"({process_time_ms:.2f}ms)"
        )
        return response
