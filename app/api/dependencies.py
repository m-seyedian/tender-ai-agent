from functools import lru_cache
from fastapi import HTTPException, status

from app.core.logger import get_logger
from app.services.gemini_client import GeminiTenderAnalyzer

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _create_cached_analyzer() -> GeminiTenderAnalyzer:
    """Crée une instance unique (Singleton) du client Gemini en mémoire."""
    return GeminiTenderAnalyzer()


def get_gemini_analyzer() -> GeminiTenderAnalyzer:
    """Dépendance FastAPI injectable dans les routes (permet le mocking dans les tests)."""
    try:
        return _create_cached_analyzer()
    except (ValueError, FileNotFoundError) as exc:
        logger.error(f"Impossible d'initialiser GeminiTenderAnalyzer : {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service d'analyse IA indisponible : {exc}",
        ) from exc
