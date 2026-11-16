from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_gemini_analyzer
from app.core.config import settings
from app.core.logger import get_logger
from app.schemas.tender import (
    BatchTenderRequest,
    BatchTenderResponse,
    CleanedTenderOutput,
    CleanTenderRequest,
)
from app.services.batch_processor import clean_single_tender, process_tenders_batch
from app.services.gemini_client import GeminiTenderAnalyzer

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", tags=["Santé"])
def health_check():
    """Vérifie l'état de santé de l'API et la présence de la configuration IA."""
    gemini_configured = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip())
    return {
        "status": "ok" if gemini_configured else "degraded",
        "app": settings.PROJECT_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "gemini_api_configured": gemini_configured,
    }


@router.post(
    "/tenders/clean",
    response_model=CleanedTenderOutput,
    tags=["Appels d'offres"],
)
def clean_tender_endpoint(payload: CleanTenderRequest):
    """Nettoie, normalise et évalue l'éligibilité d'un appel d'offres brut (sans appel IA)."""
    try:
        return clean_single_tender(payload.tender, payload.criteria)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/tenders/batch",
    response_model=BatchTenderResponse,
    tags=["Analyse IA"],
)
async def batch_analyze_tenders(
    payload: BatchTenderRequest,
    analyzer: GeminiTenderAnalyzer = Depends(get_gemini_analyzer),
):
    """Filtre un lot d'appels d'offres et analyse les plus pertinents via Gemini (concurrence bornée)."""
    try:
        return await process_tenders_batch(
            tenders=payload.tenders,
            criteria=payload.criteria,
            max_ai_calls=payload.max_ai_calls,
            analyzer=analyzer,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
