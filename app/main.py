import json
from pathlib import Path
from typing import List
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_gemini_analyzer
from app.api.routes import router as core_router
from app.core.config import settings
from app.core.logger import get_logger
from app.core.middleware import RequestLoggingMiddleware
from app.schemas.tender import AITenderEvaluation, TenderBase
from app.services.gemini_client import GeminiTenderAnalyzer

logger = get_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.APP_VERSION,
    description="API de veille et d'analyse intelligente des appels d'offres publics (BOAMP)",
)

# Enregistrement du middleware de journalisation et de chronométrage
app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Capture toute erreur serveur imprévue et renvoie une réponse JSON standardisée."""
    logger.error(f"Erreur serveur inattendue sur {request.url.path} : {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Une erreur interne est survenue lors du traitement de la requête."},
    )


# Inclusion des routes principales (/health, /tenders/clean, /tenders/batch)
app.include_router(core_router)

SAMPLE_DATA_PATH = Path("data/sample_boamp_tenders.json")


def load_sample_tenders() -> List[TenderBase]:
    if not SAMPLE_DATA_PATH.exists():
        raise FileNotFoundError("Fichier de données de test introuvable.")
    raw_data = json.loads(SAMPLE_DATA_PATH.read_text(encoding="utf-8"))
    return [TenderBase(**item) for item in raw_data]


@app.get("/", tags=["Santé"])
def root_status():
    return {
        "status": "ok",
        "app": settings.PROJECT_NAME,
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
    }


@app.get("/api/v1/tenders/samples", response_model=List[TenderBase], tags=["Appels d'offres"])
def get_sample_tenders():
    """Retourne les 10 appels d'offres anonymisés avec génération automatique des liens PDF"""
    return load_sample_tenders()


@app.post(
    "/api/v1/tenders/samples/{index}/analyze",
    response_model=AITenderEvaluation,
    tags=["Analyse IA"],
)
def analyze_sample_tender(
    index: int,
    analyzer: GeminiTenderAnalyzer = Depends(get_gemini_analyzer),
):
    """Analyse un appel d'offres de démonstration (index 0 à 9) avec Gemini"""
    tenders = load_sample_tenders()
    if index < 0 or index >= len(tenders):
        raise HTTPException(
            status_code=404,
            detail=f"Index invalide. Choisissez un nombre entre 0 et {len(tenders) - 1}.",
        )

    return analyzer.analyze_tender(tenders[index])

from app.api.documents import router as documents_router, search_router
app.include_router(documents_router)
app.include_router(search_router)

