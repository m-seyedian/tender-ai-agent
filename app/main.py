import json
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException

from app.core.config import settings
from app.core.logger import get_logger
from app.schemas.tender import AITenderEvaluation, TenderBase
from app.services.gemini_client import GeminiTenderAnalyzer

logger = get_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.APP_VERSION,
    description="API de veille et d'analyse intelligente des appels d'offres publics (BOAMP)",
)

SAMPLE_DATA_PATH = Path("data/sample_boamp_tenders.json")


def load_sample_tenders() -> List[TenderBase]:
    if not SAMPLE_DATA_PATH.exists():
        raise FileNotFoundError("Fichier de données de test introuvable.")
    raw_data = json.loads(SAMPLE_DATA_PATH.read_text(encoding="utf-8"))
    return [TenderBase(**item) for item in raw_data]


@app.get("/", tags=["Santé"])
def health_check():
    return {
        "status": "ok",
        "app": settings.PROJECT_NAME,
        "version": settings.APP_VERSION,
        "env": settings.ENVIRONMENT,
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
def analyze_sample_tender(index: int):
    """Analyse un appel d'offres de démonstration (index 0 à 9) avec Gemini"""
    tenders = load_sample_tenders()
    if index < 0 or index >= len(tenders):
        raise HTTPException(
            status_code=404,
            detail=f"Index invalide. Choisissez un nombre entre 0 et {len(tenders) - 1}.",
        )

    analyzer = GeminiTenderAnalyzer()
    return analyzer.analyze_tender(tenders[index])
