import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from app.core.config import settings
from app.schemas.tender import MarketType, TenderBase

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API d'acquisition et d'analyse intelligente des appels d'offres BOAMP",
    version="0.1.0",
)

SAMPLE_DATA_PATH = Path("data/sample_boamp_tenders.json")


def load_sample_tenders() -> List[TenderBase]:
    if not SAMPLE_DATA_PATH.exists():
        return []
    raw_data = json.loads(SAMPLE_DATA_PATH.read_text(encoding="utf-8"))
    return [TenderBase(**item) for item in raw_data]


@app.get("/")
def health_check():
    return {"status": "online", "project": settings.PROJECT_NAME}


@app.get("/api/v1/tenders", response_model=List[TenderBase])
def get_tenders(
    market_type: Optional[MarketType] = Query(None, description="Filtrer par type de marché"),
    department: Optional[str] = Query(None, description="Filtrer par code département (ex: 75)"),
    limit: int = Query(10, ge=1, le=50, description="Nombre maximum de résultats"),
):
    """Récupère la liste des appels d'offres avec filtres optionnels (Query Parameters)"""
    tenders = load_sample_tenders()
    if market_type:
        tenders = [t for t in tenders if t.type_marche == market_type.value]
    if department:
        tenders = [t for t in tenders if t.departement == department]
    return tenders[:limit]


@app.get("/api/v1/tenders/{id_annonce}", response_model=TenderBase)
def get_tender_by_id(id_annonce: str):
    """Récupère un appel d'offres spécifique par son identifiant (Path Parameter)"""
    tenders = load_sample_tenders()
    for tender in tenders:
        if tender.id_annonce == id_annonce:
            return tender
    raise HTTPException(status_code=404, detail="Appel d'offres non trouvé")