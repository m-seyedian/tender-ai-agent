from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, computed_field, field_validator


class MarketType(str, Enum):
    TRAVAUX = "TRAVAUX"
    SERVICES = "SERVICES"
    FOURNITURES = "FOURNITURES"


class TenderStatus(str, Enum):
    EN_ATTENTE = "En attente"
    A_ETUDIER = "À étudier"
    A_IGNORER = "À ignorer"


class TenderSearchParams(BaseModel):
    """Équivalent de la configuration lue depuis Airtable dans le nœud Read_Config_BOAMP"""
    period_days: int = Field(default=7, ge=1, le=90, description="Période de recherche en jours")
    market_types: List[MarketType] = Field(
        default=[MarketType.TRAVAUX, MarketType.SERVICES, MarketType.FOURNITURES]
    )
    target_cpv_codes: List[str] = Field(default_factory=list, description="Liste des codes CPV cibles à 8 chiffres")
    target_departments: List[str] = Field(default_factory=list, description="Codes des départements cibles (ex: 75, 91)")
    keywords: List[str] = Field(default_factory=list, description="Mots-clés à rechercher dans le contenu du PDF")


class TenderBase(BaseModel):
    """Modèle de base d'un appel d'offres extrait depuis l'API BOAMP"""
    id_annonce: str = Field(..., description="Identifiant unique de l'annonce (ex: 26-102635)")
    objet: str = Field(..., description="Titre ou objet de l'appel d'offres")
    date_parution: str = Field(..., description="Date de publication (YYYY-MM-DD)")
    date_limite_reponse: Optional[str] = Field(None, description="Date limite de remise des offres")
    acheteur: Optional[str] = Field(None, description="Nom de l'acheteur public (anonymisé pour les tests)")
    departement: Optional[str] = Field(None, description="Code du département")
    type_marche: Optional[str] = Field(None, description="Type de marché")
    source: str = Field(default="BOAMP")

    @field_validator("id_annonce", mode="before")
    @classmethod
    def normalize_id_annonce(cls, v: str) -> str:
        """Équivalent de la logique idAnnonce dans le nœud Filter_Market_BOAMP"""
        if not v:
            return "Non spécifié"
        return v if "-" in str(v) else f"26-{v}"

    @computed_field
    @property
    def pdf_url(self) -> Optional[str]:
        """Génération automatique du lien PDF (équivalent de generatePdfUrl dans n8n)"""
        if not self.id_annonce or "-" not in self.id_annonce or not self.date_parution:
            return None
        try:
            dt = datetime.fromisoformat(self.date_parution[:10])
            return f"https://www.boamp.fr/telechargements/FILES/PDF/{dt.year}/{dt.month:02d}/{self.id_annonce}.pdf"
        except ValueError:
            return None


class TenderAnalysisResult(TenderBase):
    """Résultat final de l'appel d'offres après extraction PDF, filtrage CPV et analyse IA"""
    matching_key: str
    code_cpv_list: List[str] = Field(default_factory=list)
    is_cpv_matched: bool = False
    is_department_matched: bool = False
    matched_keywords: List[str] = Field(default_factory=list)
    total_keyword_mentions: int = 0
    statut: TenderStatus = TenderStatus.EN_ATTENTE
    ai_summary: Optional[str] = Field(None, description="Résumé intelligent généré par Gemini")


class AITenderEvaluation(BaseModel):
    """Schéma structuré imposé à Gemini pour l'analyse d'un appel d'offres"""
    cleaned_title: str = Field(..., description="Titre de l'appel d'offres reformulé de manière claire et lisible")
    executive_summary: str = Field(..., description="Résumé exécutif en 2 à 3 phrases maximum")
    key_requirements: List[str] = Field(default_factory=list, description="Liste des exigences techniques ou administratives clés")
    risk_factors: List[str] = Field(default_factory=list, description="Points de vigilance ou risques identifiés")
    relevance_score: int = Field(..., ge=1, le=10, description="Note d'attractivité et de clarté de 1 à 10")
    recommended_action: Literal["À étudier", "À ignorer"] = Field(..., description="Décision recommandée")


# --- Nouveaux schémas pour les endpoints /tenders/clean et /tenders/batch ---

class FilterCriteriaSchema(BaseModel):
    """Critères de filtrage envoyés via l'API"""
    allowed_departments: List[str] = Field(default_factory=list, description="Ex: ['75', '91', '92']")
    target_cpv_prefixes: List[str] = Field(default_factory=list, description="Ex: ['7200', '48000000']")
    keywords: List[str] = Field(default_factory=list, description="Mots-clés recherchés")
    min_keyword_mentions: int = Field(default=1, ge=1)


class RawTenderInput(BaseModel):
    """Données brutes d'un appel d'offres à nettoyer et évaluer"""
    id_annonce: str = Field(..., min_length=1, description="Identifiant de l'annonce BOAMP")
    objet: str = Field(..., min_length=1, description="Titre brut pouvant contenir du HTML")
    description: Optional[str] = Field(None, description="Description détaillée brute")
    date_parution: Optional[str] = Field(None, description="Date de publication brute (ISO ou DD/MM/YYYY)")
    date_limite_reponse: Optional[str] = Field(None, description="Date limite brute (ISO ou DD/MM/YYYY)")
    acheteur: Optional[str] = Field(None, description="Nom de l'acheteur public")
    departement: Optional[str] = Field(None, description="Code département")
    type_marche: Optional[str] = Field(None, description="Type de marché")
    raw_cpv_codes: List[str] = Field(default_factory=list, description="Codes CPV bruts éventuels")


class CleanTenderRequest(BaseModel):
    """Requête pour l'endpoint POST /tenders/clean"""
    tender: RawTenderInput
    criteria: FilterCriteriaSchema = Field(default_factory=FilterCriteriaSchema)


class CleanedTenderOutput(BaseModel):
    """Résultat déterministe après nettoyage et validation (sans appel IA)"""
    matching_key: str
    id_annonce: str
    cleaned_objet: str
    cleaned_description: str
    normalized_date_parution: Optional[str] = None
    normalized_date_limite: Optional[str] = None
    estimated_amount_eur: Optional[float] = None
    acheteur: Optional[str] = None
    departement: Optional[str] = None
    type_marche: Optional[str] = None
    extracted_cpv_codes: List[str] = Field(default_factory=list)
    matched_keywords: Dict[str, int] = Field(default_factory=dict)
    total_keyword_mentions: int = 0
    is_eligible: bool
    rejection_reason: Optional[str] = None


class BatchTenderRequest(BaseModel):
    """Requête batch acceptant jusqu'à 50 annonces avec plafond configurable d'appels IA"""
    tenders: List[RawTenderInput] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Liste de 1 à 50 appels d'offres bruts à filtrer et analyser",
    )
    criteria: FilterCriteriaSchema = Field(default_factory=FilterCriteriaSchema)
    max_ai_calls: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Nombre maximum d'appels envoyés à Gemini dans ce batch (protection du quota gratuit)",
    )


class BatchTenderItemResult(BaseModel):
    """Résultat individuel pour chaque appel d'offres du batch"""
    id_annonce: str
    matching_key: str
    status: Literal["analyzed", "skipped", "quota_deferred", "error"]
    cleaned_data: CleanedTenderOutput
    ai_evaluation: Optional[AITenderEvaluation] = None
    error_detail: Optional[str] = None


class BatchTenderResponse(BaseModel):
    """Réponse globale de l'endpoint POST /tenders/batch"""
    total_received: int
    total_eligible: int
    total_analyzed: int
    total_skipped: int
    total_deferred: int
    total_failed: int
    results: List[BatchTenderItemResult]
