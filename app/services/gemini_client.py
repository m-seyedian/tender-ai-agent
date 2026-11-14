from pathlib import Path
from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logger import get_logger
from app.schemas.tender import AITenderEvaluation, TenderBase

logger = get_logger(__name__)

PROMPT_PATH = Path("prompts/tender_analysis.txt")


def _is_retryable_error(exc: BaseException) -> bool:
    """Ne relance la requête que pour les erreurs réseau ou serveur temporaires."""
    if isinstance(exc, errors.ClientError):
        return exc.code == 429  # Uniquement en cas de limite de quota (Rate Limit)
    return True


class GeminiTenderAnalyzer:
    """Client wrapper pour Gemini avec gestion des retries, timeout et Structured Outputs."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        if not settings.GEMINI_API_KEY:
            logger.error("Clé GEMINI_API_KEY manquante dans le fichier .env")
            raise ValueError("GEMINI_API_KEY non configurée.")

        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        self.model_name = model_name
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        if not PROMPT_PATH.exists():
            raise FileNotFoundError(f"Fichier prompt introuvable : {PROMPT_PATH}")
        return PROMPT_PATH.read_text(encoding="utf-8")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable_error),
        before_sleep=lambda retry_state: logger.warning(
            f"Échec temporaire de l'API Gemini. Nouvelle tentative ({retry_state.attempt_number}/3)..."
        ),
    )
    def analyze_tender(self, tender: TenderBase) -> AITenderEvaluation:
        """Envoie un appel d'offres à Gemini et retourne une évaluation validée par Pydantic."""
        logger.info(f"Début de l'analyse IA pour l'annonce {tender.id_annonce}...")

        formatted_prompt = self.prompt_template.format(
            id_annonce=tender.id_annonce,
            objet=tender.objet,
            acheteur=tender.acheteur or "Non précisé",
            departement=tender.departement or "Non précisé",
            type_marche=tender.type_marche or "Non précisé",
            date_limite_reponse=tender.date_limite_reponse or "Non précisée",
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AITenderEvaluation,
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        if not response.parsed:
            logger.error(f"Réponse vide ou invalide reçue de Gemini pour {tender.id_annonce}")
            raise ValueError("Impossible de parser la réponse structurée de Gemini.")

        logger.info(
            f"Analyse terminée avec succès pour {tender.id_annonce} | Score: {response.parsed.relevance_score}/10"
        )
        return response.parsed
