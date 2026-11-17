import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.schemas.tender import (
    AITenderEvaluation,
    FilterCriteriaSchema,
    RawTenderInput,
    TenderBase,
)
from app.services.batch_processor import clean_single_tender, process_tenders_batch
from app.services.gemini_client import GeminiTenderAnalyzer


class SimulatedGeminiAnalyzer:
    """Simule une réponse Gemini calibrée pour valider le pipeline sans consommer le quota."""

    def analyze_tender(self, tender: TenderBase) -> AITenderEvaluation:
        time.sleep(0.45)
        return AITenderEvaluation(
            cleaned_title=f"[IA] {tender.objet[:70]}",
            executive_summary="Synthèse structurée générée selon le schéma Pydantic défini.",
            key_requirements=["Conformité au cahier des charges", "Références similaires"],
            risk_factors=["Délai de réponse à anticiper"],
            relevance_score=8,
            recommended_action="À étudier",
        )


async def run_validation(use_live_ai: bool = False) -> None:
    sample_path = ROOT_DIR / "data" / "sample_boamp_tenders.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {sample_path}")

    raw_items = json.loads(sample_path.read_text(encoding="utf-8"))
    raw_tenders = [
        RawTenderInput(
            id_annonce=item["id_annonce"],
            objet=item["objet"],
            description=item.get("objet", ""),
            date_parution=item.get("date_parution"),
            date_limite_reponse=item.get("date_limite_reponse"),
            acheteur=item.get("acheteur"),
            departement=item.get("departement"),
            type_marche=item.get("type_marche"),
        )
        for item in raw_items
    ]

    criteria = FilterCriteriaSchema(
        allowed_departments=["75", "91", "92", "93", "94", "69", "33", "31"],
        keywords=["informatique", "logiciel", "maintenance", "études", "fourniture", "travaux", "services"],
        min_keyword_mentions=1,
    )

    t0_clean = time.perf_counter()
    _ = [clean_single_tender(t, criteria) for t in raw_tenders]
    clean_duration_ms = (time.perf_counter() - t0_clean) * 1000

    analyzer = GeminiTenderAnalyzer() if use_live_ai else SimulatedGeminiAnalyzer()
    t0_batch = time.perf_counter()
    batch_response = await process_tenders_batch(
        tenders=raw_tenders,
        criteria=criteria,
        max_ai_calls=5,
        analyzer=analyzer,
    )
    batch_duration_ms = (time.perf_counter() - t0_batch) * 1000

    saved_calls = batch_response.total_received - batch_response.total_analyzed
    savings_pct = (saved_calls / batch_response.total_received) * 100

    report_lines = [
        "# Rapport de Validation du Pipeline sur 10 Échantillons BOAMP",
        "",
        f"- **Mode d'exécution IA :** {'Gemini Live API' if use_live_ai else 'Simulation calibrée (latence 450ms/requête)'}",
        f"- **Échantillons traités :** {batch_response.total_received} annonces anonymisées",
        "",
        "## 1. Métriques d'Exécution du Microservice",
        "",
        "| Indicateur | Résultat Mesuré |",
        "| :--- | :--- |",
        f"| **Temps de normalisation déterministe (10 annonces)** | `{clean_duration_ms:.2f} ms` |",
        f"| **Temps total du traitement Batch (filtrage + IA bornée)** | `{batch_duration_ms:.2f} ms` |",
        f"| **Annonces éligibles après pré-filtrage** | `{batch_response.total_eligible} / {batch_response.total_received}` |",
        f"| **Appels IA exécutés (plafond quota respecté)** | `{batch_response.total_analyzed}` (`{batch_response.total_deferred}` différés, `{batch_response.total_skipped}` filtrés) |",
        f"| **Réduction des appels LLM grâce au pré-filtrage** | **`{savings_pct:.0f}%`** |",
        "",
        "## 2. Détail des Résultats par Échantillon",
        "",
        "| ID Annonce | Département | Éligible | Statut Batch | Score IA | Action Recommandée |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for res in batch_response.results:
        dept = res.cleaned_data.departement or "N/A"
        eligible_str = "Oui" if res.cleaned_data.is_eligible else "Non"
        score_str = f"{res.ai_evaluation.relevance_score}/10" if res.ai_evaluation else "-"
        action_str = res.ai_evaluation.recommended_action if res.ai_evaluation else (res.cleaned_data.rejection_reason or "Différé quota")
        report_lines.append(
            f"| `{res.id_annonce}` | {dept} | {eligible_str} | `{res.status}` | {score_str} | {action_str} |"
        )

    report_path = ROOT_DIR / "VALIDATION_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Validation terminée en {batch_duration_ms:.2f} ms. Rapport généré : {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation du microservice FastAPI sur 10 échantillons BOAMP")
    parser.add_argument("--live-ai", action="store_true", help="Utiliser la vraie API Gemini (consomme du quota)")
    args = parser.parse_args()
    asyncio.run(run_validation(use_live_ai=args.live_ai))
