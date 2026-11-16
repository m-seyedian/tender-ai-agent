import asyncio
from typing import List

from app.cleaner.normalizer import clean_text, extract_amount_eur, normalize_date
from app.cleaner.validator import CompanyFilterCriteria, evaluate_tender_eligibility
from app.core.logger import get_logger
from app.schemas.tender import (
    BatchTenderItemResult,
    BatchTenderResponse,
    CleanedTenderOutput,
    FilterCriteriaSchema,
    RawTenderInput,
    TenderBase,
)
from app.services.gemini_client import GeminiTenderAnalyzer

logger = get_logger(__name__)

# Plafond de concurrence pour respecter les limites RPM du plan gratuit Gemini
MAX_CONCURRENT_GEMINI_REQUESTS = 2


def clean_single_tender(
    raw_tender: RawTenderInput,
    criteria_schema: FilterCriteriaSchema,
) -> CleanedTenderOutput:
    """Nettoie, normalise et évalue l'éligibilité déterministe d'un appel d'offres (sans IA)."""
    cleaned_objet = clean_text(raw_tender.objet)
    cleaned_description = clean_text(raw_tender.description)
    combined_text = f"{cleaned_objet} {cleaned_description}".strip()

    norm_date_parution = normalize_date(raw_tender.date_parution)
    norm_date_limite = normalize_date(raw_tender.date_limite_reponse)
    estimated_amount = extract_amount_eur(combined_text)

    criteria = CompanyFilterCriteria(
        allowed_departments=criteria_schema.allowed_departments,
        target_cpv_prefixes=criteria_schema.target_cpv_prefixes,
        keywords=criteria_schema.keywords,
        min_keyword_mentions=criteria_schema.min_keyword_mentions,
    )

    validation = evaluate_tender_eligibility(
        idweb=raw_tender.id_annonce,
        title=cleaned_objet,
        description=cleaned_description,
        department=raw_tender.departement,
        raw_cpv_codes=raw_tender.raw_cpv_codes,
        criteria=criteria,
    )

    return CleanedTenderOutput(
        matching_key=validation.matching_key,
        id_annonce=raw_tender.id_annonce,
        cleaned_objet=cleaned_objet,
        cleaned_description=cleaned_description,
        normalized_date_parution=norm_date_parution,
        normalized_date_limite=norm_date_limite,
        estimated_amount_eur=estimated_amount,
        acheteur=clean_text(raw_tender.acheteur) if raw_tender.acheteur else None,
        departement=clean_text(raw_tender.departement) if raw_tender.departement else None,
        type_marche=clean_text(raw_tender.type_marche) if raw_tender.type_marche else None,
        extracted_cpv_codes=validation.extracted_cpv_codes,
        matched_keywords=validation.matched_keywords,
        total_keyword_mentions=validation.total_keyword_mentions,
        is_eligible=validation.is_eligible,
        rejection_reason=validation.rejection_reason,
    )


async def _analyze_with_semaphore(
    semaphore: asyncio.Semaphore,
    analyzer: GeminiTenderAnalyzer,
    cleaned: CleanedTenderOutput,
) -> BatchTenderItemResult:
    """Exécute l'appel Gemini sous contrôle d'un Semaphore sans bloquer la boucle asyncio."""
    async with semaphore:
        tender_payload = TenderBase(
            id_annonce=cleaned.id_annonce,
            objet=cleaned.cleaned_objet,
            date_parution=cleaned.normalized_date_parution or "2026-01-01",
            date_limite_reponse=cleaned.normalized_date_limite,
            acheteur=cleaned.acheteur,
            departement=cleaned.departement,
            type_marche=cleaned.type_marche,
        )
        try:
            evaluation = await asyncio.to_thread(analyzer.analyze_tender, tender_payload)
            return BatchTenderItemResult(
                id_annonce=cleaned.id_annonce,
                matching_key=cleaned.matching_key,
                status="analyzed",
                cleaned_data=cleaned,
                ai_evaluation=evaluation,
            )
        except Exception as exc:
            logger.error(f"Échec de l'analyse IA pour {cleaned.id_annonce} : {exc}")
            return BatchTenderItemResult(
                id_annonce=cleaned.id_annonce,
                matching_key=cleaned.matching_key,
                status="error",
                cleaned_data=cleaned,
                error_detail=str(exc),
            )


async def process_tenders_batch(
    tenders: List[RawTenderInput],
    criteria: FilterCriteriaSchema,
    max_ai_calls: int,
    analyzer: GeminiTenderAnalyzer,
) -> BatchTenderResponse:
    """Pipeline complet : nettoyage déterministe -> filtrage -> analyse IA concurrente bornée."""
    cleaned_list = [clean_single_tender(item, criteria) for item in tenders]

    eligible_items = [c for c in cleaned_list if c.is_eligible]
    to_analyze = eligible_items[:max_ai_calls]
    deferred_keys = {c.matching_key for c in eligible_items[max_ai_calls:]}

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_GEMINI_REQUESTS)
    ai_tasks = [
        _analyze_with_semaphore(semaphore, analyzer, item)
        for item in to_analyze
    ]
    ai_results_list = await asyncio.gather(*ai_tasks)
    ai_results_by_key = {res.matching_key: res for res in ai_results_list}

    final_results: List[BatchTenderItemResult] = []
    for cleaned in cleaned_list:
        if not cleaned.is_eligible:
            final_results.append(
                BatchTenderItemResult(
                    id_annonce=cleaned.id_annonce,
                    matching_key=cleaned.matching_key,
                    status="skipped",
                    cleaned_data=cleaned,
                )
            )
        elif cleaned.matching_key in deferred_keys:
            final_results.append(
                BatchTenderItemResult(
                    id_annonce=cleaned.id_annonce,
                    matching_key=cleaned.matching_key,
                    status="quota_deferred",
                    cleaned_data=cleaned,
                )
            )
        else:
            final_results.append(ai_results_by_key[cleaned.matching_key])

    total_analyzed = sum(1 for r in final_results if r.status == "analyzed")
    total_skipped = sum(1 for r in final_results if r.status == "skipped")
    total_deferred = sum(1 for r in final_results if r.status == "quota_deferred")
    total_failed = sum(1 for r in final_results if r.status == "error")

    return BatchTenderResponse(
        total_received=len(tenders),
        total_eligible=len(eligible_items),
        total_analyzed=total_analyzed,
        total_skipped=total_skipped,
        total_deferred=total_deferred,
        total_failed=total_failed,
        results=final_results,
    )
