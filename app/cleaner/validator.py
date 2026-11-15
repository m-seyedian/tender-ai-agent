from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.cleaner.extractor import count_keyword_matches, extract_cpv_codes
from app.cleaner.normalizer import clean_text


@dataclass
class CompanyFilterCriteria:
    """Critères de ciblage de l'entreprise pour filtrer les appels d'offres avant l'IA."""
    allowed_departments: List[str] = field(default_factory=list)
    target_cpv_prefixes: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    min_keyword_mentions: int = 1


@dataclass
class ValidationResult:
    """Résultat du filtrage déterministe avant appel à Gemini."""
    matching_key: str
    is_eligible: bool
    is_department_matched: bool
    is_cpv_matched: bool
    extracted_cpv_codes: List[str]
    matched_keywords: Dict[str, int]
    total_keyword_mentions: int
    rejection_reason: Optional[str] = None


def build_matching_key(idweb: str) -> str:
    """Génère une clé unique basée directement sur l'identifiant officiel BOAMP (idweb)."""
    clean_id = clean_text(idweb).lower()
    if not clean_id:
        raise ValueError("L'identifiant 'idweb' ne peut pas être vide.")
    return f"boamp_{clean_id}"


def evaluate_tender_eligibility(
    idweb: str,
    title: str,
    description: Optional[str],
    department: Optional[str],
    raw_cpv_codes: Optional[List[str]],
    criteria: CompanyFilterCriteria,
) -> ValidationResult:
    """Évalue si un appel d'offres correspond aux critères de l'entreprise (équivalent Filter_Market_BOAMP)."""
    full_text = clean_text(f"{title or ''} {description or ''}")
    matching_key = build_matching_key(idweb=idweb)

    # 1. Vérification du département (si une liste est définie)
    dept_clean = clean_text(department).zfill(2) if department else ""
    allowed_depts = [d.strip().zfill(2) for d in criteria.allowed_departments if d.strip()]
    is_dept_matched = True if not allowed_depts else (dept_clean in allowed_depts)

    # 2. Extraction et vérification des codes CPV (comparaison par préfixe, ex: '9051' ou code complet '90511000')
    extracted_cpvs = extract_cpv_codes(full_text)
    if raw_cpv_codes:
        for c in raw_cpv_codes:
            for extracted in extract_cpv_codes(str(c)):
                if extracted not in extracted_cpvs:
                    extracted_cpvs.append(extracted)

    target_prefixes = [p.strip() for p in criteria.target_cpv_prefixes if p.strip()]
    if not target_prefixes:
        is_cpv_matched = True
    else:
        is_cpv_matched = any(
            code.startswith(prefix)
            for code in extracted_cpvs
            for prefix in target_prefixes
        )

    # 3. Comptage des mots-clés
    keyword_counts = count_keyword_matches(full_text, criteria.keywords)
    total_mentions = sum(keyword_counts.values())
    has_enough_keywords = (
        True if not criteria.keywords else (total_mentions >= criteria.min_keyword_mentions)
    )

    # 4. Décision finale (Département valide ET (CPV valide OU mots-clés suffisants))
    if not is_dept_matched:
        return ValidationResult(
            matching_key=matching_key,
            is_eligible=False,
            is_department_matched=False,
            is_cpv_matched=is_cpv_matched,
            extracted_cpv_codes=extracted_cpvs,
            matched_keywords=keyword_counts,
            total_keyword_mentions=total_mentions,
            rejection_reason=f"Département '{dept_clean}' hors zone cible ({allowed_depts})",
        )

    if not is_cpv_matched and not has_enough_keywords:
        return ValidationResult(
            matching_key=matching_key,
            is_eligible=False,
            is_department_matched=True,
            is_cpv_matched=False,
            extracted_cpv_codes=extracted_cpvs,
            matched_keywords=keyword_counts,
            total_keyword_mentions=total_mentions,
            rejection_reason="Aucun code CPV cible ni mot-clé pertinent détecté",
        )

    return ValidationResult(
        matching_key=matching_key,
        is_eligible=True,
        is_department_matched=True,
        is_cpv_matched=is_cpv_matched,
        extracted_cpv_codes=extracted_cpvs,
        matched_keywords=keyword_counts,
        total_keyword_mentions=total_mentions,
        rejection_reason=None,
    )
