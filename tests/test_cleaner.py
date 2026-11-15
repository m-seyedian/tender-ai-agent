import pytest

from app.cleaner.extractor import count_keyword_matches, extract_cpv_codes
from app.cleaner.normalizer import clean_text, extract_amount_eur, normalize_date
from app.cleaner.validator import (
    CompanyFilterCriteria,
    build_matching_key,
    evaluate_tender_eligibility,
)


# 1. Test nettoyage de texte HTML et espaces
def test_clean_text_removes_html_and_extra_spaces():
    raw = "  <p>Collecte & tri des   <b>déchets</b>\n\t ménagers</p>  "
    assert clean_text(raw) == "Collecte & tri des déchets ménagers"
    assert clean_text(None) == ""


# 2. Test normalisation des dates (ISO et format français)
def test_normalize_date_iso_and_french_formats():
    assert normalize_date("2026-12-14T16:30:00Z") == "2026-12-14"
    assert normalize_date("14/12/2026") == "2026-12-14"
    assert normalize_date("05-03-2026") == "2026-03-05"


# 3. Test gestion des dates invalides
def test_normalize_date_invalid_returns_none():
    assert normalize_date("32/13/2026") is None
    assert normalize_date("Date limite non précisée") is None
    assert normalize_date(None) is None


# 4. Test extraction des montants financiers en euros
def test_extract_amount_eur_formats():
    assert extract_amount_eur("Budget estimé : 150 000,50 € HT") == 150000.5
    assert extract_amount_eur("Montant maximum de 45.000 EUR sur 3 ans") == 45000.0
    assert extract_amount_eur("Aucun montant mentionné") is None


# 5. Test extraction des codes CPV uniques
def test_extract_cpv_codes_unique_and_ordered():
    text = "Codes CPV principaux : 90511000-2, 90512000 et rappel 90511000."
    assert extract_cpv_codes(text) == ["90511000", "90512000"]


# 6. Test comptage des mots-clés cibles
def test_count_keyword_matches_case_insensitive():
    text = "Gestion des DÉCHETS et valorisation des déchets industriels par réemploi."
    counts = count_keyword_matches(text, ["déchets", "réemploi", "informatique"])
    assert counts == {"déchets": 2, "réemploi": 1}


# 7. Test génération de la clé unique matching_key depuis idweb
def test_build_matching_key_from_idweb():
    assert build_matching_key("  26-145890 ") == "boamp_26-145890"
    with pytest.raises(ValueError):
        build_matching_key("   ")


# 8. Test validation positive via département + code CPV
def test_validator_accepts_matching_department_and_cpv():
    criteria = CompanyFilterCriteria(
        allowed_departments=["75", "91", "92"],
        target_cpv_prefixes=["9051"],
        keywords=["réemploi"],
    )
    res = evaluate_tender_eligibility(
        idweb="26-998877",
        title="Collecte et traitement des ordures (CPV 90511000-2)",
        description="Prestation sur la commune de Palaiseau.",
        department="91",
        raw_cpv_codes=None,
        criteria=criteria,
    )
    assert res.matching_key == "boamp_26-998877"
    assert res.is_eligible is True
    assert res.is_department_matched is True
    assert res.is_cpv_matched is True
    assert res.rejection_reason is None


# 9. Test validation positive via mots-clés même sans CPV explicite
def test_validator_accepts_via_keywords_even_without_cpv():
    criteria = CompanyFilterCriteria(
        allowed_departments=["91"],
        target_cpv_prefixes=["9051"],
        keywords=["réemploi", "valorisation"],
        min_keyword_mentions=2,
    )
    res = evaluate_tender_eligibility(
        idweb="26-112233",
        title="Opération de réemploi et valorisation d'objets",
        description="Territoire de l'Essonne.",
        department="91",
        raw_cpv_codes=[],
        criteria=criteria,
    )
    assert res.is_eligible is True
    assert res.is_cpv_matched is False
    assert res.total_keyword_mentions == 2


# 10. Test rejet d'un appel d'offres hors département ou hors sujet
def test_validator_rejects_wrong_department_or_irrelevant_content():
    criteria = CompanyFilterCriteria(
        allowed_departments=["75", "91"],
        target_cpv_prefixes=["9051"],
        keywords=["déchets"],
    )
    # Mauvais département (69 - Rhône)
    res_wrong_dept = evaluate_tender_eligibility(
        idweb="26-445566",
        title="Collecte des déchets (90511000)",
        description="Ville de Lyon",
        department="69",
        raw_cpv_codes=["90511000"],
        criteria=criteria,
    )
    assert res_wrong_dept.is_eligible is False
    assert res_wrong_dept.is_department_matched is False
    assert "69" in res_wrong_dept.rejection_reason

    # Bon département (91) mais hors sujet (informatique)
    res_irrelevant = evaluate_tender_eligibility(
        idweb="26-778899",
        title="Fourniture de serveurs informatiques (48820000)",
        description="Maintenance réseau",
        department="91",
        raw_cpv_codes=["48820000"],
        criteria=criteria,
    )
    assert res_irrelevant.is_eligible is False
    assert res_irrelevant.is_department_matched is True
    assert res_irrelevant.is_cpv_matched is False
