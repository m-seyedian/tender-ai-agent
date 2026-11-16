from fastapi.testclient import TestClient

from app.api.dependencies import get_gemini_analyzer
from app.main import app
from app.schemas.tender import AITenderEvaluation, TenderBase

client = TestClient(app)


class FakeGeminiAnalyzer:
    """Mock déterministe remplaçant Gemini pour ne pas consommer le quota gratuit en test."""

    def analyze_tender(self, tender: TenderBase) -> AITenderEvaluation:
        return AITenderEvaluation(
            cleaned_title=f"Analyse IA : {tender.objet}",
            executive_summary="Marché public pertinent pour notre périmètre technique.",
            key_requirements=["Certification ISO", "Références similaires"],
            risk_factors=["Délai de réponse court"],
            relevance_score=8,
            recommended_action="À étudier",
        )


def test_health_endpoint_returns_ok_and_headers():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers


def test_clean_tender_endpoint_normalizes_and_validates():
    payload = {
        "tender": {
            "id_annonce": "26-998877",
            "objet": "<p>Développement d'une plateforme <b>Cloud</b> IA</p>",
            "description": "Budget estimé : 120 000,50 € HT. Code CPV 72000000-5.",
            "date_parution": "15/11/2026",
            "date_limite_reponse": "2026-12-20T12:00:00",
            "departement": "75",
        },
        "criteria": {
            "allowed_departments": ["75", "91"],
            "target_cpv_prefixes": ["7200"],
            "keywords": ["cloud", "ia"],
        },
    }
    response = client.post("/tenders/clean", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_eligible"] is True
    assert data["cleaned_objet"] == "Développement d'une plateforme Cloud IA"
    assert data["normalized_date_parution"] == "2026-11-15"
    assert data["normalized_date_limite"] == "2026-12-20"
    assert data["estimated_amount_eur"] == 120000.5
    assert "72000000" in data["extracted_cpv_codes"]


def test_batch_endpoint_respects_eligibility_and_quota_guard():
    app.dependency_overrides[get_gemini_analyzer] = lambda: FakeGeminiAnalyzer()
    try:
        payload = {
            "max_ai_calls": 1,
            "criteria": {
                "allowed_departments": ["75"],
                "target_cpv_prefixes": ["7200"],
                "keywords": ["logiciel"],
            },
            "tenders": [
                {
                    "id_annonce": "26-100001",
                    "objet": "Fourniture de logiciel métier CPV 72000000",
                    "departement": "75",
                },
                {
                    "id_annonce": "26-100002",
                    "objet": "Maintenance logiciel cloud CPV 72000000",
                    "departement": "75",
                },
                {
                    "id_annonce": "26-100003",
                    "objet": "Travaux de voirie",
                    "departement": "33",
                },
            ],
        }
        response = client.post("/tenders/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_received"] == 3
        assert data["total_eligible"] == 2
        assert data["total_analyzed"] == 1
        assert data["total_deferred"] == 1
        assert data["total_skipped"] == 1
        assert data["total_failed"] == 0
    finally:
        app.dependency_overrides.clear()
