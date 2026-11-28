from datetime import datetime, timezone
from typing import Any

from app.cleaner.normalizer import clean_text, normalize_date
from app.cleaner.validator import CompanyFilterCriteria, evaluate_tender_eligibility
from app.rag.retriever import HybridTenderRetriever
from app.schemas.document import SearchRequest

DEFAULT_TENDER_CATALOG: dict[str, dict[str, Any]] = {
    "26-104520": {
        "tender_id": "26-104520",
        "title": "Maintenance et hébergement souverain SecNumCloud d'une plateforme SI",
        "description": "Prestations d'infogérance cloud souverain, sécurité ISO 27001, supervision 24/7 et PRA.",
        "buyer": "Ministère de l'Intérieur - SGAMI",
        "department": "75",
        "cpv_codes": ["72267100", "72000000"],
        "market_type": "SERVICES",
        "estimated_budget_eur": 450000,
        "publication_date": "2026-11-10T09:00:00Z",
        "deadline": "2026-12-18T12:00:00Z",
        "documents": ["RC_26-104520.pdf", "CCTP_26-104520.pdf", "CCAP_26-104520.pdf"],
    },
    "26-104889": {
        "tender_id": "26-104889",
        "title": "Fourniture de mobilier de bureau ergonomique",
        "description": "Livraison et installation de postes de travail et sièges ergonomiques.",
        "buyer": "Mairie de Lyon",
        "department": "69",
        "cpv_codes": ["39130000"],
        "market_type": "FOURNITURES",
        "estimated_budget_eur": 85000,
        "publication_date": "2026-11-01T08:00:00Z",
        "deadline": "2026-11-20T16:00:00Z",
        "documents": ["RC_26-104889.pdf", "BPU_26-104889.xlsx"],
    },
    "26-105112": {
        "tender_id": "26-105112",
        "title": "Développement d'un portail décisionnel IA et data gouvernance",
        "description": "Conception d'une architecture RAG, API Python FastAPI et tableaux de bord décisionnels.",
        "buyer": "Région Île-de-France",
        "department": "93",
        "cpv_codes": ["72212000", "72000000"],
        "market_type": "SERVICES",
        "estimated_budget_eur": 320000,
        "publication_date": "2026-11-15T10:00:00Z",
        "deadline": "2026-12-12T17:00:00Z",
        "documents": ["RC_26-105112.pdf", "CCTP_26-105112.docx"],
    },
}


class TenderToolExecutor:
    """Executes the 5 procurement agent tools connected to cleaner and Hybrid RAG services."""

    def __init__(
        self,
        retriever: HybridTenderRetriever | None = None,
        catalog: dict[str, dict[str, Any]] | None = None,
    ):
        self._retriever = retriever
        self.catalog = catalog if catalog is not None else DEFAULT_TENDER_CATALOG

    @property
    def retriever(self) -> HybridTenderRetriever:
        if self._retriever is None:
            self._retriever = HybridTenderRetriever()
        return self._retriever

    def search_tenders(
        self,
        keywords: list[str],
        cpv_codes: list[str] | None = None,
        departments: list[str] | None = None,
        max_results: int = 5,
    ) -> dict[str, Any]:
        criteria = CompanyFilterCriteria(
            allowed_departments=departments or [],
            target_cpv_prefixes=cpv_codes or [],
            keywords=keywords,
        )
        matches: list[dict[str, Any]] = []
        for tender in self.catalog.values():
            validation = evaluate_tender_eligibility(
                idweb=tender["tender_id"],
                title=tender["title"],
                description=tender.get("description"),
                department=tender.get("department"),
                raw_cpv_codes=tender.get("cpv_codes", []),
                criteria=criteria,
            )
            if validation.is_eligible:
                matches.append(
                    {
                        "tender_id": tender["tender_id"],
                        "title": clean_text(tender["title"]),
                        "buyer": tender["buyer"],
                        "department": tender.get("department"),
                        "cpv_codes": tender.get("cpv_codes", []),
                        "deadline": normalize_date(tender["deadline"]) or tender["deadline"],
                        "match_reason": getattr(validation, "reason", None),
                    }
                )
        return {"total_found": len(matches), "tenders": matches[:max_results]}

    def get_tender_details(self, tender_id: str) -> dict[str, Any]:
        tender = self.catalog.get(tender_id)
        if not tender:
            raise ValueError(f"Tender '{tender_id}' not found in catalog.")
        enriched = dict(tender)
        enriched["title"] = clean_text(tender["title"])
        enriched["deadline_normalized"] = normalize_date(tender["deadline"])
        return {"found": True, "tender": enriched}

    def check_deadline(self, tender_id: str, reference_date: str | None = None) -> dict[str, Any]:
        tender = self.catalog.get(tender_id)
        if not tender:
            raise ValueError(f"Tender '{tender_id}' not found in catalog.")

        norm_deadline = normalize_date(tender["deadline"])
        if not norm_deadline:
            raise ValueError(f"Invalid deadline format for tender '{tender_id}'")

        deadline_dt = datetime.fromisoformat(norm_deadline.replace("Z", "+00:00"))
        if deadline_dt.tzinfo is None:
            deadline_dt = deadline_dt.replace(tzinfo=timezone.utc)

        if reference_date:
            norm_ref = normalize_date(reference_date) or reference_date
            now_dt = datetime.fromisoformat(norm_ref.replace("Z", "+00:00"))
            if now_dt.tzinfo is None:
                now_dt = now_dt.replace(tzinfo=timezone.utc)
        else:
            now_dt = datetime.now(timezone.utc)

        days_remaining = round((deadline_dt - now_dt).total_seconds() / 86400, 2)
        if days_remaining < 0:
            urgency = "EXPIRED"
        elif days_remaining < 3:
            urgency = "CRITICAL"
        elif days_remaining <= 7:
            urgency = "TIGHT"
        else:
            urgency = "HEALTHY"

        return {
            "tender_id": tender_id,
            "deadline": deadline_dt.isoformat(),
            "days_remaining": days_remaining,
            "urgency": urgency,
            "is_expired": days_remaining < 0,
        }

    def extract_requirements(self, tender_id: str, category: str = "all") -> dict[str, Any]:
        if tender_id not in self.catalog:
            raise ValueError(f"Tender '{tender_id}' not found in catalog.")

        query_map = {
            "technical": "exigences techniques certifications SecNumCloud ISO architecture SLA",
            "administrative": "pièces administratives candidature RC habilitations délais",
            "financial": "critères financiers chiffre d'affaires pénalités BPU budget",
            "all": "exigences obligatoires certifications critères techniques administratifs financiers",
        }
        search_req = SearchRequest(
            query=query_map.get(category, query_map["all"]),
            tender_id=tender_id,
            top_k=5,
            use_hybrid=True,
            use_rerank=True,
        )
        response = self.retriever.search(search_req)
        requirements = [
            {
                "text": clean_text(item.content),
                "source_file": item.metadata.source_file,
                "page_number": item.metadata.page_number,
                "section_title": item.metadata.section_title,
                "score": round(item.score, 4),
            }
            for item in response.results
        ]
        return {
            "tender_id": tender_id,
            "category": category,
            "total_extracted": len(requirements),
            "requirements": requirements,
        }

    def score_qualification(
        self,
        tender_id: str,
        company_capabilities: list[str],
        mandatory_requirements: list[str],
        days_remaining: float,
    ) -> dict[str, Any]:
        if days_remaining < 0:
            return {
                "tender_id": tender_id,
                "score": 0,
                "decision": "NO-GO",
                "matched_strengths": [],
                "missing_gaps": ["Tender deadline has already expired."],
            }

        caps_lower = [c.lower() for c in company_capabilities]
        matched: list[str] = []
        gaps: list[str] = []

        for req in mandatory_requirements:
            req_tokens = [tok for tok in req.lower().split() if len(tok) > 2]
            if any(any(tok in cap for tok in req_tokens) or cap in req.lower() for cap in caps_lower):
                matched.append(req)
            else:
                gaps.append(req)

        coverage_score = 70.0 if not mandatory_requirements else (len(matched) / len(mandatory_requirements)) * 85.0
        time_bonus = 15.0 if days_remaining >= 7 else (8.0 if days_remaining >= 3 else 2.0)
        final_score = min(100, round(coverage_score + time_bonus))

        if final_score >= 75 and not gaps:
            decision = "GO"
        elif final_score >= 50:
            decision = "REVIEW"
        else:
            decision = "NO-GO"

        return {
            "tender_id": tender_id,
            "score": final_score,
            "decision": decision,
            "matched_strengths": matched,
            "missing_gaps": gaps,
        }

    def dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "search_tenders": self.search_tenders,
            "get_tender_details": self.get_tender_details,
            "check_deadline": self.check_deadline,
            "extract_requirements": self.extract_requirements,
            "score_qualification": self.score_qualification,
        }
        if tool_name not in handlers:
            raise KeyError(f"Unsupported tool requested by model: '{tool_name}'")
        return handlers[tool_name](**tool_input)
