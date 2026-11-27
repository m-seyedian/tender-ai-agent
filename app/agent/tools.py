from typing import Any

SEARCH_TENDERS_TOOL: dict[str, Any] = {
    "name": "search_tenders",
    "description": (
        "Search French public procurement tenders (BOAMP) using keywords, CPV codes, and/or "
        "French department codes. Use this tool FIRST when the user asks to find, discover, or list "
        "tenders matching a topic, sector, or region without providing a specific tender_id. "
        "Returns a ranked list of matching tender summaries with their tender_id, title, buyer, "
        "department, CPV code, and submission deadline."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Search terms in French or English (e.g., ['cloud', 'cybersécurité', 'IA']).",
            },
            "cpv_codes": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[0-9]{8}$"},
                "description": "Optional list of 8-digit CPV procurement codes (e.g., ['72267100', '72000000']).",
            },
            "departments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional French department codes (e.g., ['75', '92', '69']).",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 25,
                "default": 5,
                "description": "Maximum number of matching tenders to return (default: 5, max: 25).",
            },
        },
        "required": ["keywords"],
    },
}

GET_TENDER_DETAILS_TOOL: dict[str, Any] = {
    "name": "get_tender_details",
    "description": (
        "Retrieve the complete structured metadata and administrative summary of a single BOAMP tender "
        "by its exact tender_id (e.g., '26-104520'). Use this AFTER search_tenders or whenever the user "
        "mentions a specific tender ID. Returns full buyer details, market type (TRAVAUX/SERVICES/FOURNITURES), "
        "estimated budget in EUR, CPV codes, publication date, deadline, and indexed document list (RC, CCTP, CCAP, BPU)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tender_id": {
                "type": "string",
                "description": "The unique BOAMP tender identifier (e.g., '26-104520').",
            }
        },
        "required": ["tender_id"],
    },
}

EXTRACT_REQUIREMENTS_TOOL: dict[str, Any] = {
    "name": "extract_requirements",
    "description": (
        "Query the indexed tender specifications (RC, CCTP, CCAP, BPU) via Hybrid RAG to extract "
        "mandatory technical, administrative, legal, or financial requirements for a specific tender_id. "
        "Use this tool BEFORE calling score_qualification whenever detailed compliance criteria "
        "(such as certifications like SecNumCloud/ISO 27001, SLAs, penalties, or turnover thresholds) "
        "must be verified from the tender documents. Returns categorized requirements with exact source file and page citations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tender_id": {
                "type": "string",
                "description": "The unique BOAMP tender identifier whose indexed documents should be analyzed.",
            },
            "category": {
                "type": "string",
                "enum": ["all", "technical", "administrative", "financial"],
                "default": "all",
                "description": "Filter extracted requirements by domain category (default: 'all').",
            },
        },
        "required": ["tender_id"],
    },
}

CHECK_DEADLINE_TOOL: dict[str, Any] = {
    "name": "check_deadline",
    "description": (
        "Evaluate the submission deadline urgency and feasibility for a given tender_id. "
        "Calculates exact days and hours remaining until the tender closes, and classifies urgency as "
        "'EXPIRED', 'CRITICAL' (< 3 days), 'TIGHT' (3 to 7 days), or 'HEALTHY' (> 7 days). "
        "Always call this tool before recommending a bid so expired tenders are immediately disqualified."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tender_id": {
                "type": "string",
                "description": "The unique BOAMP tender identifier to check.",
            },
            "reference_date": {
                "type": "string",
                "description": "Optional ISO-8601 timestamp (e.g., '2026-11-27T12:00:00Z'). Defaults to current UTC time.",
            },
        },
        "required": ["tender_id"],
    },
}

SCORE_QUALIFICATION_TOOL: dict[str, Any] = {
    "name": "score_qualification",
    "description": (
        "Compute a structured Go/No-Go qualification score (0 to 100) for a tender by comparing "
        "extracted mandatory requirements and deadline feasibility against the bidding company's profile. "
        "Use this as the FINAL analytical step after calling get_tender_details, check_deadline, and "
        "extract_requirements. Returns a numeric score, decision ('GO', 'REVIEW', or 'NO-GO'), "
        "matched strengths, and missing compliance gaps."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tender_id": {
                "type": "string",
                "description": "The unique BOAMP tender identifier being scored.",
            },
            "company_capabilities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of company certifications, technical skills, and past references (e.g., ['ISO 27001', 'Cloud souverain', 'Python/FastAPI']).",
            },
            "mandatory_requirements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Mandatory requirements extracted from the tender documents.",
            },
            "days_remaining": {
                "type": "number",
                "description": "Number of days remaining before the tender deadline (negative if expired).",
            },
        },
        "required": ["tender_id", "company_capabilities", "mandatory_requirements", "days_remaining"],
    },
}

AGENT_TOOLS_SCHEMA: list[dict[str, Any]] = [
    SEARCH_TENDERS_TOOL,
    GET_TENDER_DETAILS_TOOL,
    EXTRACT_REQUIREMENTS_TOOL,
    CHECK_DEADLINE_TOOL,
    SCORE_QUALIFICATION_TOOL,
]
