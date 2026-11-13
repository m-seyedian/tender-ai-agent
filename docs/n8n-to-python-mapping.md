# n8n (BOAMP Workflow) to Python / FastAPI Architecture Mapping

This document details the technical migration of the legacy **BOAMP Tender Monitoring n8n Workflow** (`Phase 1 & 2`) into a scalable, testable, and AI-enhanced **Python / FastAPI** backend.

---

## Phase 1: Data Acquisition & Pagination

| n8n Node | Legacy Logic | Python / FastAPI Implementation | Target Module |
| :--- | :--- | :--- | :--- |
| `Sub_Trigger_BOAMP` | Manual / sub-workflow execution trigger | `POST /api/v1/tenders/sync` REST endpoint | `app/api/tenders.py` |
| `Read_Config_BOAMP` | Reads period (`Période`), market types (`Travaux`, `Services`, `Fournitures`), target CPVs, departments, and keywords from Airtable | Type-safe configuration via Pydantic `BaseSettings` & request schemas | `app/core/config.py` & `app/schemas/tender.py` |
| `Fetch_Last_Offset` & `Update_Offset_BOAMP` | Tracks pagination offset in Airtable (`limit=100`) and resets on completion | In-memory / SQLite pagination state loop | `app/services/boamp_client.py` |
| `Call_API_BOAMP` | HTTP GET to `boamp-datadila.opendatasoft.com/api/explore/v2.1/catalog/datasets/boamp/records` | Async `httpx.AsyncClient` with retry & timeout handling | `app/services/boamp_client.py` |
| `Filter_Market_BOAMP` | Filters raw API records by allowed `type_marche` and normalizes `id_annonce` | Pydantic validation & filtering pipeline | `app/schemas/tender.py` |
| `Format_AO_BOAMP` | Constructs direct PDF download URL (`https://www.boamp.fr/telechargements/FILES/PDF/{year}/{month}/{idweb}.pdf`) | Computed property / validator in Tender schema | `app/schemas/tender.py` |

---

## Phase 2: PDF Extraction, Rule-Based Filtering & AI Upgrade

| n8n Node | Legacy Logic | Python / FastAPI Implementation | Target Module |
| :--- | :--- | :--- | :--- |
| `Aggregate_All_Pages` & `Loop_All_BOAMP` | Aggregates paginated items and iterates sequentially (`splitInBatches`) | Async batch processing (`asyncio.gather` with semaphore rate limiting) | `app/services/pipeline.py` |
| `Generate_MatchKey_BOAMP` | Normalizes title (removes accents/punctuation) to generate deduplication key (`matching_key`) | `unicodedata` normalization helper | `app/services/cleaner.py` |
| `Download_AO_PDF` & `Extract_PDF_BOAMP` | Downloads tender PDF and extracts raw text | `httpx` binary download + `pypdf` text extraction | `app/services/pdf_extractor.py` |
| `Code_CPV_BOAMP` | Extracts 8-digit CPV codes via dual Regex (`regexBefore` & `regexAfter`) and checks target CPVs & departments | Regex extractor + **Gemini Structured Output** fallback for unstructured PDFs | `app/services/analyzer.py` |
| `Attribution_AO_BOAMP` & `Check_Attributed` | Filters strictly for `Avis de marché` or `Avis rectificatif` | Rule filter + LLM document classification | `app/services/analyzer.py` |
| `Counter_Keyword_BOAMP` | Case-sensitive matching for short terms (`< 4 chars`, e.g., `EU`) and accent-insensitive matching for longer terms, excluding standalone `"Rehabilitation"` | Hybrid Regex keyword counter + **Gemini Semantic Relevance Scoring & RAG** | `app/services/analyzer.py` & `app/services/rag.py` |
| `Check_Match_CPV` / `Dept` & `Upsert_FluxBOAMP` | Routes tenders to `À étudier` (CPV + Dept match) or `En attente` and upserts into Airtable | SQLite relational storage + ChromaDB vector indexing | `app/services/storage.py` |
| `Preparation_Report` & `Send_Notification` | Aggregates `matched_count` and `empty_count` and dispatches HTML email summary | Structured JSON sync report + automated summary generator | `app/services/notifier.py` |