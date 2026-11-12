# BOAMP Tender AI Agent 🚀

An intelligent tender acquisition, filtering, and analysis pipeline evolved from an **n8n automation workflow** into a modular **Python / FastAPI** backend, enhanced with **Gemini LLM**, **RAG**, and **Autonomous AI Agents**.

## 🎯 Project Objective
Automate the fetching of French public tenders from the **BOAMP OpenData API**, extract full text from tender PDFs, filter opportunities by CPV codes, departments, and keywords, and leverage AI to score, summarize, and query tender documents.

## 🏗️ Architecture Mapping (n8n BOAMP Workflow ➔ Python)

| n8n Workflow Node | Legacy Responsibility | Python / FastAPI Equivalent | Target Module |
| :--- | :--- | :--- | :--- |
| `Sub_Trigger_BOAMP` | Workflow entry point | FastAPI REST Endpoints | `app/api/tenders.py` |
| `Read_Config` & `Offset` | Fetch target CPVs, departments, keywords & pagination state | Pydantic Settings & State Tracker | `app/core/config.py` |
| `Call_API_BOAMP` + Pagination | Paginated fetching from BOAMP v2.1 API | Async `httpx` BOAMP Client | `app/services/boamp_client.py` |
| `Filter_Market` & `Format_AO` | Filter market types (`TRAVAUX`, `SERVICES`, `FOURNITURES`) & build PDF URLs | Pydantic Data Schemas | `app/schemas/tender.py` |
| `Download_AO_PDF` & `Extract_PDF` | Download tender PDFs & parse raw text | `httpx` + PDF Parser (`pypdf`) | `app/services/pdf_extractor.py` |
| `Code_CPV` & `Counter_Keyword` | Regex 8-digit CPV extraction & keyword frequency counting | Hybrid Regex + **Gemini AI Analyzer** | `app/services/analyzer.py` |
| `Upsert_FluxBOAMP` | Deduplicated storage (`À étudier` vs `En attente`) | SQLite + ChromaDB (Vector Store for RAG) | `app/services/storage.py` |
| `Send_Notification` | Summary report dispatch | Automated Reporting Service | `app/services/notifier.py` |

## 📂 Project Structure
```text
tender-ai-agent/
├── app/
│   ├── api/        # FastAPI route handlers
│   ├── core/       # Configuration & environment settings
│   ├── schemas/    # Pydantic data validation models
│   └── services/   # Business logic (BOAMP API, PDF, Gemini, RAG)
├── data/           # Local sample tender PDFs & JSONs (gitignored)
├── tests/          # Unit and integration tests
├── .env.example    # Template for environment variables
└── requirements.txt