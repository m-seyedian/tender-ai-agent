import hashlib
from pathlib import Path
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from app.api.documents import get_tender_retriever
from app.main import app
from app.rag.retriever import HybridTenderRetriever, build_chroma_where
from app.rag.vector_store import ChromaVectorStore
from app.schemas.document import DocumentChunk, DocumentMetadata, SearchRequest


class DeterministicFakeEmbeddings(Embeddings):
    """Offline deterministic 16-d embeddings for fast, reproducible unit tests."""

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _seed_store(tmp_path: Path) -> ChromaVectorStore:
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_test"),
        collection_name="test_retriever_tenders",
        embedding_function=DeterministicFakeEmbeddings(),
    )
    chunks = [
        DocumentChunk(
            content="Article 14.2 : Les pénalités de retard sont fixées à 1/1000 du montant total par jour calendaire.",
            metadata=DocumentMetadata(
                source_file="ccap_26_100.pdf",
                file_type="pdf",
                page_number=12,
                section_title="CCAP",
                chunk_index=0,
                tender_id="BOAMP-26-100",
                deadline="2026-12-15",
                budget=250000.0,
            ),
        ),
        DocumentChunk(
            content="Le prestataire doit disposer de la certification ISO 27001 pour l'hébergement des données SIG.",
            metadata=DocumentMetadata(
                source_file="cctp_26_100.pdf",
                file_type="pdf",
                page_number=4,
                section_title="CCTP",
                chunk_index=1,
                tender_id="BOAMP-26-100",
                deadline="2026-12-15",
                budget=250000.0,
            ),
        ),
        DocumentChunk(
            content="Article 14.2 : Pénalités applicables uniquement aux travaux de voirie urbaine à Lyon.",
            metadata=DocumentMetadata(
                source_file="ccap_26_999.pdf",
                file_type="pdf",
                page_number=8,
                section_title="CCAP",
                chunk_index=0,
                tender_id="BOAMP-26-999",
                deadline="2026-12-20",
                budget=500000.0,
            ),
        ),
    ]
    store.add_chunks_idempotent(chunks)
    return store


def test_build_chroma_where_clauses():
    assert build_chroma_where() is None
    assert build_chroma_where(tender_id="T-1") == {"tender_id": "T-1"}
    multi = build_chroma_where(tender_id="T-1", section_title="CCTP", file_type="pdf")
    assert multi == {
        "$and": [
            {"tender_id": "T-1"},
            {"section_title": "CCTP"},
            {"file_type": "pdf"},
        ]
    }


def test_retriever_metadata_filtering_isolates_tenders(tmp_path: Path):
    store = _seed_store(tmp_path)
    retriever = HybridTenderRetriever(vector_store=store)

    response = retriever.search(
        SearchRequest(
            query="Article 14.2 pénalités de retard",
            top_k=5,
            tender_id="BOAMP-26-100",
            use_hybrid=True,
            use_rerank=True,
        )
    )

    assert response.total_results == 2
    assert all(item.metadata["tender_id"] == "BOAMP-26-100" for item in response.results)
    assert response.results[0].metadata["section_title"] == "CCAP"
    assert response.results[0].metadata["page_number"] == 12


def test_retriever_hybrid_bm25_and_rerank_promotes_exact_technical_terms(tmp_path: Path):
    store = _seed_store(tmp_path)
    retriever = HybridTenderRetriever(vector_store=store)

    response = retriever.search(
        SearchRequest(
            query="certification ISO 27001 données SIG",
            top_k=1,
            tender_id="BOAMP-26-100",
            use_hybrid=True,
            use_rerank=True,
        )
    )

    assert response.total_results == 1
    assert response.hybrid_used is True
    assert response.reranked is True
    assert "ISO 27001" in response.results[0].content
    assert response.results[0].score == 1.0


def test_post_search_endpoint_integration(tmp_path: Path):
    store = _seed_store(tmp_path)
    retriever = HybridTenderRetriever(vector_store=store)
    app.dependency_overrides[get_tender_retriever] = lambda: retriever

    try:
        client = TestClient(app)
        res = client.post(
            "/search",
            json={
                "query": "ISO 27001",
                "top_k": 2,
                "tender_id": "BOAMP-26-100",
                "section_title": "CCTP",
                "use_hybrid": True,
                "use_rerank": True,
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["total_results"] == 1
        assert body["results"][0]["metadata"]["section_title"] == "CCTP"
        assert "ISO 27001" in body["results"][0]["content"]

        # Invalid top_k (> 20) should trigger FastAPI/Pydantic 422 validation error
        bad_res = client.post("/search", json={"query": "test", "top_k": 50})
        assert bad_res.status_code == 422
    finally:
        app.dependency_overrides.clear()
