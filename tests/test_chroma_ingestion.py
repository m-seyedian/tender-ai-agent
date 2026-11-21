import hashlib
from pathlib import Path

from docx import Document as DocxDocument
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from app.api.documents import get_ingestion_pipeline
from app.main import app
from app.rag.ingestion import TenderIngestionPipeline
from app.rag.vector_store import ChromaVectorStore
from app.schemas.document import TenderContext


class DeterministicFakeEmbeddings(Embeddings):
    """Lightweight deterministic embeddings for offline unit/integration testing."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.embed_calls = 0

    def _vectorize(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += len(texts)
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


def _create_sample_docx(path: Path, title: str, body: str) -> Path:
    doc = DocxDocument()
    doc.add_heading(title, level=1)
    doc.add_paragraph(body)
    doc.save(str(path))
    return path


def test_chroma_persistent_and_idempotent_ingestion(tmp_path: Path) -> None:
    chroma_dir = tmp_path / "chroma_store"
    docx_file = _create_sample_docx(
        tmp_path / "cctp_26_999.docx",
        title="Article 1 - Objet du marché",
        body="Fourniture et maintenance de serveurs souverains pour la mairie de Palaiseau. " * 6,
    )

    fake_embeddings = DeterministicFakeEmbeddings()
    store = ChromaVectorStore(
        persist_directory=chroma_dir,
        collection_name="test_tenders",
        embedding_function=fake_embeddings,
    )
    pipeline = TenderIngestionPipeline(vector_store=store, chunk_size=250, chunk_overlap=40)
    ctx = TenderContext(
        tender_id="BOAMP-26-999",
        section_title="CCTP",
        deadline="2026-12-15",
        budget=150000.0,
    )

    # First ingestion: all chunks should be embedded and stored
    res1 = pipeline.ingest_file(docx_file, tender_context=ctx)
    assert res1.status == "success"
    assert res1.total_chunks >= 2
    assert res1.chunks_added == res1.total_chunks
    assert res1.chunks_skipped == 0
    first_embed_calls = fake_embeddings.embed_calls
    assert first_embed_calls == res1.total_chunks

    # Second ingestion of the exact same file & tender_id: 100% skipped, 0 new embedding calls
    res2 = pipeline.ingest_file(docx_file, tender_context=ctx)
    assert res2.chunks_added == 0
    assert res2.chunks_skipped == res1.total_chunks
    assert fake_embeddings.embed_calls == first_embed_calls

    # Re-open PersistentClient from disk to verify persistence across restarts
    reloaded_store = ChromaVectorStore(
        persist_directory=chroma_dir,
        collection_name="test_tenders",
        embedding_function=fake_embeddings,
    )
    assert reloaded_store.count() == res1.total_chunks

    results = reloaded_store.similarity_search(
        query="serveurs souverains Palaiseau",
        k=2,
        tender_id="BOAMP-26-999",
    )
    assert len(results) >= 1
    assert results[0].metadata["tender_id"] == "BOAMP-26-999"
    assert results[0].metadata["deadline"] == "2026-12-15"
    assert results[0].metadata["budget"] == 150000.0


def test_post_documents_ingest_endpoint(tmp_path: Path) -> None:
    chroma_dir = tmp_path / "api_chroma"
    fake_embeddings = DeterministicFakeEmbeddings()
    store = ChromaVectorStore(
        persist_directory=chroma_dir,
        collection_name="api_tenders",
        embedding_function=fake_embeddings,
    )
    test_pipeline = TenderIngestionPipeline(vector_store=store)

    app.dependency_overrides[get_ingestion_pipeline] = lambda: test_pipeline
    client = TestClient(app)

    try:
        docx_file = _create_sample_docx(
            tmp_path / "rc_lot2.docx",
            title="Règlement de Consultation",
            body="Critères d'attribution : 60% valeur technique, 40% prix des prestations.",
        )

        with docx_file.open("rb") as f:
            response = client.post(
                "/documents/ingest",
                files={"file": ("rc_lot2.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                data={
                    "tender_id": "26-77889",
                    "section_title": "RC",
                    "deadline": "2026-12-20",
                    "budget": "85000",
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "success"
        assert payload["tender_id"] == "26-77889"
        assert payload["chunks_added"] >= 1
        assert payload["chunks_skipped"] == 0

        # Unsupported file extension should return 400
        bad_resp = client.post(
            "/documents/ingest",
            files={"file": ("notes.txt", b"hello", "text/plain")},
            data={"tender_id": "26-77889"},
        )
        assert bad_resp.status_code == 400
    finally:
        app.dependency_overrides.clear()
