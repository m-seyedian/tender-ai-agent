import hashlib
from pathlib import Path
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from app.api.documents import get_qa_engine
from app.main import app
from app.rag.qa import DEFAULT_NO_ANSWER_MESSAGE, TenderQAEngine
from app.rag.retriever import HybridTenderRetriever
from app.rag.vector_store import ChromaVectorStore
from app.schemas.document import AskRequest, DocumentChunk, DocumentMetadata, GroundedLLMAnswer


class DeterministicFakeEmbeddings(Embeddings):
    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeGroundedLLM:
    """Controllable mock LLM to test grounded citations and anti-hallucination guardrails."""

    def __init__(self, canned_response: GroundedLLMAnswer) -> None:
        self.model_name = "fake-gemini-2.5-flash"
        self.canned_response = canned_response
        self.call_count = 0
        self.last_prompt = ""

    def generate_grounded_answer(self, prompt: str) -> GroundedLLMAnswer:
        self.call_count += 1
        self.last_prompt = prompt
        return self.canned_response


def _seed_qa_store(tmp_path: Path) -> ChromaVectorStore:
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_qa_test"),
        collection_name="test_qa_tenders",
        embedding_function=DeterministicFakeEmbeddings(),
    )
    chunks = [
        DocumentChunk(
            content="Article 14.2 : Les pénalités de retard sont fixées à 1/1000 du montant total par jour calendaire.",
            metadata=DocumentMetadata(
                source_file="ccap_lot1.pdf",
                file_type="pdf",
                page_number=12,
                section_title="CCAP",
                chunk_index=0,
                tender_id="BOAMP-26-500",
            ),
        ),
        DocumentChunk(
            content="La durée maximale d'exécution du marché est de 18 mois à compter de l'ordre de service.",
            metadata=DocumentMetadata(
                source_file="rc_consultation.pdf",
                file_type="pdf",
                page_number=3,
                section_title="RC",
                chunk_index=1,
                tender_id="BOAMP-26-500",
            ),
        ),
    ]
    store.add_chunks_idempotent(chunks)
    return store


def test_qa_engine_returns_grounded_answer_with_verified_citations(tmp_path: Path):
    store = _seed_qa_store(tmp_path)
    retriever = HybridTenderRetriever(vector_store=store)
    fake_llm = FakeGroundedLLM(
        GroundedLLMAnswer(
            answered=True,
            answer="Les pénalités de retard s'élèvent à 1/1000 du montant total par jour calendaire [Source 1].",
            used_source_indices=[1],
        )
    )
    engine = TenderQAEngine(retriever=retriever, llm_client=fake_llm)

    res = engine.ask(
        AskRequest(
            question="Quel est le montant des pénalités de retard ?",
            tender_id="BOAMP-26-500",
        )
    )

    assert fake_llm.call_count == 1
    assert res.answered is True
    assert "1/1000" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].source_file == "ccap_lot1.pdf"
    assert res.citations[0].page_number == 12
    assert res.citations[0].section_title == "CCAP"


def test_qa_engine_short_circuits_when_no_context_found(tmp_path: Path):
    store = _seed_qa_store(tmp_path)
    retriever = HybridTenderRetriever(vector_store=store)
    fake_llm = FakeGroundedLLM(
        GroundedLLMAnswer(answered=True, answer="Should never be called", used_source_indices=[1])
    )
    engine = TenderQAEngine(retriever=retriever, llm_client=fake_llm)

    res = engine.ask(
        AskRequest(
            question="Quelle est la durée du marché ?",
            tender_id="NON-EXISTENT-TENDER",
        )
    )

    assert fake_llm.call_count == 0
    assert res.answered is False
    assert res.answer == DEFAULT_NO_ANSWER_MESSAGE
    assert res.citations == []
    assert res.sources_consulted == 0


def test_qa_engine_prevents_hallucination_on_unanswered_or_fake_citations(tmp_path: Path):
    store = _seed_qa_store(tmp_path)
    retriever = HybridTenderRetriever(vector_store=store)

    # Case A: LLM explicitly reports answer is not in documents
    unanswered_llm = FakeGroundedLLM(
        GroundedLLMAnswer(
            answered=False,
            answer="Les documents fournis ne mentionnent pas le taux de TVA applicable.",
            used_source_indices=[],
        )
    )
    engine_a = TenderQAEngine(retriever=retriever, llm_client=unanswered_llm)
    res_a = engine_a.ask(AskRequest(question="Quel est le taux de TVA ?", tender_id="BOAMP-26-500"))
    assert res_a.answered is False
    assert res_a.citations == []

    # Case B: LLM hallucinates a non-existent source index [99] -> downgraded to answered=False
    bad_citation_llm = FakeGroundedLLM(
        GroundedLLMAnswer(
            answered=True,
            answer="Réponse inventée [Source 99].",
            used_source_indices=[99],
        )
    )
    engine_b = TenderQAEngine(retriever=retriever, llm_client=bad_citation_llm)
    res_b = engine_b.ask(AskRequest(question="Durée d'exécution ?", tender_id="BOAMP-26-500"))
    assert res_b.answered is False
    assert res_b.citations == []


def test_post_ask_endpoint_integration(tmp_path: Path):
    store = _seed_qa_store(tmp_path)
    retriever = HybridTenderRetriever(vector_store=store)
    fake_llm = FakeGroundedLLM(
        GroundedLLMAnswer(
            answered=True,
            answer="La durée maximale d'exécution est de 18 mois [Source 1].",
            used_source_indices=[1],
        )
    )
    engine = TenderQAEngine(retriever=retriever, llm_client=fake_llm)
    app.dependency_overrides[get_qa_engine] = lambda: engine

    try:
        client = TestClient(app)
        res = client.post(
            "/ask",
            json={
                "question": "Quelle est la durée maximale d'exécution ?",
                "tender_id": "BOAMP-26-500",
                "section_title": "RC",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["answered"] is True
        assert body["citations"][0]["source_file"] == "rc_consultation.pdf"
        assert body["citations"][0]["page_number"] == 3
    finally:
        app.dependency_overrides.clear()
