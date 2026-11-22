import re
from typing import Any, Literal, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.rag.vector_store import ChromaVectorStore, compute_chunk_hash
from app.schemas.document import DocumentChunk, DocumentMetadata, SearchRequest, SearchResponse, SearchResultItem

_FRENCH_EN_STOPWORDS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "à", "au", "aux",
    "pour", "par", "sur", "dans", "avec", "ce", "ces", "est", "sont", "qui", "que",
    "the", "and", "of", "to", "in", "for", "on", "with", "is", "are", "a", "an",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer preserving French accents and technical codes."""
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_-]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _FRENCH_EN_STOPWORDS]


def build_chroma_where(
    tender_id: Optional[str] = None,
    section_title: Optional[str] = None,
    file_type: Optional[Literal["pdf", "docx"]] = None,
) -> Optional[dict[str, Any]]:
    """Build a valid ChromaDB where filter dictionary from optional metadata fields."""
    conditions: list[dict[str, Any]] = []
    if tender_id:
        conditions.append({"tender_id": tender_id})
    if section_title:
        conditions.append({"section_title": section_title})
    if file_type:
        conditions.append({"file_type": file_type})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _extract_doc_id(doc: Document) -> str:
    """Resolve a stable chunk identifier from document metadata or deterministic hash."""
    if doc.metadata.get("chunk_id"):
        return str(doc.metadata["chunk_id"])
    meta = DocumentMetadata(
        source_file=doc.metadata.get("source_file", "unknown.pdf"),
        file_type=doc.metadata.get("file_type", "pdf"),
        page_number=int(doc.metadata.get("page_number", 1)),
        section_title=doc.metadata.get("section_title"),
        chunk_index=doc.metadata.get("chunk_index"),
        tender_id=doc.metadata.get("tender_id"),
        deadline=doc.metadata.get("deadline"),
        budget=doc.metadata.get("budget"),
    )
    return compute_chunk_hash(DocumentChunk(content=doc.page_content, metadata=meta))


class HybridTenderRetriever:
    """Hybrid retriever combining ChromaDB semantic search, BM25 lexical search (RRF), and lightweight re-ranking."""

    def __init__(self, vector_store: Optional[ChromaVectorStore] = None, rrf_k: int = 60) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.rrf_k = rrf_k

    def _bm25_search(
        self,
        query: str,
        where_clause: Optional[dict[str, Any]],
        top_n: int,
    ) -> list[Document]:
        corpus_docs = self.vector_store.get_documents(where_clause=where_clause)
        if not corpus_docs:
            return []

        tokenized_corpus = [_tokenize(doc.page_content) for doc in corpus_docs]
        if not any(tokenized_corpus):
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query_tokens)

        scored_docs = [
            (score, doc)
            for score, doc in zip(scores, corpus_docs)
            if score > 0
        ]
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_n]]

    def _rerank(self, query: str, candidates: list[tuple[str, float, Document]]) -> list[tuple[str, float, Document]]:
        """Lightweight lexical coverage + exact phrase re-ranker."""
        query_tokens = set(_tokenize(query))
        normalized_query = " ".join(query.lower().split())
        if not query_tokens:
            return candidates

        reranked: list[tuple[str, float, Document]] = []
        for doc_id, base_score, doc in candidates:
            doc_tokens = set(_tokenize(doc.page_content))
            overlap_ratio = len(query_tokens & doc_tokens) / len(query_tokens)
            exact_phrase_bonus = 0.25 if normalized_query and normalized_query in doc.page_content.lower() else 0.0
            final_score = base_score * (1.0 + 0.5 * overlap_ratio + exact_phrase_bonus)
            reranked.append((doc_id, final_score, doc))

        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked

    def search(self, request: SearchRequest) -> SearchResponse:
        where_clause = build_chroma_where(
            tender_id=request.tender_id,
            section_title=request.section_title,
            file_type=request.file_type,
        )
        candidate_pool_size = max(request.top_k * 3, 10)

        # 1. Semantic vector search
        semantic_docs = self.vector_store._vectorstore.similarity_search(
            query=request.query,
            k=candidate_pool_size,
            filter=where_clause,
        )

        # 2. Optional BM25 lexical search + Reciprocal Rank Fusion (RRF)
        bm25_docs: list[Document] = []
        if request.use_hybrid:
            bm25_docs = self._bm25_search(
                query=request.query,
                where_clause=where_clause,
                top_n=candidate_pool_size,
            )

        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(semantic_docs, start=1):
            doc_id = _extract_doc_id(doc)
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, doc in enumerate(bm25_docs, start=1):
            doc_id = _extract_doc_id(doc)
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))

        candidates: list[tuple[str, float, Document]] = [
            (doc_id, score, doc_map[doc_id])
            for doc_id, score in rrf_scores.items()
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)

        # 3. Optional lightweight re-ranking
        if request.use_rerank and candidates:
            candidates = self._rerank(request.query, candidates)

        top_candidates = candidates[: request.top_k]

        # Normalize top score to [0, 1] range for clean API presentation
        max_score = top_candidates[0][1] if top_candidates else 1.0
        results = [
            SearchResultItem(
                chunk_id=doc_id,
                content=doc.page_content,
                score=round(score / max_score, 4) if max_score > 0 else 0.0,
                metadata=dict(doc.metadata),
            )
            for doc_id, score, doc in top_candidates
        ]

        return SearchResponse(
            query=request.query,
            total_results=len(results),
            hybrid_used=request.use_hybrid,
            reranked=request.use_rerank,
            results=results,
        )
