import hashlib
import os
from pathlib import Path
from typing import Any, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.schemas.document import DocumentChunk

DEFAULT_CHROMA_DIR = Path("data/chroma_db")
DEFAULT_COLLECTION_NAME = "boamp_tenders"


def compute_chunk_hash(chunk: DocumentChunk) -> str:
    """Compute a deterministic SHA-256 hash from tender_id, source_file, page_number, and content."""
    tender_id = chunk.metadata.tender_id or "no_tender"
    raw_key = f"{tender_id}::{chunk.metadata.source_file}::{chunk.metadata.page_number}::{chunk.content.strip()}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def get_default_embeddings() -> Embeddings:
    """Instantiate Google Gemini Embeddings using environment credentials."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in environment.")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


class ChromaVectorStore:
    """Persistent ChromaDB wrapper with SHA-256 idempotent ingestion and metadata filtering."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_function = embedding_function or get_default_embeddings()

        self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def get_existing_ids(self, candidate_ids: list[str]) -> set[str]:
        """Return the subset of candidate_ids already stored in the Chroma collection."""
        if not candidate_ids:
            return set()
        collection = self._client.get_or_create_collection(name=self.collection_name)
        existing = collection.get(ids=candidate_ids, include=[])
        return set(existing.get("ids", []))

    def add_chunks_idempotent(self, chunks: list[DocumentChunk]) -> tuple[int, int]:
        """
        Embed and store only chunks whose SHA-256 hash is not yet in ChromaDB.

        Returns:
            Tuple of (chunks_added, chunks_skipped)
        """
        if not chunks:
            return 0, 0

        # Deduplicate within the same incoming batch as well as against existing DB records
        unique_incoming: dict[str, DocumentChunk] = {}
        for chunk in chunks:
            chunk_hash = compute_chunk_hash(chunk)
            chunk.chunk_id = chunk_hash
            if chunk_hash not in unique_incoming:
                unique_incoming[chunk_hash] = chunk

        candidate_ids = list(unique_incoming.keys())
        existing_ids = self.get_existing_ids(candidate_ids)

        new_docs: list[Document] = []
        new_ids: list[str] = []

        for chunk_id, chunk in unique_incoming.items():
            if chunk_id in existing_ids:
                continue
            meta = chunk.metadata.to_chroma_metadata()
            meta["chunk_id"] = chunk_id
            new_docs.append(Document(page_content=chunk.content, metadata=meta))
            new_ids.append(chunk_id)

        if new_docs:
            self._vectorstore.add_documents(documents=new_docs, ids=new_ids)

        added_count = len(new_docs)
        skipped_count = len(chunks) - added_count
        return added_count, skipped_count

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        tender_id: Optional[str] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        """Search relevant chunks with optional tender_id or custom Chroma metadata filter."""
        where_clause: Optional[dict[str, Any]] = None
        if tender_id and metadata_filter:
            where_clause = {"$and": [{"tender_id": tender_id}, metadata_filter]}
        elif tender_id:
            where_clause = {"tender_id": tender_id}
        elif metadata_filter:
            where_clause = metadata_filter

        return self._vectorstore.similarity_search(query=query, k=k, filter=where_clause)

    def count(self) -> int:
        """Return total number of stored vectors in the collection."""
        collection = self._client.get_or_create_collection(name=self.collection_name)
        return collection.count()

    def get_documents(self, where_clause: Optional[dict[str, Any]] = None) -> list[Document]:
        """Fetch stored LangChain Documents matching an optional ChromaDB metadata filter."""
        collection = self._client.get_or_create_collection(name=self.collection_name)
        raw = collection.get(where=where_clause, include=["documents", "metadatas"])
        docs: list[Document] = []
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        for doc_id, text, meta in zip(ids, documents, metadatas):
            merged_meta = dict(meta or {})
            merged_meta.setdefault("chunk_id", doc_id)
            docs.append(Document(page_content=text, metadata=merged_meta))
        return docs
