from pathlib import Path
from typing import Optional

from app.rag.chunker import RecursiveTextChunker
from app.rag.loader import DocumentLoader
from app.rag.vector_store import ChromaVectorStore
from app.schemas.document import IngestResponse, TenderContext


class TenderIngestionPipeline:
    """End-to-end idempotent ingestion pipeline: Load -> Chunk -> Enrich Metadata -> Index in ChromaDB."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.loader = DocumentLoader()
        self.chunker = RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = vector_store or ChromaVectorStore()

    def ingest_file(self, file_path: str | Path, tender_context: TenderContext) -> IngestResponse:
        path = Path(file_path)
        pages = self.loader.load(path)
        chunks = self.chunker.chunk_pages(pages)

        for chunk in chunks:
            chunk.metadata.tender_id = tender_context.tender_id
            if tender_context.section_title:
                chunk.metadata.section_title = tender_context.section_title
            if tender_context.deadline:
                chunk.metadata.deadline = tender_context.deadline
            if tender_context.budget is not None:
                chunk.metadata.budget = tender_context.budget

        added, skipped = self.vector_store.add_chunks_idempotent(chunks)

        return IngestResponse(
            status="success",
            tender_id=tender_context.tender_id,
            filename=path.name,
            pages_extracted=len(pages),
            total_chunks=len(chunks),
            chunks_added=added,
            chunks_skipped=skipped,
        )
