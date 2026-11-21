from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, computed_field


class TenderContext(BaseModel):
    """Domain metadata attached to all chunks belonging to a specific public tender."""

    tender_id: str = Field(..., min_length=1, description="Unique BOAMP or internal tender identifier")
    section_title: Optional[str] = Field(default=None, description="Document section or heading (e.g., CCTP, RC, BPU)")
    deadline: Optional[str] = Field(default=None, description="Submission deadline in ISO format (YYYY-MM-DD)")
    budget: Optional[float] = Field(default=None, ge=0, description="Estimated tender budget in EUR")


class DocumentMetadata(BaseModel):
    """Metadata attached to every extracted page and text chunk."""

    source_file: str = Field(..., description="Original filename (e.g., cctp_lot1.pdf)")
    file_type: Literal["pdf", "docx"] = Field(..., description="Normalized file extension")
    page_number: int = Field(..., ge=1, description="1-based page or logical section number")
    section_title: Optional[str] = Field(default=None, description="Heading or section title extracted from document")
    chunk_index: Optional[int] = Field(default=None, ge=0, description="0-based chunk index within the document")
    tender_id: Optional[str] = Field(default=None, description="Associated public tender ID")
    deadline: Optional[str] = Field(default=None, description="Submission deadline (YYYY-MM-DD)")
    budget: Optional[float] = Field(default=None, ge=0, description="Estimated budget in EUR")

    def to_chroma_metadata(self) -> dict[str, Any]:
        """Convert metadata into a flat dict with only non-null primitive values accepted by ChromaDB."""
        raw = self.model_dump(exclude_none=True)
        return {k: v for k, v in raw.items() if isinstance(v, (str, int, float, bool))}


class DocumentPage(BaseModel):
    """Represents raw extracted text from a single page or logical section."""

    content: str = Field(..., min_length=1, description="Cleaned text content of the page")
    metadata: DocumentMetadata


class DocumentChunk(BaseModel):
    """Represents a chunked text segment ready for embedding and vector indexing."""

    chunk_id: Optional[str] = Field(default=None, description="Deterministic content ID for idempotency")
    content: str = Field(..., min_length=1, description="Text content of the chunk")
    metadata: DocumentMetadata

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.content)


class IngestResponse(BaseModel):
    """Standardized response returned by POST /documents/ingest."""

    status: Literal["success"] = "success"
    tender_id: str
    filename: str
    pages_extracted: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    chunks_added: int = Field(..., ge=0, description="Number of new chunks embedded and stored")
    chunks_skipped: int = Field(..., ge=0, description="Number of duplicate chunks skipped via SHA-256 hash")
