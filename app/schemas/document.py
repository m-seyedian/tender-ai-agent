from typing import Literal, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata attached to every extracted page and text chunk for traceability."""

    source_file: str = Field(..., description="Original file name (e.g., cctp.pdf)")
    file_type: Literal["pdf", "docx"] = Field(..., description="Document format")
    page_number: int = Field(default=1, ge=1, description="1-based page or section index")
    section_title: Optional[str] = Field(
        default=None, description="Detected heading or section title if available"
    )
    chunk_index: Optional[int] = Field(
        default=None, ge=0, description="Sequential index of the chunk within the document"
    )


class DocumentPage(BaseModel):
    """Represents a raw extracted unit (a PDF page or a DOCX logical section)."""

    content: str = Field(..., min_length=1, description="Cleaned text content of the page/section")
    metadata: DocumentMetadata


class DocumentChunk(BaseModel):
    """A single overlapping text chunk ready for embedding and vector search."""

    chunk_id: str = Field(..., description="Deterministic unique identifier for the chunk")
    content: str = Field(..., min_length=1, description="Chunk text content")
    char_count: int = Field(..., ge=1, description="Length of the chunk in characters")
    metadata: DocumentMetadata
