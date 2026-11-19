import hashlib
from typing import List, Optional, Sequence
from app.schemas.document import DocumentChunk, DocumentMetadata, DocumentPage

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", " ", "")


class RecursiveTextChunker:
    """
    Splits text recursively using natural boundaries (paragraphs -> lines -> sentences -> words)
    with configurable character overlap and full metadata preservation.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators)

    def _split_text_recursive(self, text: str, separators: Sequence[str]) -> List[str]:
        """Recursively breaks down text until every piece is <= chunk_size."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        # Select the first separator present in the text
        chosen_sep: Optional[str] = None
        remaining_seps: Sequence[str] = ()
        for idx, sep in enumerate(separators):
            if sep == "" or sep in text:
                chosen_sep = sep
                remaining_seps = separators[idx + 1 :]
                break

        # Fallback to character-level slicing if no separator remains
        if not chosen_sep:
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        raw_parts = text.split(chosen_sep)
        pieces: List[str] = []
        for i, part in enumerate(raw_parts):
            cleaned = part.strip()
            if not cleaned:
                continue
            # Preserve sentence-ending punctuation when splitting by sentence delimiters
            if chosen_sep in (". ", "? ", "! ") and i < len(raw_parts) - 1:
                cleaned += chosen_sep.strip()

            if len(cleaned) <= self.chunk_size:
                pieces.append(cleaned)
            else:
                pieces.extend(self._split_text_recursive(cleaned, remaining_seps))

        return pieces

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """Merges atomic pieces up to chunk_size while carrying over chunk_overlap characters."""
        if not pieces:
            return []

        chunks: List[str] = []
        current_chunk = ""

        for piece in pieces:
            if not current_chunk:
                current_chunk = piece
                continue

            candidate = f"{current_chunk} {piece}"
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                chunks.append(current_chunk.strip())
                if self.chunk_overlap > 0:
                    overlap_tail = current_chunk[-self.chunk_overlap :].strip()
                    # Avoid starting the overlapped prefix mid-word
                    if " " in overlap_tail and len(current_chunk) > self.chunk_overlap:
                        overlap_tail = overlap_tail.split(" ", 1)[1]
                    combined = f"{overlap_tail} {piece}".strip()
                    current_chunk = combined if len(combined) <= self.chunk_size else piece
                else:
                    current_chunk = piece

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """Splits a raw string into overlapping chunks."""
        atomic_pieces = self._split_text_recursive(text, self.separators)
        return self._merge_with_overlap(atomic_pieces)

    def chunk_pages(self, pages: List[DocumentPage]) -> List[DocumentChunk]:
        """Chunks a list of DocumentPage objects while propagating and enriching metadata."""
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in pages:
            text_chunks = self.chunk_text(page.content)
            for chunk_str in text_chunks:
                raw_key = f"{page.metadata.source_file}:{page.metadata.page_number}:{global_chunk_idx}:{chunk_str[:40]}"
                chunk_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

                enriched_meta = DocumentMetadata(
                    source_file=page.metadata.source_file,
                    file_type=page.metadata.file_type,
                    page_number=page.metadata.page_number,
                    section_title=page.metadata.section_title,
                    chunk_index=global_chunk_idx,
                )

                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        content=chunk_str,
                        char_count=len(chunk_str),
                        metadata=enriched_meta,
                    )
                )
                global_chunk_idx += 1

        return all_chunks
