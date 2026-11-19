import re
from pathlib import Path
from typing import List, Optional

from docx import Document as DocxDocument
from pypdf import PdfReader

from app.schemas.document import DocumentMetadata, DocumentPage

# Regex pour détecter un titre de section au début d'une page PDF (ex: "Article 1", "1. Objet", "CHAPITRE I")
HEADING_PATTERN = re.compile(
    r"^(?:ARTICLE|CHAPITRE|SECTION|TITRE|ANNEXE|\d+(?:\.\d+)*\.?)\s+[^\n]{3,90}$",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_extracted_text(text: str) -> str:
    """Normalizes whitespace and removes redundant blank lines while preserving paragraph breaks."""
    if not text:
        return ""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Supprime les lignes vides consécutives (garde au maximum un saut de paragraphe)
    cleaned_lines: List[str] = []
    previous_empty = False
    for line in lines:
        if not line:
            if not previous_empty:
                cleaned_lines.append("")
            previous_empty = True
        else:
            cleaned_lines.append(line)
            previous_empty = False
    return "\n".join(cleaned_lines).strip()


def _detect_section_title(text: str) -> Optional[str]:
    """Extracts the first probable heading from a block of text."""
    match = HEADING_PATTERN.search(text)
    if match:
        return match.group(0).strip()
    return None


def _load_pdf(file_path: Path) -> List[DocumentPage]:
    """Extracts text page-by-page from a PDF file with exact 1-based page numbers."""
    reader = PdfReader(str(file_path))
    pages: List[DocumentPage] = []
    last_known_section: Optional[str] = None

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = _clean_extracted_text(raw_text)
        if not cleaned_text:
            continue

        detected_title = _detect_section_title(cleaned_text)
        if detected_title:
            last_known_section = detected_title

        pages.append(
            DocumentPage(
                content=cleaned_text,
                metadata=DocumentMetadata(
                    source_file=file_path.name,
                    file_type="pdf",
                    page_number=page_idx,
                    section_title=last_known_section,
                ),
            )
        )

    return pages


def _load_docx(file_path: Path) -> List[DocumentPage]:
    """Extracts text grouped by headings from a DOCX file, including table rows."""
    doc = DocxDocument(str(file_path))
    sections: List[DocumentPage] = []

    current_title: Optional[str] = "Introduction"
    current_lines: List[str] = []
    section_index = 1

    def flush_section() -> None:
        nonlocal section_index, current_lines
        joined = _clean_extracted_text("\n".join(current_lines))
        if joined:
            sections.append(
                DocumentPage(
                    content=joined,
                    metadata=DocumentMetadata(
                        source_file=file_path.name,
                        file_type="docx",
                        page_number=section_index,
                        section_title=current_title,
                    ),
                )
            )
            section_index += 1
        current_lines = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style_name = (para.style.name or "").lower() if para.style else ""
        if style_name.startswith("heading") or style_name == "title":
            flush_section()
            current_title = text
        else:
            current_lines.append(text)

    # Extraction des tableaux éventuels (ex: bordereaux de prix, plannings)
    for table in doc.tables:
        table_rows: List[str] = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells if cell.text.strip()]
            if cells:
                table_rows.append(" | ".join(cells))
        if table_rows:
            current_lines.append("\n".join(table_rows))

    flush_section()
    return sections


def load_document(file_path: str | Path) -> List[DocumentPage]:
    """
    Unified loader for PDF and DOCX files.

    Raises:
        FileNotFoundError: If the target file does not exist.
        ValueError: If the file extension is not supported (.pdf or .docx).
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)

    raise ValueError(f"Unsupported file format '{suffix}'. Only .pdf and .docx are supported.")
