from pathlib import Path
import pytest
from docx import Document as DocxDocument

from app.rag.chunker import RecursiveTextChunker
from app.rag.loader import load_document
from app.schemas.document import DocumentMetadata, DocumentPage


def test_load_docx_extracts_headings_tables_and_metadata(tmp_path: Path) -> None:
    docx_file = tmp_path / "cctp_test.docx"
    doc = DocxDocument()
    doc.add_heading("Article 1 - Objet du marché", level=1)
    doc.add_paragraph("Développement et maintenance d'une plateforme cloud souveraine.")
    doc.add_heading("Article 2 - Critères techniques", level=1)
    doc.add_paragraph("Disponibilité minimale exigée de 99.9% sur l'ensemble du périmètre.")

    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Lot"
    table.cell(0, 1).text = "Budget Max"
    table.cell(1, 0).text = "Lot 1 - Cloud"
    table.cell(1, 1).text = "150 000 EUR"

    doc.save(str(docx_file))

    pages = load_document(docx_file)
    assert len(pages) == 2
    assert pages[0].metadata.source_file == "cctp_test.docx"
    assert pages[0].metadata.file_type == "docx"
    assert pages[0].metadata.page_number == 1
    assert pages[0].metadata.section_title == "Article 1 - Objet du marché"
    assert "plateforme cloud" in pages[0].content

    assert pages[1].metadata.page_number == 2
    assert pages[1].metadata.section_title == "Article 2 - Critères techniques"
    assert "Lot 1 - Cloud | 150 000 EUR" in pages[1].content


def test_load_document_unsupported_extension_raises_value_error(tmp_path: Path) -> None:
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Contenu texte brut", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_document(txt_file)


def test_load_document_missing_file_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_document("non_existent_tender.pdf")


def test_recursive_chunker_preserves_overlap_and_metadata() -> None:
    long_text = (
        "Premier paragraphe décrivant le contexte administratif de l'appel d'offres public. "
        "Les candidats doivent fournir leurs références des trois dernières années.\n\n"
        "Deuxième paragraphe détaillant les exigences de cybersécurité et d'hébergement SecNumCloud. "
        "Toute sous-traitance doit être déclarée dès le dépôt de l'offre initiale.\n\n"
        "Troisième paragraphe relatif aux pénalités de retard et aux délais de livraison des livrables."
    )

    page = DocumentPage(
        content=long_text,
        metadata=DocumentMetadata(
            source_file="reglement_consultation.pdf",
            file_type="pdf",
            page_number=7,
            section_title="Article 5 - Exigences de sécurité",
        ),
    )

    chunker = RecursiveTextChunker(chunk_size=180, chunk_overlap=40)
    chunks = chunker.chunk_pages([page])

    assert len(chunks) >= 3
    for idx, chunk in enumerate(chunks):
        assert chunk.char_count <= 180
        assert chunk.metadata.page_number == 7
        assert chunk.metadata.section_title == "Article 5 - Exigences de sécurité"
        assert chunk.metadata.source_file == "reglement_consultation.pdf"
        assert chunk.metadata.chunk_index == idx
        assert len(chunk.chunk_id) == 16

    # Verify character overlap between consecutive chunks
    overlap_found = any(
        word in chunks[1].content
        for word in chunks[0].content.split()[-4:]
        if len(word) > 4
    )
    assert overlap_found
