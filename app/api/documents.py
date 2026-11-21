import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.rag.ingestion import TenderIngestionPipeline
from app.schemas.document import IngestResponse, TenderContext

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


def get_ingestion_pipeline() -> TenderIngestionPipeline:
    """Dependency provider for TenderIngestionPipeline (easily overridable in tests)."""
    return TenderIngestionPipeline()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a PDF or DOCX tender document into ChromaDB",
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF or DOCX tender document"),
    tender_id: str = Form(..., min_length=1, description="Unique BOAMP or internal tender ID"),
    section_title: Optional[str] = Form(default=None, description="Optional section label (e.g., CCTP, RC, BPU)"),
    deadline: Optional[str] = Form(default=None, description="Submission deadline (YYYY-MM-DD)"),
    budget: Optional[float] = Form(default=None, ge=0, description="Estimated budget in EUR"),
    pipeline: TenderIngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is missing.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{suffix}'. Allowed formats: .pdf, .docx",
        )

    tender_context = TenderContext(
        tender_id=tender_id.strip(),
        section_title=section_title.strip() if section_title else None,
        deadline=deadline.strip() if deadline else None,
        budget=budget,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="tender_ingest_"))
    tmp_file_path = tmp_dir / Path(file.filename).name

    try:
        with tmp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return pipeline.ingest_file(file_path=tmp_file_path, tender_context=tender_context)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {exc}",
        ) from exc
    finally:
        await file.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
