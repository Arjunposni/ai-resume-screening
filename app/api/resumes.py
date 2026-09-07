# Resume upload and processing endpoints will go here.
from pathlib import Path
from uuid import uuid4
from app.services.anonymizer import anonymizer
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.resume_parser import resume_parser


router = APIRouter(
    prefix="/resumes",
    tags=["Resumes"],
)


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


@router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    """Upload a resume and extract its text."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX resumes are supported.",
        )

    # Generate a unique filename so two resumes never collide.
    resume_id = str(uuid4())
    stored_filename = f"{resume_id}{extension}"
    file_path = UPLOAD_DIR / stored_filename

    try:
        file_content = await file.read()
        file_path.write_bytes(file_content)

        extracted_text = resume_parser.parse(str(file_path))
        anonymized_text, detected_entities = anonymizer.anonymize(
        extracted_text)

        if not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "The resume was uploaded successfully, "
                    "but no readable text was extracted."
                ),
            )

        return {
    "resume_id": resume_id,
    "filename": file.filename,
    "file_type": extension,
    "text_length": len(extracted_text),
    "anonymized_text": anonymized_text,
    "detected_entities": [
        entity.label
        for entity in detected_entities
    ],
}

    except HTTPException:
        raise

    except Exception as exc:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process resume: {exc}",
        ) from exc