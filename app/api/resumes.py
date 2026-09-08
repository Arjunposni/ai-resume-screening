from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.anonymizer import anonymizer
from app.services.feature_extractor import feature_extractor
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
    """Upload, parse, anonymize, and extract a candidate profile."""

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
        # -----------------------------
        # 1. Save uploaded resume
        # -----------------------------
        file_content = await file.read()
        file_path.write_bytes(file_content)

        # -----------------------------
        # 2. Extract resume text
        # -----------------------------
        extracted_text = resume_parser.parse(
            str(file_path)
        )

        if not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "The resume was uploaded successfully, "
                    "but no readable text was extracted."
                ),
            )

        # -----------------------------
        # 3. Remove/anonymize PII
        # -----------------------------
        anonymized_text, detected_entities = anonymizer.anonymize(
            extracted_text
        )

        # -----------------------------
        # 4. Extract candidate features
        # -----------------------------
        candidate_profile = feature_extractor.extract(
            anonymized_text
        )

        # -----------------------------
        # 5. Return processed result
        # -----------------------------
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
            "candidate_profile": candidate_profile.model_dump(),
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