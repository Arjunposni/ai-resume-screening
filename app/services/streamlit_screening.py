from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from app.services.anonymizer import anonymizer
from app.services.explainer import generate_screening_explanation
from app.services.feature_extractor import feature_extractor
from app.services.jd_parser import jd_parser
from app.services.matcher import candidate_matcher
from app.services.resume_parser import resume_parser
from app.services.screening_features import match_result_to_features


SHORTLIST_THRESHOLD = 75.0

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


def parse_job_description(text: str) -> dict[str, Any]:
    """Parse raw job-description text into a structured dictionary."""

    if not text or not text.strip():
        raise ValueError("Job description cannot be empty.")

    job = jd_parser.parse(text)

    return job.model_dump()


def process_resume(
    file_name: str,
    file_bytes: bytes,
) -> dict[str, Any]:
    """
    Process a resume without requiring the FastAPI upload endpoint.

    Pipeline:
        upload → parse → anonymize → feature extraction
    """

    if not file_name:
        raise ValueError("Resume filename is required.")

    extension = Path(file_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Only PDF and DOCX resumes are supported."
        )

    resume_id = str(uuid4())
    stored_filename = f"{resume_id}{extension}"
    file_path = UPLOAD_DIR / stored_filename

    try:
        file_path.write_bytes(file_bytes)

        extracted_text = resume_parser.parse(
            str(file_path)
        )

        if not extracted_text.strip():
            raise ValueError(
                "The resume was uploaded successfully, "
                "but no readable text was extracted."
            )

        anonymized_text, detected_entities = (
            anonymizer.anonymize(extracted_text)
        )

        candidate_profile = feature_extractor.extract(
            anonymized_text
        )

        return {
            "resume_id": resume_id,
            "filename": file_name,
            "file_type": extension,
            "text_length": len(extracted_text),
            "anonymized_text": anonymized_text,
            "detected_entities": [
                entity.label
                for entity in detected_entities
            ],
            "candidate_profile": candidate_profile,
            "candidate_text": anonymized_text,
        }

    except Exception:
        if file_path.exists():
            file_path.unlink()

        raise


def screen_candidate(
    candidate_data: dict[str, Any],
    job_data: dict[str, Any],
    candidate_text: str,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """
    Screen one candidate directly through the service layer.

    This replaces:
        Streamlit → HTTP → FastAPI → services

    with:
        Streamlit → services
    """

    # Import schemas here so Streamlit startup remains lightweight.
    from app.models.schemas import (
        CandidateProfile,
        JobDescription,
    )

    candidate = (
        candidate_data
        if isinstance(candidate_data, CandidateProfile)
        else CandidateProfile.model_validate(candidate_data)
    )

    job = (
        job_data
        if isinstance(job_data, JobDescription)
        else JobDescription.model_validate(job_data)
    )

    candidate_id = candidate_id or str(uuid4())
    run_id = str(uuid4())
    job_id = str(uuid4())

    result = candidate_matcher.match(
        candidate=candidate,
        job=job,
        candidate_text=candidate_text,
        job_text=job.raw_text or "",
    )

    feature_values = match_result_to_features(result)

    # Load the trained ML model through the existing service.
    ml_prediction = _get_ml_prediction(feature_values)

    recommendation = (
        "SHORTLIST"
        if result.final_score >= SHORTLIST_THRESHOLD
        else "REVIEW"
    )

    explanation = generate_screening_explanation(
        result=result,
        ml_prediction=ml_prediction,
        recommendation=recommendation,
        shortlist_threshold=SHORTLIST_THRESHOLD,
    )

    return {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "job_id": job_id,
        "candidate_filename": None,

        "final_score": float(result.final_score),

        "shortlist_threshold": SHORTLIST_THRESHOLD,

        "recommendation": recommendation,

        "ml_prediction": ml_prediction,

        "features": feature_values,

        "skill_match": {
            "required_matched": (
                result.skill_match.required_matched
            ),
            "required_missing": (
                result.skill_match.required_missing
            ),
            "preferred_matched": (
                result.skill_match.preferred_matched
            ),
            "required_score": (
                result.skill_match.required_match_score
            ),
            "required_match_score": (
                result.skill_match.required_match_score
            ),
            "preferred_score": (
                result.skill_match.preferred_match_score
            ),
            "preferred_match_score": (
                result.skill_match.preferred_match_score
            ),
        },

        "experience_match": {
            "candidate_years": (
                result.experience_match.candidate_years
            ),
            "required_years": (
                result.experience_match.required_years
            ),
            "meets_requirement": (
                result.experience_match.meets_requirement
            ),
            "score": (
                result.experience_match.score
            ),
        },

        "education_match": {
            "matched_requirements": (
                result.education_match
                .matched_requirements
            ),
            "missing_requirements": (
                result.education_match
                .missing_requirements
            ),
            "score": (
                result.education_match.score
            ),
        },

        "semantic_match": {
            "score": (
                result.semantic_match.score
            ),
        },

        "explanation": explanation,
    }


def _get_ml_prediction(
    feature_values: dict[str, float],
) -> int | None:
    try:
        import pandas as pd
        from app.services.fairness import fairness_model, FEATURE_COLUMNS

        dataframe = pd.DataFrame(
            [[feature_values[column] for column in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )

        prediction = fairness_model.predict(dataframe)
        return int(prediction[0])

    except Exception:
        return None
