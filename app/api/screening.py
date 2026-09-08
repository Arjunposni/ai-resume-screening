from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.models.schemas import CandidateProfile, JobDescription

from app.services.anonymizer import anonymizer
from app.services.database import database
from app.services.explainer import generate_screening_explanation
from app.services.feature_extractor import feature_extractor
from app.services.fairness import fairness_model
from app.services.jd_parser import jd_parser
from app.services.matcher import (
    candidate_matcher,
    candidate_profile_to_text,
)
from app.services.resume_parser import resume_parser
from app.services.screening_features import match_result_to_features


router = APIRouter(
    prefix="/screening",
    tags=["Screening"],
)

SHORTLIST_THRESHOLD = 75.0

ALLOWED_EXTENSIONS = {".pdf", ".docx"}

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class ScreeningRequest(BaseModel):
    candidate: CandidateProfile
    job: JobDescription

    semantic_score: float | None = None
    candidate_text: str | None = None

    candidate_id: str | None = None
    job_id: str | None = None
    candidate_filename: str | None = None
    run_id: str | None = None


def _safe_ml_prediction(
    feature_values: dict[str, float],
) -> int | None:
    """
    Run the fairness-aware model safely.

    If the model is unavailable or prediction fails,
    screening still continues.
    """

    if not fairness_model.is_trained:
        return None

    try:
        feature_df = pd.DataFrame([feature_values])

        prediction = fairness_model.predict_fair(
            feature_df
        )

        if prediction is None or len(prediction) == 0:
            return None

        value = int(prediction[0])

        if value in {0, 1}:
            return value

        return None

    except Exception:
        return None


def _build_candidate_text(
    candidate: CandidateProfile,
) -> str:
    """
    Convert the structured candidate profile into text
    for semantic matching.
    """

    try:
        return candidate_profile_to_text(candidate)

    except Exception:
        return ""


def _model_to_dict(
    model: BaseModel,
) -> dict[str, Any]:
    """Support Pydantic v2 and older versions."""

    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def _persist_screening(
    *,
    run_id: str,
    candidate_id: str,
    job_id: str,
    candidate: CandidateProfile,
    job: JobDescription,
    candidate_filename: str | None,
    candidate_text: str,
    result_payload: dict[str, Any],
) -> None:
    """
    Persist job, candidate, screening result,
    and audit event.
    """

    database.save_job(
        job_id=job_id,
        title=job.title,
        raw_text=job.raw_text,
        parsed=_model_to_dict(job),
    )

    database.save_candidate(
        candidate_id=candidate_id,
        original_filename=(
            candidate_filename or candidate_id
        ),
        stored_filename=None,
        file_type=None,
        resume_path=None,
        anonymized_text=candidate_text,
        profile=_model_to_dict(candidate),
    )

    try:
        database.create_screening_run(
            run_id=run_id,
            job_id=job_id,
            candidate_count=1,
            status="completed",
        )

    except Exception:
        database.update_screening_run(
            run_id=run_id,
            status="completed",
            candidate_count=1,
        )

    database.save_screening_result(
        run_id=run_id,
        candidate_id=candidate_id,
        final_score=float(
            result_payload["final_score"]
        ),
        recommendation=str(
            result_payload["recommendation"]
        ),
        result=result_payload,
        ml_prediction=result_payload.get(
            "ml_prediction"
        ),
    )

    database.add_audit_event(
        event_id=str(uuid4()),
        event_type="SCREENING_COMPLETED",
        run_id=run_id,
        candidate_id=candidate_id,
        details={
            "job_id": job_id,
            "final_score": result_payload[
                "final_score"
            ],
            "recommendation": result_payload[
                "recommendation"
            ],
            "ml_prediction": result_payload.get(
                "ml_prediction"
            ),
        },
    )


def _screen_profile(
    *,
    candidate: CandidateProfile,
    job: JobDescription,
    candidate_text: str,
    run_id: str,
    candidate_id: str,
    job_id: str,
    candidate_filename: str | None,
) -> dict[str, Any]:
    """
    Core screening logic.

    Used by both single and batch screening.
    """

    result = candidate_matcher.match(
        candidate=candidate,
        job=job,
        candidate_text=candidate_text,
        job_text=job.raw_text or "",
    )

    feature_values = match_result_to_features(
        result
    )

    ml_prediction = _safe_ml_prediction(
        feature_values
    )

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

    response_payload: dict[str, Any] = {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "job_id": job_id,
        "candidate_filename": candidate_filename,

        "final_score": float(
            result.final_score
        ),

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

    try:
        _persist_screening(
            run_id=run_id,
            candidate_id=candidate_id,
            job_id=job_id,
            candidate=candidate,
            job=job,
            candidate_filename=candidate_filename,
            candidate_text=candidate_text,
            result_payload=response_payload,
        )

    except Exception as exc:

        response_payload["persistence_warning"] = (
            "Screening completed, but the result "
            "could not be persisted to the database."
        )

        try:
            database.add_audit_event(
                event_id=str(uuid4()),
                event_type=(
                    "SCREENING_PERSISTENCE_FAILED"
                ),
                run_id=run_id,
                candidate_id=candidate_id,
                details={
                    "error_type": type(exc).__name__
                },
            )

        except Exception:
            pass

    return response_payload


# ============================================================
# SINGLE CANDIDATE SCREENING
# ============================================================

@router.post("/match")
def screen_candidate(
    request: ScreeningRequest,
) -> dict[str, Any]:
    """
    Screen one structured candidate against one JD.
    """

    run_id = request.run_id or str(uuid4())

    candidate_id = (
        request.candidate_id
        or str(uuid4())
    )

    job_id = (
        request.job_id
        or str(uuid4())
    )

    candidate_text = (
        request.candidate_text
        or _build_candidate_text(
            request.candidate
        )
    )

    return _screen_profile(
        candidate=request.candidate,
        job=request.job,
        candidate_text=candidate_text,
        run_id=run_id,
        candidate_id=candidate_id,
        job_id=job_id,
        candidate_filename=(
            request.candidate_filename
        ),
    )


# ============================================================
# BATCH SCREENING
# ============================================================

@router.post("/batch")
async def screen_batch(
    job: str = Form(...),
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """
    Screen multiple resumes against one job description.

    One batch = one screening run.

    Each resume gets:
        - candidate ID
        - screening result
        - ranking
        - explanation
        - database persistence
    """

    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one resume is required.",
        )

    if len(files) > 500:
        raise HTTPException(
            status_code=400,
            detail=(
                "A maximum of 500 resumes can be "
                "processed in one batch."
            ),
        )

    # --------------------------------------------------------
    # 1. Parse Job Description
    # --------------------------------------------------------

    try:
        parsed_job = jd_parser.parse(job)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to parse the job description."
            ),
        ) from exc

    run_id = str(uuid4())
    job_id = str(uuid4())

    # --------------------------------------------------------
    # 2. Create one screening run
    # --------------------------------------------------------

    try:
        database.save_job(
            job_id=job_id,
            title=parsed_job.title,
            raw_text=parsed_job.raw_text,
            parsed=_model_to_dict(parsed_job),
        )

        database.create_screening_run(
            run_id=run_id,
            job_id=job_id,
            candidate_count=len(files),
            status="processing",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to initialize screening run."
            ),
        ) from exc

    results: list[dict[str, Any]] = []

    failures: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # 3. Process every resume
    # --------------------------------------------------------

    for index, uploaded_file in enumerate(files):

        filename = (
            uploaded_file.filename
            or f"candidate_{index + 1}"
        )

        suffix = Path(filename).suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:

            failures.append(
                {
                    "filename": filename,
                    "error": (
                        f"Unsupported file type: {suffix}"
                    ),
                }
            )

            continue

        candidate_id = str(uuid4())

        # Use UUID-based stored filename.
        # This avoids filename collisions.
        stored_filename = (
            f"{candidate_id}{suffix}"
        )

        stored_path = (
            UPLOAD_DIR / stored_filename
        )

        try:
            # ------------------------------------------------
            # Read uploaded file
            # ------------------------------------------------

            file_bytes = await uploaded_file.read()

            if not file_bytes:
                raise ValueError(
                    "Resume file is empty."
                )

            # ------------------------------------------------
            # Save file because ResumeParser.parse()
            # expects a file path.
            # ------------------------------------------------

            stored_path.write_bytes(
                file_bytes
            )

            # ------------------------------------------------
            # Extract resume text
            # ------------------------------------------------

            parsed_text = resume_parser.parse(
                str(stored_path)
            )

            if not parsed_text.strip():
                raise ValueError(
                    "No readable text was extracted "
                    "from the resume."
                )

            # ------------------------------------------------
            # PII anonymization
            #
            # anonymize() returns:
            #
            # (anonymized_text, detected_entities)
            # ------------------------------------------------

            (
                anonymized_text,
                detected_entities,
            ) = anonymizer.anonymize(
                parsed_text
            )

            # ------------------------------------------------
            # Structured candidate extraction
            # ------------------------------------------------

            candidate_profile = (
                feature_extractor.extract(
                    anonymized_text
                )
            )

            # ------------------------------------------------
            # Semantic matching text
            # ------------------------------------------------

            candidate_text = (
                _build_candidate_text(
                    candidate_profile
                )
            )

            # ------------------------------------------------
            # Screen candidate
            # ------------------------------------------------

            result = _screen_profile(
                candidate=candidate_profile,
                job=parsed_job,
                candidate_text=candidate_text,
                run_id=run_id,
                candidate_id=candidate_id,
                job_id=job_id,
                candidate_filename=filename,
            )

            # ------------------------------------------------
            # Additional batch metadata
            # ------------------------------------------------

            result["batch_index"] = index

            result["detected_entity_types"] = [
                getattr(
                    entity,
                    "label",
                    str(entity),
                )
                for entity in detected_entities
            ]

            result["stored_filename"] = (
                stored_filename
            )

            results.append(result)

        except Exception as exc:

            failures.append(
                {
                    "filename": filename,
                    "error": type(exc).__name__,
                }
            )

            # Continue processing the remaining resumes.
            continue

    # --------------------------------------------------------
    # 4. Mark screening run complete
    # --------------------------------------------------------

    try:
        database.update_screening_run(
            run_id=run_id,
            status="completed",
            candidate_count=len(results),
        )

        database.add_audit_event(
            event_id=str(uuid4()),
            event_type="BATCH_SCREENING_COMPLETED",
            run_id=run_id,
            details={
                "job_id": job_id,
                "submitted": len(files),
                "processed": len(results),
                "failed": len(failures),
            },
        )

    except Exception:
        # Screening results are already valid.
        pass

    # --------------------------------------------------------
    # 5. Rank candidates
    # --------------------------------------------------------

    results.sort(
        key=lambda item: float(
            item.get(
                "final_score",
                0.0,
            )
        ),
        reverse=True,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        result["rank"] = rank

    # --------------------------------------------------------
    # 6. Summary statistics
    # --------------------------------------------------------

    shortlisted = sum(
        result["recommendation"]
        == "SHORTLIST"
        for result in results
    )

    needs_review = sum(
        result["recommendation"]
        == "REVIEW"
        for result in results
    )

    average_score = (
        round(
            sum(
                float(
                    result["final_score"]
                )
                for result in results
            )
            / len(results),
            2,
        )
        if results
        else 0.0
    )

    # --------------------------------------------------------
    # 7. Return batch response
    # --------------------------------------------------------

    return {
        "run_id": run_id,
        "job_id": job_id,

        "submitted_candidates": len(files),

        "processed_candidates": len(
            results
        ),

        "failed_candidates": len(
            failures
        ),

        "shortlisted": shortlisted,

        "needs_review": needs_review,

        "average_score": average_score,

        "results": results,

        "failures": failures,
    }