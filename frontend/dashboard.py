"""
AI-Powered Resume Screening Dashboard
======================================
Production-ready Streamlit frontend for the resume-screening project.

Design goals
------------
- Native Streamlit processing for JD parsing and resume screening.
- Dynamic candidate count: works with 1, 10, 100+ uploaded resumes.
- Backend response normalization so UI is resilient to small API changes.
- Demo mode retained only as an explicit fallback/testing option.
- Recruiter filters, ranking, candidate evidence, notes and decisions.
- Fairness metrics loaded from the backend evaluation artifact when present.
- Environment-driven configuration for local development and Docker.
- No secrets or candidate data are hardcoded into the application.

Run
---
    uv run streamlit run frontend/dashboard.py

Environment
-----------
    BACKEND_URL=http://127.0.0.1:8000
    REQUEST_TIMEOUT=60
    SHORTLIST_THRESHOLD=75
    ENABLE_DEMO_MODE=true

The previous backend API contracts are retained only for the FastAPI application:
    POST /jobs/parse
    POST /resumes/upload
    POST /screening/match
    POST /screening/batch
    GET  /health

The screening request construction is isolated in one function so the
backend contract can evolve without rewriting the dashboard.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Project root / import path
# ---------------------------------------------------------------------------

# dashboard.py lives in frontend/, while the application package lives
# in the project root under app/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from app.services.streamlit_screening import (
    parse_job_description,
    process_resume,
    screen_candidate,
)


# SHAP/Fairness are loaded defensively so the dashboard can still start
# even when model artifacts are unavailable.
SHAPExplainer = None
FairnessModel = None
SHAP_IMPORT_ERROR = None

try:
    from app.services.shap_explainer import SHAPExplainer
    from app.services.fairness import FairnessModel
except Exception as exc:
    SHAP_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
).strip().rstrip("/")

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
SHORTLIST_THRESHOLD = float(os.getenv("SHORTLIST_THRESHOLD", "75"))
ENABLE_DEMO_MODE = os.getenv("ENABLE_DEMO_MODE", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

NAVIGATION_PAGES = [
    "Dashboard",
    "Candidates",
    "Fairness",
    "System",
]

SCORE_COMPONENTS = [
    ("Required Skills", "required_skills", 40),
    ("Experience", "experience", 25),
    ("Semantic Similarity", "semantic", 15),
    ("Education", "education", 10),
    ("Preferred Skills", "preferred_skills", 10),
]

EVALUATION_FILE = PROJECT_ROOT / "data" / "evaluation" / "fairness_results.json"
EVALUATION_DATASET = PROJECT_ROOT / "data" / "evaluation" / "candidates.csv"
FAIRNESS_FEATURES = [
    "required_skill_score",
    "experience_score",
    "education_score",
    "semantic_score",
    "preferred_skill_score",
]
FAIRNESS_TARGET = "selected"
FAIRNESS_PROTECTED = "protected_group"
FAIRNESS_TEST_SIZE = 0.40
FAIRNESS_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Backend client
# ---------------------------------------------------------------------------

class BackendClient:
    """Small HTTP client used by the Streamlit application."""

    def __init__(self, base_url: str, timeout: int) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @staticmethod
    def _error(response: requests.Response) -> str:
        """Return a readable error for normal and FastAPI validation responses."""
        try:
            payload = response.json()
        except (ValueError, TypeError):
            return response.text or f"HTTP {response.status_code}"

        detail = payload.get("detail") if isinstance(payload, dict) else None

        # FastAPI/Pydantic commonly returns a list of validation errors.
        if isinstance(detail, list):
            messages = []
            for item in detail:
                if isinstance(item, dict):
                    location = " → ".join(
                        str(part) for part in item.get("loc", [])
                    )
                    message = str(item.get("msg", "Validation error"))
                    messages.append(
                        f"{location}: {message}" if location else message
                    )
                else:
                    messages.append(str(item))
            return " | ".join(messages)

        if detail:
            return str(detail)

        return json.dumps(payload, ensure_ascii=False)

    def health(self) -> tuple[bool, str]:
        try:
            response = self.session.get(
                self._url("/health"),
                timeout=min(self.timeout, 10),
            )
            if response.ok:
                return True, "Backend Online"
            return False, self._error(response)
        except requests.RequestException as exc:
            return False, str(exc)

    def parse_job(self, raw_text: str) -> dict[str, Any]:
        try:
            response = self.session.post(
                self._url("/jobs/parse"),
                json={"text": raw_text},
                timeout=self.timeout,
            )
            if response.ok:
                return response.json()
            return {"error": self._error(response)}
        except requests.RequestException as exc:
            return {"error": f"Backend connection failed: {exc}"}
        except ValueError as exc:
            return {"error": f"Backend returned invalid JSON: {exc}"}

    def upload_resume(self, uploaded_file: Any) -> dict[str, Any]:
        try:
            content = uploaded_file.getvalue()
            files = {
                "file": (
                    uploaded_file.name,
                    content,
                    uploaded_file.type or "application/octet-stream",
                )
            }
            response = self.session.post(
                self._url("/resumes/upload"),
                files=files,
                timeout=max(self.timeout, 90),
            )
            if response.ok:
                return response.json()
            return {"error": self._error(response)}
        except requests.RequestException as exc:
            return {"error": f"Backend connection failed: {exc}"}
        except ValueError as exc:
            return {"error": f"Backend returned invalid JSON: {exc}"}

    def screen_candidate(
        self,
        candidate_profile: dict[str, Any],
        job_description: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the current backend candidate-vs-JD screening contract."""
        payload = {
            "candidate": candidate_profile,
            "job": job_description,
        }

        try:
            response = self.session.post(
                self._url("/screening/match"),
                json=payload,
                timeout=max(self.timeout, 120),
            )
            if response.ok:
                return response.json()
            return {"error": self._error(response)}
        except requests.RequestException as exc:
            return {"error": f"Backend connection failed: {exc}"}
        except ValueError as exc:
            return {"error": f"Backend returned invalid JSON: {exc}"}

    def screen_batch(
        self,
        job_description: str,
        uploaded_files: list[Any],
    ) -> dict[str, Any]:
        """Run batch screening through POST /screening/batch."""

        files = []

        for uploaded_file in uploaded_files:
            content = uploaded_file.getvalue()
            files.append(
                (
                    "files",
                    (
                        uploaded_file.name,
                        content,
                        uploaded_file.type or "application/octet-stream",
                    ),
                )
            )

        data = {"job": job_description}

        try:
            response = self.session.post(
                self._url("/screening/batch"),
                data=data,
                files=files,
                timeout=max(self.timeout, 180),
            )

            if response.ok:
                return response.json()

            return {"error": self._error(response)}

        except requests.RequestException as exc:
            return {"error": f"Backend connection failed: {exc}"}
        except ValueError as exc:
            return {"error": f"Backend returned invalid JSON: {exc}"}


backend = BackendClient(BACKEND_URL, REQUEST_TIMEOUT)


@st.cache_resource
def get_shap_explainer() -> Any:
    """Load the trained FairnessModel and create the SHAP explainer."""
    if SHAPExplainer is None or FairnessModel is None:
        detail = (
            f"{type(SHAP_IMPORT_ERROR).__name__}: {SHAP_IMPORT_ERROR}"
            if SHAP_IMPORT_ERROR is not None
            else "unknown import error"
        )
        raise RuntimeError(
            "SHAP/Fairness model could not be imported. "
            f"Import error: {detail}"
        )

    fairness_model = FairnessModel()

    models_dir = str(PROJECT_ROOT / "models")
    fairness_model.load_models(directory=models_dir)

    if getattr(fairness_model, "model", None) is None:
        raise RuntimeError(
            f"Trained Logistic Regression model was not loaded from {models_dir}."
        )

    # SHAPExplainer requires the held-out evaluation matrix (X_test) to
    # construct its background/reference data. The persisted model artifacts
    # intentionally contain the trained model/scaler, but FairnessModel does
    # not persist X_test. Recreate the SAME deterministic test split used
    # during training instead of retraining the model.
    evaluation_file = PROJECT_ROOT / "data" / "evaluation" / "candidates.csv"

    if not evaluation_file.exists():
        raise RuntimeError(
            f"SHAP evaluation dataset not found: {evaluation_file}"
        )

    evaluation_df = pd.read_csv(evaluation_file)

    required_columns = [
        "required_skill_score",
        "experience_score",
        "education_score",
        "semantic_score",
        "preferred_skill_score",
        "selected",
        "protected_group",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in evaluation_df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "SHAP evaluation dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    from sklearn.model_selection import train_test_split

    feature_columns = required_columns[:5]

    X = evaluation_df[feature_columns].copy()
    y = evaluation_df["selected"].copy()
    sensitive = evaluation_df["protected_group"].copy()

    _, X_test, _, y_test, _, sensitive_test = train_test_split(
        X,
        y,
        sensitive,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    # Keep the exact scaled representation expected by the trained model.
    fairness_model.X_test = pd.DataFrame(
        fairness_model.scaler.transform(X_test),
        columns=feature_columns,
    )
    fairness_model.y_test = y_test.reset_index(drop=True)
    fairness_model.sensitive_test = sensitive_test.reset_index(drop=True)

    return SHAPExplainer(fairness_model)


def explain_candidate_with_shap(candidate: dict[str, Any]) -> dict[str, Any]:
    """Generate a SHAP explanation from the five screening model features."""
    features = {
        "required_skill_score": normalize_score(candidate.get("required_skills")),
        "experience_score": normalize_score(candidate.get("experience")),
        "education_score": normalize_score(candidate.get("education")),
        "semantic_score": normalize_score(candidate.get("semantic")),
        "preferred_skill_score": normalize_score(candidate.get("preferred_skills")),
    }
    explainer = get_shap_explainer()
    return explainer.explain_candidate(features)


def render_shap_explanation(candidate: dict[str, Any]) -> None:
    """Render actual SHAP model explanations for the selected candidate."""
    st.subheader("🧠 Explainable AI — SHAP")
    st.caption(
        "SHAP explains the Logistic Regression model prediction using the same "
        "five screening features used during model evaluation. Protected attributes "
        "are not included."
    )

    try:
        explanation = explain_candidate_with_shap(candidate)
    except Exception as exc:
        st.warning(f"SHAP explanation is temporarily unavailable: {exc}")
        return

    probability = safe_float(explanation.get("selection_probability"))
    prediction_label = str(explanation.get("prediction_label", "unknown"))
    model_name = str(explanation.get("model", "logistic_regression"))

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", model_name.replace("_", " ").title())
    col2.metric("Selection Probability", f"{probability * 100:.1f}%")
    col3.metric("Model Prediction", prediction_label.replace("_", " ").title())

    contributions = explanation.get("feature_contributions", [])
    if not isinstance(contributions, list) or not contributions:
        st.info("No SHAP feature contributions were returned for this candidate.")
        return

    rows = []
    for item in contributions:
        if not isinstance(item, dict):
            continue
        shap_value = safe_float(item.get("shap_value"))
        impact = str(item.get("impact", "neutral")).upper()
        rows.append(
            {
                "Feature": str(item.get("feature", "Unknown")),
                "Input Value": round(safe_float(item.get("value")), 2),
                "SHAP Contribution": round(shap_value, 4),
                "Impact": impact,
            }
        )

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### What influenced the prediction?")
        for row in rows:
            direction = "increased" if row["SHAP Contribution"] > 0 else "decreased"
            st.write(
                f"**{row['Feature']}** ({row['Input Value']:.1f}) "
                f"{direction} the model's selection score "
                f"(SHAP: {row['SHAP Contribution']:+.4f})."
            )

    with st.expander("How to interpret SHAP values"):
        st.write(
            "Positive SHAP values push the Logistic Regression prediction toward "
            "selection; negative values push it toward non-selection. SHAP values "
            "shown here are model-internal contribution values (log-odds), not "
            "percentage changes in hiring probability."
        )


# ---------------------------------------------------------------------------
# Demo data — explicit UI testing only
# ---------------------------------------------------------------------------

DEMO_CANDIDATES: list[dict[str, Any]] = [
    {
        "candidate": "fake_resume_test_candidate.docx",
        "score": 90.63,
        "recommendation": "SHORTLIST",
        "required_skills": 100.0,
        "experience": 100.0,
        "education": 100.0,
        "semantic": 82.0,
        "preferred_skills": 33.33,
        "matched_skills": ["docker", "fastapi", "postgresql", "python"],
        "missing_skills": [],
        "preferred_matched": ["aws"],
        "candidate_years": 3.17,
        "required_years": 3.0,
        "education_match": "Bachelor's Degree in Computer Science",
        "strengths": [
            "All required skills are matched.",
            "The candidate satisfies the minimum experience requirement.",
            "The candidate satisfies the stated education requirement.",
            "Strong semantic alignment with the job description.",
            "AWS preferred skill matched.",
        ],
        "concerns": ["Some preferred skills were not matched."],
        "ml_prediction": 1,
    },
    {
        "candidate": "candidate_backend_02.docx",
        "score": 82.10,
        "recommendation": "SHORTLIST",
        "required_skills": 100.0,
        "experience": 100.0,
        "education": 100.0,
        "semantic": 74.0,
        "preferred_skills": 0.0,
        "matched_skills": ["docker", "fastapi", "postgresql", "python"],
        "missing_skills": [],
        "preferred_matched": [],
        "candidate_years": 4.2,
        "required_years": 3.0,
        "education_match": "Bachelor's Degree in Computer Science",
        "strengths": [
            "All required skills are matched.",
            "Experience exceeds the minimum requirement.",
            "Education requirement is satisfied.",
        ],
        "concerns": ["No preferred skills matched."],
        "ml_prediction": 1,
    },
    {
        "candidate": "candidate_backend_03.pdf",
        "score": 68.40,
        "recommendation": "REVIEW",
        "required_skills": 75.0,
        "experience": 66.67,
        "education": 100.0,
        "semantic": 71.0,
        "preferred_skills": 33.33,
        "matched_skills": ["docker", "python", "fastapi"],
        "missing_skills": ["postgresql"],
        "preferred_matched": ["aws"],
        "candidate_years": 2.0,
        "required_years": 3.0,
        "education_match": "Bachelor's Degree in Computer Science",
        "strengths": [
            "Good semantic alignment.",
            "Education requirement is satisfied.",
        ],
        "concerns": [
            "PostgreSQL is missing.",
            "Minimum experience requirement is not satisfied.",
        ],
        "ml_prediction": 0,
    },
]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

DEFAULT_STATE: dict[str, Any] = {
    "page": "Dashboard",
    "job_description": "",
    "parsed_jd": None,
    "job_parsed": False,
    "uploaded_files": [],
    "backend_resume_results": [],
    "screening_results": [],
    "screening_errors": [],
    "data_source": "None",
    "recruiter_decisions": {},
    "recruiter_notes": {},
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_score(value: Any) -> float:
    score = safe_float(value)
    if 0 <= score <= 1:
        score *= 100
    return max(0.0, min(score, 100.0))


def first_value(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def nested(mapping: dict[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def display_list(value: Any) -> str:
    items = [str(item) for item in list_value(value) if str(item).strip()]
    return " • ".join(items) if items else "—"


def candidate_key(candidate: dict[str, Any], index: int) -> str:
    return str(
        first_value(
            candidate,
            "candidate_id",
            "resume_id",
            "candidate",
            "filename",
            default=f"candidate-{index + 1}",
        )
    )


def recommendation_from_score(score: float) -> str:
    return "SHORTLIST" if score >= SHORTLIST_THRESHOLD else "REVIEW"


def load_demo_results() -> list[dict[str, Any]]:
    return copy.deepcopy(DEMO_CANDIDATES)


def calculate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "total": 0,
            "shortlisted": 0,
            "review": 0,
            "average": 0.0,
        }

    shortlisted = sum(
        str(item.get("recommendation", "REVIEW")).upper() == "SHORTLIST"
        for item in results
    )
    scores = [safe_float(item.get("score")) for item in results]

    return {
        "total": len(results),
        "shortlisted": shortlisted,
        "review": len(results) - shortlisted,
        "average": sum(scores) / len(scores),
    }


def filter_candidates(
    results: list[dict[str, Any]],
    search_query: str,
    decision_filter: str,
    minimum_score: float,
) -> list[dict[str, Any]]:
    query = search_query.strip().lower()
    filtered: list[dict[str, Any]] = []

    for candidate in results:
        name = str(
            first_value(
                candidate,
                "candidate",
                "filename",
                default="Unknown",
            )
        )
        decision = str(candidate.get("recommendation", "REVIEW")).upper()
        score = safe_float(candidate.get("score"))

        if query and query not in name.lower():
            continue
        if decision_filter != "All" and decision != decision_filter:
            continue
        if score < minimum_score:
            continue

        filtered.append(candidate)

    return sorted(
        filtered,
        key=lambda item: safe_float(item.get("score")),
        reverse=True,
    )


def build_ranking_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for rank, candidate in enumerate(results, start=1):
        rows.append(
            {
                "Rank": rank,
                "Candidate": first_value(
                    candidate,
                    "candidate",
                    "filename",
                    default="Unknown",
                ),
                "Score": round(safe_float(candidate.get("score")), 2),
                "Decision": str(
                    candidate.get("recommendation", "REVIEW")
                ).upper(),
                "Required Skills": round(
                    safe_float(candidate.get("required_skills")), 1
                ),
                "Experience": round(
                    safe_float(candidate.get("experience")), 1
                ),
                "Education": round(
                    safe_float(candidate.get("education")), 1
                ),
                "Semantic": round(
                    safe_float(candidate.get("semantic")), 1
                ),
                "Preferred Skills": round(
                    safe_float(candidate.get("preferred_skills")), 1
                ),
            }
        )

    return pd.DataFrame(rows)


def normalize_screening_result(
    raw: dict[str, Any],
    filename: str,
    resume_payload: dict[str, Any],
) -> dict[str, Any]:
    """Normalize MatchResult-like API output for the dashboard."""

    raw_match = raw.get("match_result", raw.get("result", raw))
    if not isinstance(raw_match, dict):
        raw_match = raw

    skill = first_value(
        raw_match,
        "skill_match",
        "skills",
        default={},
    ) or {}
    experience = first_value(
        raw_match,
        "experience_match",
        "experience",
        default={},
    ) or {}
    education = first_value(
        raw_match,
        "education_match",
        "education",
        default={},
    ) or {}
    semantic = first_value(
        raw_match,
        "semantic_match",
        "semantic",
        default={},
    ) or {}

    if not isinstance(skill, dict):
        skill = {}
    if not isinstance(experience, dict):
        experience = {}
    if not isinstance(education, dict):
        education = {}
    if not isinstance(semantic, dict):
        semantic = {}

    score = normalize_score(
        first_value(
            raw_match,
            "final_score",
            "score",
            default=0,
        )
    )

    recommendation = str(
        first_value(
            raw_match,
            "recommendation",
            "decision",
            default=recommendation_from_score(score),
        )
    ).upper()

    candidate_profile = resume_payload.get("candidate_profile", {})
    if not isinstance(candidate_profile, dict):
        candidate_profile = {}

    candidate_years = safe_float(
        first_value(
            experience,
            "candidate_years",
            default=safe_float(candidate_profile.get("total_experience_months")) / 12,
        )
    )

    required_years = first_value(
        experience,
        "required_years",
        default=nested(
            st.session_state.parsed_jd or {},
            "min_experience_years",
            default=0,
        ),
    )

    matched_skills = list_value(
        first_value(
            skill,
            "required_matched",
            "matched_skills",
            default=[],
        )
    )
    missing_skills = list_value(
        first_value(
            skill,
            "required_missing",
            "missing_skills",
            default=[],
        )
    )
    preferred_matched = list_value(
        first_value(
            skill,
            "preferred_matched",
            "preferred_skills_matched",
            default=[],
        )
    )

    education_matched = list_value(
        first_value(
            education,
            "matched_requirements",
            "matched",
            default=[],
        )
    )

    return {
        "candidate": filename,
        "candidate_id": first_value(
            raw_match,
            "candidate_id",
            "resume_id",
            default=resume_payload.get("resume_id"),
        ),
        "score": round(score, 2),
        "recommendation": recommendation,
        "required_skills": normalize_score(
            first_value(skill, "required_match_score", default=0)
        ),
        "experience": normalize_score(
            first_value(experience, "score", default=0)
        ),
        "education": normalize_score(
            first_value(education, "score", default=0)
        ),
        "semantic": normalize_score(
            first_value(semantic, "score", default=0)
        ),
        "preferred_skills": normalize_score(
            first_value(skill, "preferred_match_score", default=0)
        ),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "preferred_matched": preferred_matched,
        "candidate_years": candidate_years,
        "required_years": safe_float(required_years),
        "education_match": display_list(education_matched),
        "strengths": list_value(raw_match.get("strengths")),
        "concerns": list_value(raw_match.get("concerns")),
        "ml_prediction": raw_match.get("ml_prediction"),
        "raw_screening": raw_match,
        "candidate_profile": candidate_profile,
        "resume_payload": resume_payload,
    }


def process_uploaded_resumes(
    uploaded_files: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Process resumes directly inside Streamlit without FastAPI."""
    if not uploaded_files:
        return [], []

    if not st.session_state.parsed_jd:
        return [], [
            {
                "candidate": "Batch",
                "error": "No parsed job description is available.",
            }
        ]

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    progress = st.progress(0.0)
    status = st.empty()

    total = len(uploaded_files)

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        filename = str(uploaded_file.name)

        try:
            status.write(f"Processing {index}/{total}: {filename}")

            resume_payload = process_resume(
                filename,
                uploaded_file.getvalue(),
            )

            raw_result = screen_candidate(
                candidate_data=resume_payload["candidate_profile"].model_dump(),
                job_data=st.session_state.parsed_jd,
                candidate_text=resume_payload["candidate_text"],
                candidate_id=resume_payload["resume_id"],
            )

            normalized = normalize_screening_result(
                raw=raw_result,
                filename=filename,
                resume_payload=resume_payload,
            )

            normalized["run_id"] = raw_result.get("run_id")
            normalized["job_id"] = raw_result.get("job_id")
            normalized["candidate_id"] = raw_result.get("candidate_id")
            normalized["source_filename"] = filename

            results.append(normalized)

        except Exception as exc:
            errors.append(
                {
                    "candidate": filename,
                    "error": str(exc),
                }
            )

        progress.progress(index / total)

    status.empty()

    results.sort(
        key=lambda item: safe_float(item.get("score")),
        reverse=True,
    )

    return results, errors

def load_fairness_results() -> dict[str, Any] | None:
    if not EVALUATION_FILE.exists():
        return None
    try:
        return json.loads(EVALUATION_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def get_fairness_value(
    data: dict[str, Any] | None,
    model: str,
    section: str,
    metric: str,
    default: Any = None,
) -> Any:
    if not data:
        return default
    return (
        data.get(model, {})
        .get(section, {})
        .get(metric, default)
    )


def render_backend_json(title: str, payload: Any) -> None:
    with st.expander(title):
        st.json(payload)


# ---------------------------------------------------------------------------
# Fairness dashboard data helpers
# ---------------------------------------------------------------------------

def _metric(value: Any, digits: int = 4) -> float | None:
    """Convert metric values safely while preserving missing values."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits)


def _selection_rates(y_true: pd.Series, predictions: Any, sensitive: pd.Series) -> dict[str, float]:
    """Return selection rate for every protected group."""
    frame = pd.DataFrame({
        "y_true": list(y_true),
        "prediction": list(predictions),
        "group": list(sensitive),
    })
    rates: dict[str, float] = {}
    for group, group_df in frame.groupby("group", dropna=False):
        key = str(group)
        rates[key] = round(float(group_df["prediction"].mean()), 4)
    return rates


def _dp_difference(selection_rates: dict[str, float]) -> float:
    if not selection_rates:
        return 0.0
    values = list(selection_rates.values())
    return round(max(values) - min(values), 4)


def _classification_metrics(y_true: Any, predictions: Any, probabilities: Any = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "accuracy": _metric(accuracy_score(y_true, predictions)),
        "precision": _metric(precision_score(y_true, predictions, zero_division=0)),
        "recall": _metric(recall_score(y_true, predictions, zero_division=0)),
        "f1": _metric(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": None,
    }
    if probabilities is not None:
        try:
            result["roc_auc"] = _metric(roc_auc_score(y_true, probabilities))
        except ValueError:
            result["roc_auc"] = None
    return result


def _model_result(y_true: Any, predictions: Any, sensitive: Any, probabilities: Any = None) -> dict[str, Any]:
    rates = _selection_rates(pd.Series(y_true), predictions, pd.Series(sensitive))
    return {
        "performance": _classification_metrics(y_true, predictions, probabilities),
        "fairness": {
            "selection_rate_by_group": rates,
            "demographic_parity_difference": _dp_difference(rates),
        },
    }


def _recreate_fairness_test_split() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Recreate the deterministic held-out set used by model training."""
    if not EVALUATION_DATASET.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {EVALUATION_DATASET}")

    dataframe = pd.read_csv(EVALUATION_DATASET)
    required = FAIRNESS_FEATURES + [FAIRNESS_TARGET, FAIRNESS_PROTECTED]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError("Evaluation dataset is missing: " + ", ".join(missing))

    X = dataframe[FAIRNESS_FEATURES].copy()
    y = dataframe[FAIRNESS_TARGET].copy()
    sensitive = dataframe[FAIRNESS_PROTECTED].copy()

    _, X_test, _, y_test, _, sensitive_test = train_test_split(
        X,
        y,
        sensitive,
        test_size=FAIRNESS_TEST_SIZE,
        random_state=FAIRNESS_RANDOM_STATE,
        stratify=y,
    )
    return X_test, y_test.reset_index(drop=True), sensitive_test.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_fairness_dashboard_data() -> dict[str, Any] | None:
    """Load saved fairness results, falling back to live artifact evaluation."""
    saved: dict[str, Any] = {}
    if EVALUATION_FILE.exists():
        try:
            payload = json.loads(EVALUATION_FILE.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                saved = payload
        except (OSError, json.JSONDecodeError):
            saved = {}

    # Prefer the complete saved artifact when it contains all three models.
    if all(key in saved for key in ("logistic_regression", "knn", "fairness_aware")):
        return saved

    # Compute the dashboard values from persisted models without retraining.
    if FairnessModel is None:
        return saved or None

    try:
        X_test_raw, y_test, sensitive_test = _recreate_fairness_test_split()
        model = FairnessModel()
        model.load_models(directory=str(PROJECT_ROOT / "models"))

        X_test_scaled = pd.DataFrame(
            model.scaler.transform(X_test_raw),
            columns=FAIRNESS_FEATURES,
        )

        results: dict[str, Any] = {}

        # Baseline Logistic Regression.
        baseline_pred = model.model.predict(X_test_scaled)
        baseline_prob = None
        if hasattr(model.model, "predict_proba"):
            baseline_prob = model.model.predict_proba(X_test_scaled)[:, 1]
        results["logistic_regression"] = _model_result(
            y_test, baseline_pred, sensitive_test, baseline_prob
        )

        # KNN is present in the latest FairnessModel implementation. Keep a
        # defensive fallback so older persisted model artifacts remain usable.
        knn = getattr(model, "knn_model", None)
        if knn is not None and hasattr(knn, "predict"):
            knn_pred = knn.predict(X_test_scaled)
            knn_prob = None
            if hasattr(knn, "predict_proba"):
                knn_prob = knn.predict_proba(X_test_scaled)[:, 1]
            knn_result = _model_result(y_test, knn_pred, sensitive_test, knn_prob)
            knn_result["configuration"] = {
                "n_neighbors": getattr(knn, "n_neighbors", None),
                "weights": getattr(knn, "weights", None),
            }
            results["knn"] = knn_result
        elif "knn" in saved:
            results["knn"] = saved["knn"]

        # Fairness-aware ExponentiatedGradient model.
        fair_pred = model.fair_model.predict(X_test_scaled)
        results["fairness_aware"] = _model_result(
            y_test, fair_pred, sensitive_test
        )

        return results
    except Exception:
        return saved or None


def _model_display_name(name: str) -> str:
    return {
        "logistic_regression": "Logistic Regression",
        "knn": "KNN",
        "fairness_aware": "Fairness-Aware",
        "baseline": "Baseline",
    }.get(name, name.replace("_", " ").title())


def _get_model_section(data: dict[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        value = data.get(name)
        if isinstance(value, dict):
            return value
    return {}


def _group_rate_frame(data: dict[str, Any], model_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: set[str] = set()
    model_rates: dict[str, dict[str, float]] = {}
    for model_name in model_names:
        section = _get_model_section(data, model_name)
        fairness = section.get("fairness", {}) if isinstance(section, dict) else {}
        rates = fairness.get("selection_rate_by_group")
        if rates is None:
            rates = fairness.get("by_group", {}).get("selection_rate", {}) if isinstance(fairness.get("by_group"), dict) else {}
        if isinstance(rates, dict):
            normalized = {str(k): safe_float(v) for k, v in rates.items()}
            model_rates[model_name] = normalized
            groups.update(normalized.keys())

    for group in sorted(groups):
        row = {"Protected Group": group}
        for model_name in model_names:
            row[_model_display_name(model_name)] = model_rates.get(model_name, {}).get(group)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page configuration / CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 38px; font-weight: 750; margin-bottom: 4px; }
    .subtitle { font-size: 17px; opacity: .70; margin-bottom: 25px; }
    .section-title { font-size: 25px; font-weight: 650; margin-top: 20px; margin-bottom: 15px; }
    .small-muted { font-size: 13px; opacity: .65; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📄 AI Resume Screening")
    st.caption("Recruiter Decision Support System")
    st.divider()

    selected_page = st.radio(
        "Navigation",
        NAVIGATION_PAGES,
        index=NAVIGATION_PAGES.index(st.session_state.page),
    )
    st.session_state.page = selected_page

    st.divider()
    st.markdown("### System Status")
    st.success("Streamlit Native")
    st.caption("All screening services run inside the Streamlit application.")

    st.divider()
    st.markdown("### Data Source")
    source = st.session_state.data_source
    if source == "Streamlit":
        st.success("Streamlit Native")
    elif source == "Demo":
        st.info("Frontend Demo")
    else:
        st.caption("No screening data loaded")

    st.divider()
    st.markdown("### Human-in-the-Loop")
    st.caption(
        "AI recommendations support recruiter decisions. "
        "They do not replace human judgment."
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

if selected_page == "Dashboard":
    st.markdown('<div class="main-title">AI-Powered Resume Screening</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Fairness-aware candidate screening and ranking</div>',
        unsafe_allow_html=True,
    )

    # Job description
    st.markdown('<div class="section-title">1️⃣ Job Description</div>', unsafe_allow_html=True)

    job_text = st.text_area(
        "Paste the job description",
        value=st.session_state.job_description,
        height=230,
        placeholder=(
            "Backend Engineer\n\n"
            "REQUIRED SKILLS\nPython, FastAPI, PostgreSQL, Docker\n\n"
            "PREFERRED SKILLS\nAWS, Kubernetes, Machine Learning\n\n"
            "EXPERIENCE\n3+ years\n\n"
            "EDUCATION\nBachelor's Degree in Computer Science"
        ),
        key="job_text_area",
    )

    st.session_state.job_description = job_text

    jd_col1, jd_col2 = st.columns([1, 5])

    with jd_col1:
        if st.button(
            "🔍 Parse JD",
            type="primary",
            use_container_width=True,
            disabled=not bool(job_text.strip()),
        ):
            with st.spinner("Parsing job description..."):
                try:
                    parsed = parse_job_description(job_text)
                    st.session_state.job_parsed = True
                    st.session_state.parsed_jd = parsed
                    st.success("? Job description parsed successfully.")
                except Exception as exc:
                    st.session_state.job_parsed = False
                    st.session_state.parsed_jd = None
                    st.error(f"JD parsing failed: {exc}")

    with jd_col2:
        if st.session_state.job_parsed:
            st.success("✓ Parsed requirements are ready for screening.")

    if st.session_state.job_parsed and st.session_state.parsed_jd:
        jd = st.session_state.parsed_jd
        required = list_value(jd.get("required_skills"))
        preferred = list_value(jd.get("preferred_skills"))
        experience = jd.get("min_experience_years")
        education = list_value(jd.get("education_requirements"))

        with st.expander("📋 Parsed Job Requirements", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Required Skills", len(required))
                st.caption(display_list(required))

            with col2:
                st.metric("Preferred Skills", len(preferred))
                st.caption(display_list(preferred))

            with col3:
                exp_text = f"{experience:g}+ years" if experience is not None else "Not specified"
                st.metric("Minimum Experience", exp_text)
                st.caption(
                    " • ".join(
                        str(item.get("degree", item))
                        if isinstance(item, dict)
                        else str(item)
                        for item in education
                    )
                    or "No education requirement"
                )

        render_backend_json("🔎 View Raw Parsed JD", jd)

    # Resume upload
    st.markdown('<div class="section-title">2️⃣ Candidate Resumes</div>', unsafe_allow_html=True)
    st.caption(
        "All selected resumes are processed directly inside Streamlit. "
        "The backend creates one screening run and ranks the candidates."
    )

    uploaded_files = st.file_uploader(
        "Upload candidate resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload one or more PDF or DOCX resumes. Streamlit processes every selected file.",
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"{len(uploaded_files)} resume(s) selected.")

        can_process = bool(st.session_state.job_parsed)
        if not can_process:
            st.warning("Parse the job description before processing resumes.")

        if st.button(
            "⚡ Process & Screen All Resumes",
            type="primary",
            use_container_width=True,
            disabled=not can_process,
        ):
            with st.spinner(
                f"Running screening for {len(uploaded_files)} resume(s)..."
            ):
                results, errors = process_uploaded_resumes(uploaded_files)

            st.session_state.screening_results = results
            st.session_state.backend_resume_results = results
            st.session_state.screening_errors = errors
            st.session_state.data_source = "Streamlit"

            if results:
                st.success(
                    f"Batch screening completed: {len(results)} candidate(s) processed."
                )
            if errors:
                st.warning(
                    f"{len(errors)} candidate(s) could not be screened."
                )

    # Explicit demo mode
    if ENABLE_DEMO_MODE:
        with st.expander("🧪 Demo / UI Testing", expanded=False):
            st.caption(
                "Demo data is isolated from the live backend pipeline. "
                "It is useful for validating the recruiter interface."
            )
            if st.button("Load Demo Results", use_container_width=True):
                st.session_state.screening_results = load_demo_results()
                st.session_state.screening_errors = []
                st.session_state.data_source = "Demo"
                st.success("Demo screening results loaded.")

    # Errors
    if st.session_state.screening_errors:
        with st.expander(
            f"⚠️ Processing Errors ({len(st.session_state.screening_errors)})",
            expanded=False,
        ):
            st.dataframe(
                pd.DataFrame(st.session_state.screening_errors),
                use_container_width=True,
                hide_index=True,
            )

    # Screening overview
    results = st.session_state.screening_results
    if results:
        st.markdown('<div class="section-title">3️⃣ Screening Overview</div>', unsafe_allow_html=True)
        summary = calculate_summary(results)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Candidates", summary["total"])
        col2.metric("Shortlisted", summary["shortlisted"])
        col3.metric("Needs Review", summary["review"])
        col4.metric("Average Score", f'{summary["average"]:.1f}')

        st.markdown('<div class="section-title">4️⃣ Candidate Ranking</div>', unsafe_allow_html=True)

        filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
        with filter_col1:
            search_query = st.text_input(
                "🔎 Search candidates",
                placeholder="Search by filename...",
            )
        with filter_col2:
            decision_filter = st.selectbox(
                "Decision",
                ["All", "SHORTLIST", "REVIEW"],
            )
        with filter_col3:
            minimum_score = st.number_input(
                "Minimum score",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=5.0,
            )

        filtered = filter_candidates(
            results,
            search_query,
            decision_filter,
            minimum_score,
        )

        st.caption(f"Showing {len(filtered)} of {len(results)} candidates.")

        ranking_df = build_ranking_dataframe(filtered)
        if not ranking_df.empty:
            st.dataframe(
                ranking_df,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "⬇️ Export Ranking CSV",
                data=ranking_df.to_csv(index=False).encode("utf-8"),
                file_name="candidate_ranking.csv",
                mime="text/csv",
            )
        else:
            st.info("No candidates match the current filters.")


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

elif selected_page == "Candidates":
    st.markdown('<div class="main-title">Candidate Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Detailed evidence behind each screening recommendation</div>',
        unsafe_allow_html=True,
    )

    results = st.session_state.screening_results

    if not results:
        st.info("No screening results available. Run screening from the Dashboard.")
    else:
        candidate_names = [
            str(first_value(item, "candidate", "filename", default="Unknown"))
            for item in results
        ]
        selected_name = st.selectbox("Select candidate", candidate_names)
        candidate = next(
            item for item in results
            if str(first_value(item, "candidate", "filename", default="Unknown")) == selected_name
        )

        score = safe_float(candidate.get("score"))
        recommendation = str(candidate.get("recommendation", "REVIEW")).upper()

        st.subheader("🎯 Screening Decision")
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Score", f"{score:.2f}/100")
        col2.metric("Shortlist Threshold", f"{SHORTLIST_THRESHOLD:.2f}")
        if recommendation == "SHORTLIST":
            col3.success("SHORTLIST")
        else:
            col3.warning("REVIEW")

        st.subheader("📊 Score Breakdown")
        breakdown_rows = []
        for label, key, weight in SCORE_COMPONENTS:
            component_score = normalize_score(candidate.get(key, 0))
            breakdown_rows.append(
                {
                    "Component": label,
                    "Score": round(component_score, 2),
                    "Weight": weight,
                    "Weighted Contribution": round(component_score * weight / 100, 2),
                }
            )
        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

        st.subheader("🛠️ Skill Analysis")
        skill_col1, skill_col2 = st.columns(2)
        with skill_col1:
            st.markdown("#### ✅ Matched Required Skills")
            matched = list_value(candidate.get("matched_skills"))
            if matched:
                for skill in matched:
                    st.success(f"✓ {skill}")
            else:
                st.caption("No required skills matched.")
        with skill_col2:
            st.markdown("#### ❌ Missing Required Skills")
            missing = list_value(candidate.get("missing_skills"))
            if missing:
                for skill in missing:
                    st.error(f"✗ {skill}")
            else:
                st.success("✓ No required skills missing.")

        preferred = list_value(candidate.get("preferred_matched"))
        st.markdown("#### ⭐ Preferred Skills Matched")
        st.write(display_list(preferred))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💼 Experience")
            candidate_years = safe_float(candidate.get("candidate_years"))
            required_years = safe_float(candidate.get("required_years"))
            st.metric("Candidate Experience", f"{candidate_years:.2f} years")
            st.metric("Required Experience", f"{required_years:.2f} years")
            if candidate_years >= required_years:
                st.success("✓ Experience requirement satisfied.")
            else:
                st.warning("⚠ Experience requirement not satisfied.")

        with col2:
            st.subheader("🎓 Education")
            education_match = str(candidate.get("education_match") or "No education information available.")
            if candidate.get("education", 0) >= 100:
                st.success(education_match)
            else:
                st.warning(education_match)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💪 Strengths")
            strengths = list_value(candidate.get("strengths"))
            for strength in strengths or ["No strengths reported."]:
                st.write(f"• {strength}")

        with col2:
            st.subheader("⚠️ Concerns")
            concerns = list_value(candidate.get("concerns"))
            if concerns:
                for concern in concerns:
                    st.write(f"• {concern}")
            else:
                st.success("No major concerns reported.")

        st.subheader("🤖 ML Screening")
        prediction = candidate.get("ml_prediction")
        if prediction == 1:
            st.success("Fairness-aware ML model predicts SHORTLIST.")
        elif prediction == 0:
            st.warning("Fairness-aware ML model predicts REVIEW.")
        else:
            st.info("ML prediction unavailable in the backend response.")
        st.caption(
            "ML output is a decision-support signal, not an autonomous hiring decision."
        )

        # Actual SHAP explanation for the Logistic Regression model.
        render_shap_explanation(candidate)

        st.subheader("👤 Recruiter Decision")
        st.warning(
            "The recruiter makes the final decision. Review the underlying candidate evidence before deciding."
        )

        key = str(first_value(candidate, "candidate_id", "candidate", default=selected_name))
        current_decision = st.session_state.recruiter_decisions.get(key, "Pending Review")
        current_notes = st.session_state.recruiter_notes.get(key, "")

        decision_options = ["Pending Review", "Shortlist", "Hold", "Reject"]
        decision_col1, decision_col2 = st.columns(2)

        with decision_col1:
            recruiter_decision = st.selectbox(
                "Recruiter decision",
                decision_options,
                index=decision_options.index(current_decision),
                key=f"decision_{key}",
            )

        with decision_col2:
            if recommendation == "SHORTLIST":
                st.success("AI recommends SHORTLIST")
            else:
                st.warning("AI recommends REVIEW")

        recruiter_notes = st.text_area(
            "Recruiter notes",
            value=current_notes,
            height=120,
            placeholder="Document the reasoning behind the recruiter decision...",
            key=f"notes_{key}",
        )

        if st.button("💾 Save Recruiter Decision", type="primary"):
            st.session_state.recruiter_decisions[key] = recruiter_decision
            st.session_state.recruiter_notes[key] = recruiter_notes
            st.success(f"Recruiter decision saved: {recruiter_decision}")

        st.subheader("🧑‍⚖️ Human Review Requirement")
        st.info(
            "Final hiring decisions must remain with a qualified human recruiter. "
            "The system provides ranking, evidence and decision support."
        )

        if candidate.get("raw_screening"):
            render_backend_json("🔧 Backend Screening Response", candidate["raw_screening"])


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------

elif selected_page == "Fairness":
    st.markdown('<div class="main-title">Fairness & Responsible AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Model-level bias evaluation, mitigation and performance trade-offs</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Fairness is evaluated separately from predictive features. "
        "Protected attributes are not used as model inputs; they are used only for auditing and mitigation."
    )

    fairness_data = load_fairness_dashboard_data()

    if fairness_data:
        model_names = [
            name for name in ("logistic_regression", "knn", "fairness_aware")
            if isinstance(fairness_data.get(name), dict)
        ]
        # Older artifacts used the name 'baseline'. Normalize it for display.
        if not model_names and isinstance(fairness_data.get("baseline"), dict):
            model_names = ["baseline", "fairness_aware"]

        if model_names:
            # ---------------------------------------------------------------
            # KPI cards
            # ---------------------------------------------------------------
            baseline_name = "logistic_regression" if "logistic_regression" in model_names else "baseline"
            baseline = _get_model_section(fairness_data, baseline_name)
            fair = _get_model_section(fairness_data, "fairness_aware")

            baseline_dp = safe_float(
                nested(baseline, "fairness", "demographic_parity_difference", default=0)
            )
            fair_dp = safe_float(
                nested(fair, "fairness", "demographic_parity_difference", default=0)
            )
            dp_reduction = (
                ((baseline_dp - fair_dp) / baseline_dp) * 100
                if baseline_dp > 0
                else 0.0
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Baseline DP Difference", f"{baseline_dp:.4f}")
            col2.metric("Fairness-Aware DP", f"{fair_dp:.4f}")
            col3.metric("Measured Reduction", f"{dp_reduction:.1f}%")
            col4.metric("Models Evaluated", str(len(model_names)))

            # ---------------------------------------------------------------
            # Selection rate by protected group
            # ---------------------------------------------------------------
            st.subheader("👥 Selection Rate by Protected Group")
            group_frame = _group_rate_frame(fairness_data, model_names)
            if not group_frame.empty:
                display_frame = group_frame.copy()
                for column in display_frame.columns[1:]:
                    display_frame[column] = display_frame[column].map(
                        lambda value: f"{safe_float(value) * 100:.1f}%" if value is not None else "—"
                    )
                st.dataframe(display_frame, use_container_width=True, hide_index=True)

                chart_frame = group_frame.set_index("Protected Group")
                st.bar_chart(chart_frame, y_label="Selection rate", x_label="Protected group")
                st.caption(
                    "Selection rate is the proportion of candidates predicted as selected within each group. "
                    "A lower demographic parity difference indicates smaller selection-rate disparity."
                )
            else:
                st.warning("No group-level selection-rate data is available in the evaluation results.")

            # ---------------------------------------------------------------
            # Model comparison
            # ---------------------------------------------------------------
            st.subheader("📊 Model Performance vs Fairness")
            rows: list[dict[str, Any]] = []
            for model_name in model_names:
                section = _get_model_section(fairness_data, model_name)
                performance = section.get("performance", {})
                fairness = section.get("fairness", {})
                rows.append(
                    {
                        "Model": _model_display_name(model_name),
                        "Accuracy": performance.get("accuracy"),
                        "Precision": performance.get("precision"),
                        "Recall": performance.get("recall"),
                        "F1": performance.get("f1"),
                        "ROC-AUC": performance.get("roc_auc"),
                        "DP Difference": fairness.get("demographic_parity_difference"),
                    }
                )
            comparison = pd.DataFrame(rows)
            st.dataframe(comparison, use_container_width=True, hide_index=True)

            # ---------------------------------------------------------------
            # Interpretation
            # ---------------------------------------------------------------
            st.subheader("🔎 Fairness Interpretation")
            if fair_dp < baseline_dp:
                st.success(
                    f"The fairness-aware model reduced demographic parity difference "
                    f"from {baseline_dp:.4f} to {fair_dp:.4f} on the current synthetic held-out evaluation set."
                )
            elif fair_dp > baseline_dp:
                st.warning(
                    f"The fairness-aware model increased demographic parity difference "
                    f"from {baseline_dp:.4f} to {fair_dp:.4f} on the current evaluation set."
                )
            else:
                st.info("The fairness-aware model produced the same demographic parity difference as the baseline.")

            fair_f1 = safe_float(nested(fair, "performance", "f1", default=0))
            baseline_f1 = safe_float(nested(baseline, "performance", "f1", default=0))
            f1_delta = fair_f1 - baseline_f1
            st.write(
                f"**F1 trade-off:** fairness-aware F1 = **{fair_f1:.3f}**, "
                f"baseline F1 = **{baseline_f1:.3f}** "
                f"({f1_delta:+.3f})."
            )

            st.caption(
                "These results come from the project's synthetic evaluation dataset and should not be interpreted "
                "as evidence of real-world hiring performance. Fairness metrics are diagnostic indicators, not a "
                "guarantee of absence of discrimination."
            )
        else:
            st.warning("Fairness evaluation data was found, but no recognized model results are available.")
    else:
        st.info(
            f"No fairness evaluation artifact or evaluable model data was found. "
            f"Expected evaluation dataset: {EVALUATION_DATASET}"
        )

    # ---------------------------------------------------------------
    # Responsible AI controls
    # ---------------------------------------------------------------
    st.subheader("🔐 Responsible AI Controls")
    controls = [
        "Protected attributes are excluded from predictive model features.",
        "Sensitive information is anonymized before candidate processing.",
        "Protected attributes are isolated for fairness auditing and mitigation.",
        "Logistic Regression and KNN are evaluated against a fairness-aware model.",
        "Selection rates and demographic parity difference are monitored by group.",
        "SHAP explanations use the same job-relevant screening features as the model.",
        "Human recruiter remains responsible for the final hiring decision.",
        "Proxy discrimination remains a potential risk even when protected attributes are removed.",
        "Synthetic evaluation results must not be treated as real-world hiring performance.",
    ]
    for control in controls:
        st.write(f"✓ {control}")

    st.warning(
        "Removing protected attributes does not guarantee absence of bias. "
        "Other variables can act as proxies for protected characteristics, so fairness monitoring and human review remain necessary."
    )


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

elif selected_page == "System":
    st.markdown('<div class="main-title">System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Application architecture, configuration and readiness</div>',
        unsafe_allow_html=True,
    )

    st.subheader("🏗️ Current Architecture")
    st.code(
        """
Recruiter
   │
   ▼
Streamlit Dashboard
   │
   ▼
FastAPI API
   │
   ├── Job Description Processing
   ├── Resume Processing
   ├── PII Anonymization
   ├── Feature Extraction
   ├── Candidate-JD Matching
   ├── ML Classification
   ├── Fairness Evaluation
   └── Explainability
        │
        ▼
 Ranked Candidates
        │
        ▼
 Human Recruiter
        │
        ▼
    Audit Log
        """,
        language="text",
    )

    st.subheader("🚀 Final Target Architecture")
    st.code(
        """
Streamlit UI
    │
    ▼
 FastAPI
    │
    ├───────────────┬────────────────┬─────────────────┐
    ▼               ▼                ▼                 ▼
PostgreSQL      ChromaDB        File Storage      Model Registry
System of       Vector Search   PDF / DOCX        ML / Fairness
Record               │
                     ▼
              Semantic Matching
                     │
                     ▼
              Screening Pipeline
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        Rule         ML      Fairness
          │          │          │
          └──────────┼──────────┘
                     ▼
              Explainability
                     │
                     ▼
              Ranked Candidates
                     │
                     ▼
               Recruiter
                     │
                     ▼
                 Audit Log
        """,
        language="text",
    )

    st.subheader("🧰 Technology Stack")
    stack = pd.DataFrame(
        [
            {"Layer": "Frontend", "Technology": "Streamlit", "Status": "Active"},
            {"Layer": "API", "Technology": "FastAPI", "Status": "Integrated"},
            {"Layer": "Resume Parsing", "Technology": "PyMuPDF / python-docx", "Status": "Backend"},
            {"Layer": "NLP", "Technology": "spaCy", "Status": "Backend"},
            {"Layer": "Embeddings", "Technology": "Sentence Transformers", "Status": "Backend"},
            {"Layer": "Matching", "Technology": "Hybrid rule + semantic", "Status": "Backend"},
            {"Layer": "ML", "Technology": "scikit-learn", "Status": "Implemented"},
            {"Layer": "Fairness", "Technology": "Fairlearn", "Status": "Implemented"},
            {"Layer": "Explainability", "Technology": "SHAP + evidence-based", "Status": "Implemented"},
            {"Layer": "Database", "Technology": "SQLite → PostgreSQL", "Status": "Migration Ready"},
            {"Layer": "Vector Database", "Technology": "ChromaDB", "Status": "Integration Ready"},
            {"Layer": "Containerization", "Technology": "Docker", "Status": "Integration Ready"},
        ]
    )
    st.dataframe(stack, use_container_width=True, hide_index=True)

    st.subheader("⚙️ Runtime Configuration")
    config = pd.DataFrame(
        [
            {"Setting": "Backend URL", "Value": BACKEND_URL},
            {"Setting": "Request Timeout", "Value": f"{REQUEST_TIMEOUT}s"},
            {"Setting": "Shortlist Threshold", "Value": f"{SHORTLIST_THRESHOLD:.1f}"},
            {"Setting": "Demo Mode", "Value": "Enabled" if ENABLE_DEMO_MODE else "Disabled"},
            {"Setting": "Data Source", "Value": st.session_state.data_source},
        ]
    )
    st.dataframe(config, use_container_width=True, hide_index=True)

    st.subheader("🔌 API Connectivity")
    healthy, message = backend.health()
    if healthy:
        st.success(f"FastAPI reachable at {BACKEND_URL}")
    else:
        st.error(f"FastAPI unavailable: {message}")

    st.caption(
        "For Docker deployment, set BACKEND_URL to the FastAPI service name, "
        "for example http://api:8000. Do not hardcode container hostnames in source code."
    )
