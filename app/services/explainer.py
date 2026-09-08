from __future__ import annotations

from app.services.matcher import MatchResult


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convert optional numeric values safely without crashing explanations."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_level(score: float) -> str:
    """
    Convert a numeric score into a human-readable level.
    """

    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Strong"
    if score >= 60:
        return "Moderate"
    if score >= 40:
        return "Weak"

    return "Low"


def _build_score_breakdown(result: MatchResult) -> dict:
    """
    Explain how the deterministic matching score is composed.

    Matcher weights:
        Required skills  = 40%
        Experience       = 25%
        Semantic         = 15%
        Education        = 10%
        Preferred skills = 10%
    """

    required_score = _safe_float(
        result.skill_match.required_match_score
    )

    experience_score = _safe_float(
        result.experience_match.score
    )

    education_score = _safe_float(
        result.education_match.score
    )

    semantic_score = _safe_float(
        result.semantic_match.score
    )

    preferred_score = _safe_float(
        result.skill_match.preferred_match_score
    )

    return {
        "required_skills": {
            "score": required_score,
            "weight": 0.40,
            "weighted_contribution": round(
                required_score * 0.40,
                2,
            ),
            "level": _score_level(required_score),
        },
        "experience": {
            "score": experience_score,
            "weight": 0.25,
            "weighted_contribution": round(
                experience_score * 0.25,
                2,
            ),
            "level": _score_level(experience_score),
        },
        "semantic_similarity": {
            "score": semantic_score,
            "weight": 0.15,
            "weighted_contribution": round(
                semantic_score * 0.15,
                2,
            ),
            "level": _score_level(semantic_score),
        },
        "education": {
            "score": education_score,
            "weight": 0.10,
            "weighted_contribution": round(
                education_score * 0.10,
                2,
            ),
            "level": _score_level(education_score),
        },
        "preferred_skills": {
            "score": preferred_score,
            "weight": 0.10,
            "weighted_contribution": round(
                preferred_score * 0.10,
                2,
            ),
            "level": _score_level(preferred_score),
        },
    }


def _build_skill_explanation(result: MatchResult) -> dict:
    """
    Explain required and preferred skill matching.
    """

    required_matched = list(
        result.skill_match.required_matched
    )

    required_missing = list(
        result.skill_match.required_missing
    )

    preferred_matched = list(
        result.skill_match.preferred_matched
    )

    return {
        "required": {
            "matched": required_matched,
            "missing": required_missing,
            "matched_count": len(required_matched),
            "missing_count": len(required_missing),
            "score": float(
                result.skill_match.required_match_score
            ),
        },
        "preferred": {
            "matched": preferred_matched,
            "score": float(
                result.skill_match.preferred_match_score
            ),
        },
    }


def _build_experience_explanation(
    result: MatchResult,
) -> dict:
    """
    Explain candidate experience against the JD requirement.
    """

    # Experience requirements are optional in a job description.
    # Never allow a missing requirement (None) to crash screening.
    raw_candidate_years = result.experience_match.candidate_years
    raw_required_years = result.experience_match.required_years

    try:
        candidate_years = float(raw_candidate_years or 0.0)
    except (TypeError, ValueError):
        candidate_years = 0.0

    if raw_required_years is None:
        required_years = None
        meets_requirement = True
        summary = (
            f"Candidate has approximately {candidate_years:.2f} years "
            "of experience. No minimum experience requirement was "
            "specified in the job description."
        )
    else:
        try:
            required_years = float(raw_required_years)
        except (TypeError, ValueError):
            required_years = None

        if required_years is None:
            meets_requirement = True
            summary = (
                f"Candidate has approximately {candidate_years:.2f} years "
                "of experience. The job description did not provide a "
                "valid minimum experience requirement."
            )
        else:
            meets_requirement = bool(
                result.experience_match.meets_requirement
            )
            if meets_requirement:
                summary = (
                    f"Candidate has {candidate_years:.2f} years of "
                    f"experience against {required_years:.2f} years "
                    "required."
                )
            else:
                summary = (
                    f"Candidate has {candidate_years:.2f} years of "
                    f"experience against {required_years:.2f} years "
                    "required."
                )

    try:
        experience_score = float(result.experience_match.score or 0.0)
    except (TypeError, ValueError):
        experience_score = 0.0

    return {
        "candidate_years": candidate_years,
        "required_years": required_years,
        "meets_requirement": meets_requirement,
        "requirement_specified": required_years is not None,
        "score": experience_score,
        "summary": summary,
    }


def _build_education_explanation(
    result: MatchResult,
) -> dict:
    """
    Explain education requirement matching.
    """

    matched = list(
        result.education_match.matched_requirements
    )

    missing = list(
        result.education_match.missing_requirements
    )

    return {
        "matched_requirements": matched,
        "missing_requirements": missing,
        "score": _safe_float(
            result.education_match.score
        ),
        "meets_requirement": len(missing) == 0,
    }


def _build_semantic_explanation(
    result: MatchResult,
) -> dict:
    """
    Explain semantic similarity between candidate and JD.
    """

    score = float(result.semantic_match.score)

    if score >= 80:
        interpretation = (
            "Strong semantic alignment between the "
            "candidate profile and job requirements."
        )
    elif score >= 60:
        interpretation = (
            "Moderate semantic alignment between the "
            "candidate profile and job requirements."
        )
    else:
        interpretation = (
            "Low semantic alignment between the "
            "candidate profile and job requirements."
        )

    return {
        "score": score,
        "level": _score_level(score),
        "interpretation": interpretation,
    }


def _build_strengths(
    result: MatchResult,
) -> list[str]:
    """
    Identify positive evidence supporting the candidate.
    """

    strengths: list[str] = []

    required_score = _safe_float(
        result.skill_match.required_match_score
    )

    experience_score = _safe_float(
        result.experience_match.score
    )

    education_score = _safe_float(
        result.education_match.score
    )

    semantic_score = _safe_float(
        result.semantic_match.score
    )

    preferred_score = _safe_float(
        result.skill_match.preferred_match_score
    )

    required_matched = (
        result.skill_match.required_matched
    )

    preferred_matched = (
        result.skill_match.preferred_matched
    )

    if required_score >= 100:
        strengths.append(
            "All required skills are matched."
        )
    elif required_matched:
        strengths.append(
            "The candidate matches several required skills: "
            + ", ".join(required_matched)
            + "."
        )

    if experience_score >= 100:
        strengths.append(
            "The candidate satisfies the minimum experience requirement."
        )
    elif experience_score >= 75:
        strengths.append(
            "The candidate has substantial relevant experience."
        )

    if education_score >= 100:
        strengths.append(
            "The candidate satisfies the stated education requirement."
        )

    if semantic_score >= 80:
        strengths.append(
            "The candidate profile has strong semantic alignment "
            "with the job description."
        )
    elif semantic_score >= 60:
        strengths.append(
            "The candidate profile has moderate semantic alignment "
            "with the job description."
        )

    if preferred_matched:
        strengths.append(
            "Preferred skills matched: "
            + ", ".join(preferred_matched)
            + "."
        )

    if preferred_score >= 100:
        strengths.append(
            "All preferred skills are matched."
        )

    return strengths


def _build_concerns(
    result: MatchResult,
) -> list[str]:
    """
    Identify evidence that may require recruiter review.
    """

    concerns: list[str] = []

    required_missing = (
        result.skill_match.required_missing
    )

    if required_missing:
        concerns.append(
            "Missing required skills: "
            + ", ".join(required_missing)
            + "."
        )

    if not result.experience_match.meets_requirement:
        concerns.append(
            "Candidate does not meet the minimum experience requirement."
        )

    missing_education = (
        result.education_match.missing_requirements
    )

    if missing_education:
        concerns.append(
            "Education requirements requiring review: "
            + ", ".join(missing_education)
            + "."
        )

    preferred_score = _safe_float(
        result.skill_match.preferred_match_score
    )

    if preferred_score < 100:
        concerns.append(
            "Some preferred skills were not matched."
        )

    semantic_score = _safe_float(
        result.semantic_match.score
    )

    if semantic_score < 60:
        concerns.append(
            "Semantic similarity with the job description is relatively low."
        )

    return concerns


def _build_ml_explanation(
    ml_prediction: int | None,
    deterministic_recommendation: str,
) -> dict:
    """
    Explain the fairness-aware ML prediction without
    exposing protected attributes.
    """

    if ml_prediction is None:
        return {
            "prediction": None,
            "decision": "UNAVAILABLE",
            "agreement": "UNAVAILABLE",
            "explanation": (
                "The fairness-aware ML model prediction "
                "was not available."
            ),
        }

    try:
        normalized_prediction = int(ml_prediction)
    except (TypeError, ValueError):
        normalized_prediction = None

    if normalized_prediction not in {0, 1}:
        return {
            "prediction": None,
            "decision": "UNAVAILABLE",
            "agreement": "UNAVAILABLE",
            "explanation": (
                "The fairness-aware ML model returned an invalid "
                "prediction, so it was excluded from the explanation."
            ),
        }

    if normalized_prediction == 1:
        ml_decision = "SHORTLIST"
    else:
        ml_decision = "REVIEW"

    if ml_decision == deterministic_recommendation:
        agreement = "AGREE"

        explanation = (
            f"The fairness-aware ML model predicts "
            f"{ml_decision}, consistent with the "
            f"deterministic screening recommendation."
        )
    else:
        agreement = "DISAGREE"

        explanation = (
            f"The fairness-aware ML model predicts "
            f"{ml_decision}, while the deterministic "
            f"screening recommendation is "
            f"{deterministic_recommendation}. "
            "Recruiter review is recommended."
        )

    return {
        "prediction": normalized_prediction,
        "decision": ml_decision,
        "agreement": agreement,
        "explanation": explanation,
    }


def _build_human_review_guidance(
    result: MatchResult,
    ml_prediction: int | None,
    recommendation: str,
) -> dict:
    """
    Provide human-in-the-loop guidance.

    The system is a decision-support tool, not an autonomous
    hiring decision-maker.
    """

    reasons: list[str] = []

    if result.skill_match.required_missing:
        reasons.append(
            "Review missing required skills."
        )

    if not result.experience_match.meets_requirement:
        reasons.append(
            "Review experience gap."
        )

    if result.education_match.missing_requirements:
        reasons.append(
            "Review education requirements."
        )

    if ml_prediction is not None:
        ml_decision = (
            "SHORTLIST"
            if ml_prediction == 1
            else "REVIEW"
        )

        if ml_decision != recommendation:
            reasons.append(
                "ML and deterministic screening signals disagree."
            )

    if reasons:
        priority = "HIGH"
    elif recommendation == "SHORTLIST":
        priority = "NORMAL"
    else:
        priority = "HIGH"

    return {
        "human_review_required": True,
        "priority": priority,
        "reasons": reasons,
        "message": (
            "Final hiring decisions must remain with a qualified "
            "human recruiter. The system provides decision support "
            "and evidence for review."
        ),
    }


def generate_screening_explanation(
    result: MatchResult,
    ml_prediction: int | None = None,
    recommendation: str | None = None,
    shortlist_threshold: float = 75.0,
) -> dict:
    """
    Generate the complete recruiter-facing explanation.

    This function combines:

    - score breakdown
    - skill evidence
    - experience evidence
    - education evidence
    - semantic similarity
    - strengths
    - concerns
    - ML prediction
    - model agreement
    - human review guidance
    - fairness/sensitive-attribute safeguards
    """

    if recommendation is None:
        recommendation = (
            "SHORTLIST"
            if result.final_score >= shortlist_threshold
            else "REVIEW"
        )

    strengths = _build_strengths(result)
    concerns = _build_concerns(result)

    score_breakdown = _build_score_breakdown(result)
    skill_explanation = _build_skill_explanation(result)
    experience_explanation = _build_experience_explanation(
        result
    )
    education_explanation = _build_education_explanation(
        result
    )
    semantic_explanation = _build_semantic_explanation(
        result
    )

    ml_explanation = _build_ml_explanation(
        ml_prediction=ml_prediction,
        deterministic_recommendation=recommendation,
    )

    human_review = _build_human_review_guidance(
        result=result,
        ml_prediction=ml_prediction,
        recommendation=recommendation,
    )

    # ---------------------------------------------------------
    # Recruiter summary
    # ---------------------------------------------------------

    if recommendation == "SHORTLIST":
        summary = (
            f"Candidate scored {result.final_score:.2f}/100 "
            f"and meets the current screening threshold of "
            f"{shortlist_threshold:.2f}."
        )
    else:
        summary = (
            f"Candidate scored {result.final_score:.2f}/100 "
            f"which is below the current screening threshold "
            f"of {shortlist_threshold:.2f}."
        )

    # ---------------------------------------------------------
    # Fairness safeguards
    # ---------------------------------------------------------

    fairness_note = {
        "protected_attributes_used_for_screening": False,
        "protected_attributes_used_for_ranking": False,
        "protected_attributes_used_for_explanation": False,
        "purpose": (
            "Protected or sensitive attributes are kept outside "
            "the candidate ranking features. They may be maintained "
            "separately for controlled fairness auditing and "
            "bias evaluation."
        ),
        "warning": (
            "Removing protected attributes does not guarantee "
            "absence of bias because other features may act as proxies."
        ),
    }

    return {
        "summary": summary,

        "decision": {
            "recommendation": recommendation,
            "final_score": round(
                _safe_float(result.final_score),
                2,
            ),
            "shortlist_threshold": shortlist_threshold,
        },

        "score_breakdown": score_breakdown,

        "skills": skill_explanation,

        "experience": experience_explanation,

        "education": education_explanation,

        "semantic_similarity": semantic_explanation,

        "strengths": strengths,

        "concerns": concerns,

        "ml_model": ml_explanation,

        "human_review": human_review,

        "fairness": fairness_note,
    }