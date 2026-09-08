from __future__ import annotations

from app.services.matcher import MatchResult


FEATURE_COLUMNS = [
    "required_skill_score",
    "experience_score",
    "education_score",
    "semantic_score",
    "preferred_skill_score",
]


def match_result_to_features(result: MatchResult) -> dict[str, float]:
    """
    Convert the deterministic matcher result into the feature
    vector expected by the ML fairness model.
    """

    return {
        "required_skill_score": float(
            result.skill_match.required_match_score
        ),
        "experience_score": float(
            result.experience_match.score
        ),
        "education_score": float(
            result.education_match.score
        ),
        "semantic_score": float(
            result.semantic_match.score
        ),
        "preferred_skill_score": float(
            result.skill_match.preferred_match_score
        ),
    }