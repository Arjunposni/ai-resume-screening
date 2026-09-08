from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.models.schemas import CandidateProfile, JobDescription


@dataclass
class SkillMatchResult:
    required_matched: list[str]
    required_missing: list[str]
    preferred_matched: list[str]
    required_match_score: float
    preferred_match_score: float


@dataclass
class ExperienceMatchResult:
    candidate_years: float
    required_years: float | None
    meets_requirement: bool
    score: float


@dataclass
class EducationMatchResult:
    matched_requirements: list[str]
    missing_requirements: list[str]
    score: float


@dataclass
class SemanticMatchResult:
    score: float


@dataclass
class MatchResult:
    final_score: float
    skill_match: SkillMatchResult
    experience_match: ExperienceMatchResult
    education_match: EducationMatchResult
    semantic_match: SemanticMatchResult


class CandidateMatcher:
    """
    Match a candidate profile against a job description.

    Final scoring:
        Required skills:     40%
        Experience:          25%
        Semantic similarity: 15%
        Education:           10%
        Preferred skills:    10%

    The matcher is deliberately deterministic except for the optional
    embedding model used for semantic similarity.
    """

    REQUIRED_SKILLS_WEIGHT = 0.40
    EXPERIENCE_WEIGHT = 0.25
    SEMANTIC_WEIGHT = 0.15
    EDUCATION_WEIGHT = 0.10
    PREFERRED_SKILLS_WEIGHT = 0.10

    def match(
        self,
        candidate: CandidateProfile,
        job: JobDescription,
        semantic_score: float | None = None,
        candidate_text: str | None = None,
        job_text: str | None = None,
    ) -> MatchResult:
        """Calculate the complete candidate-JD match safely."""

        skill_match = self.match_skills(
            candidate.skills or [],
            job.required_skills or [],
            job.preferred_skills or [],
        )

        experience_match = self.match_experience(
            candidate.total_experience_months or 0,
            job.min_experience_years,
        )

        education_match = self.match_education(candidate, job)

        if semantic_score is None:
            if candidate_text and job_text:
                semantic_score = self.calculate_semantic_similarity(
                    candidate_text,
                    job_text,
                )
            else:
                semantic_score = 0.0

        semantic_match = SemanticMatchResult(
            score=self._normalize_score(semantic_score)
        )

        final_score = (
            skill_match.required_match_score * self.REQUIRED_SKILLS_WEIGHT
            + experience_match.score * self.EXPERIENCE_WEIGHT
            + semantic_match.score * self.SEMANTIC_WEIGHT
            + education_match.score * self.EDUCATION_WEIGHT
            + skill_match.preferred_match_score * self.PREFERRED_SKILLS_WEIGHT
        )

        return MatchResult(
            final_score=round(max(0.0, min(final_score, 100.0)), 2),
            skill_match=skill_match,
            experience_match=experience_match,
            education_match=education_match,
            semantic_match=semantic_match,
        )

    def match_skills(
        self,
        candidate_skills: list[str],
        required_skills: list[str],
        preferred_skills: list[str],
    ) -> SkillMatchResult:
        """Compare candidate skills with job requirements."""

        candidate_set = {
            self._normalize_skill(skill)
            for skill in candidate_skills
            if self._normalize_skill(skill)
        }
        required_set = {
            self._normalize_skill(skill)
            for skill in required_skills
            if self._normalize_skill(skill)
        }
        preferred_set = {
            self._normalize_skill(skill)
            for skill in preferred_skills
            if self._normalize_skill(skill)
        }

        required_matched = sorted(candidate_set & required_set)
        required_missing = sorted(required_set - candidate_set)
        preferred_matched = sorted(candidate_set & preferred_set)

        required_score = (
            len(required_matched) / len(required_set) * 100.0
            if required_set
            else 100.0
        )
        preferred_score = (
            len(preferred_matched) / len(preferred_set) * 100.0
            if preferred_set
            else 100.0
        )

        return SkillMatchResult(
            required_matched=required_matched,
            required_missing=required_missing,
            preferred_matched=preferred_matched,
            required_match_score=round(required_score, 2),
            preferred_match_score=round(preferred_score, 2),
        )

    def match_experience(
        self,
        candidate_months: int,
        required_years: float | None,
    ) -> ExperienceMatchResult:
        """Compare candidate experience with an optional JD requirement."""

        try:
            safe_months = max(0, int(candidate_months or 0))
        except (TypeError, ValueError):
            safe_months = 0

        candidate_years = safe_months / 12.0

        if required_years is None:
            return ExperienceMatchResult(
                candidate_years=round(candidate_years, 2),
                required_years=None,
                meets_requirement=True,
                score=100.0,
            )

        try:
            safe_required = float(required_years)
        except (TypeError, ValueError):
            safe_required = None

        if safe_required is None or safe_required <= 0:
            return ExperienceMatchResult(
                candidate_years=round(candidate_years, 2),
                required_years=safe_required,
                meets_requirement=True,
                score=100.0,
            )

        ratio = candidate_years / safe_required
        score = min(max(ratio, 0.0), 1.0) * 100.0

        return ExperienceMatchResult(
            candidate_years=round(candidate_years, 2),
            required_years=safe_required,
            meets_requirement=candidate_years >= safe_required,
            score=round(score, 2),
        )

    def match_education(
        self,
        candidate: CandidateProfile,
        job: JobDescription,
    ) -> EducationMatchResult:
        """Compare candidate education with required education."""

        requirements = [
            requirement
            for requirement in (job.education_requirements or [])
            if requirement.required
        ]

        if not requirements:
            return EducationMatchResult([], [], 100.0)

        candidate_education = [
            {
                "degree": self._normalize_text(education.degree),
                "field_of_study": self._normalize_text(
                    education.field_of_study
                ),
                "institution": self._normalize_text(education.institution),
            }
            for education in (candidate.education or [])
        ]

        matched: list[str] = []
        missing: list[str] = []

        for requirement in requirements:
            required_degree = self._normalize_text(requirement.degree)
            required_field = self._normalize_text(
                requirement.field_of_study
            )

            matched_candidate = None
            for education in candidate_education:
                degree_ok = self._degree_matches(
                    required_degree,
                    education["degree"],
                )
                field_ok = (
                    not required_field
                    or self._field_matches(
                        required_field,
                        education["field_of_study"],
                        education["degree"],
                    )
                )

                if degree_ok and field_ok:
                    matched_candidate = education
                    break

            description = requirement.degree
            if requirement.field_of_study:
                description += f" in {requirement.field_of_study}"

            if matched_candidate is not None:
                matched.append(description)
            else:
                missing.append(description)

        score = len(matched) / len(requirements) * 100.0

        return EducationMatchResult(
            matched_requirements=matched,
            missing_requirements=missing,
            score=round(score, 2),
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_embedding_model():
        """Load the embedding model once per backend process."""
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")

    def calculate_semantic_similarity(
        self,
        candidate_text: str,
        job_text: str,
    ) -> float:
        """
        Calculate semantic similarity using all-MiniLM-L6-v2.

        Returns a score from 0 to 100. Model loading is cached so 100+
        candidate requests do not repeatedly load the model into memory.
        """

        candidate_text = str(candidate_text or "").strip()
        job_text = str(job_text or "").strip()

        if not candidate_text or not job_text:
            return 0.0

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            model = self._get_embedding_model()
            embeddings = model.encode(
                [candidate_text, job_text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            similarity = float(
                cosine_similarity(
                    [embeddings[0]],
                    [embeddings[1]],
                )[0][0]
            )

            similarity = max(0.0, min(similarity, 1.0))
            return round(similarity * 100.0, 2)

        except Exception:
            # Semantic matching is an enhancement. A model/runtime problem
            # must not crash deterministic screening.
            return 0.0

    @staticmethod
    def _normalize_score(score: Any) -> float:
        """Normalize scores expressed as either 0-1 or 0-100."""

        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0

        if 0.0 <= value <= 1.0:
            value *= 100.0

        return round(max(0.0, min(value, 100.0)), 2)

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        if not value:
            return ""
        return " ".join(str(value).lower().strip().split())

    @classmethod
    def _normalize_skill(cls, value: str | None) -> str:
        return cls._normalize_text(value)

    @classmethod
    def _field_matches(
        cls,
        required_field: str,
        candidate_field: str,
        candidate_degree: str,
    ) -> bool:
        """Match education fields while allowing common resume wording."""

        if not required_field:
            return True

        candidate_field = cls._normalize_text(candidate_field)
        candidate_degree = cls._normalize_text(candidate_degree)

        if required_field in candidate_field:
            return True

        # Some extractors keep the field inside the degree string.
        if required_field in candidate_degree:
            return True

        aliases = {
            "computer science": {
                "computer science",
                "computer science engineering",
                "cse",
                "information technology",
                "information systems",
            },
            "information technology": {
                "information technology",
                "it",
                "information systems",
                "computer science",
            },
            "software engineering": {
                "software engineering",
                "computer science",
                "computer engineering",
            },
        }

        required_aliases = aliases.get(
            required_field,
            {required_field},
        )

        combined = f"{candidate_field} {candidate_degree}".strip()
        return any(alias in combined for alias in required_aliases)

    @classmethod
    def _degree_matches(
        cls,
        required_degree: str,
        candidate_education: str,
    ) -> bool:
        """Match common degree families without requiring exact wording."""

        required = cls._normalize_text(required_degree)
        candidate = cls._normalize_text(candidate_education)

        if not required or not candidate:
            return False

        degree_groups = {
            "bachelor": {
                "bachelor",
                "bachelor's",
                "bachelor's degree",
                "b.s",
                "b.s.",
                "bs",
                "b.e",
                "b.e.",
                "be",
                "b.tech",
                "btech",
                "bca",
            },
            "master": {
                "master",
                "master's",
                "master's degree",
                "m.s",
                "m.s.",
                "ms",
                "m.e",
                "m.e.",
                "me",
                "m.tech",
                "mtech",
                "mca",
                "mba",
            },
            "phd": {
                "ph.d",
                "ph.d.",
                "phd",
                "doctorate",
                "doctoral",
            },
        }

        for family, aliases in degree_groups.items():
            required_matches_family = any(
                alias == required
                or alias in required
                for alias in aliases
            )
            if not required_matches_family:
                continue

            return any(
                alias == candidate
                or alias in candidate
                for alias in aliases
            )

        return required in candidate or candidate in required


def candidate_profile_to_text(candidate: CandidateProfile) -> str:
    """
    Convert structured candidate data into deterministic semantic text.

    This is used when the original anonymized resume text is not available
    at /screening/match. It ensures semantic similarity is not silently
    forced to zero just because the API receives a structured profile.
    """

    parts: list[str] = []

    if candidate.skills:
        parts.append("Skills: " + ", ".join(candidate.skills))

    if candidate.experience:
        for experience in candidate.experience:
            fields = [
                experience.job_title,
                experience.company,
                experience.location,
                experience.description[0]
                if experience.description
                else None,
            ]
            parts.append(
                "Experience: " + " | ".join(
                    str(value) for value in fields if value
                )
            )

    if candidate.education:
        for education in candidate.education:
            parts.append(
                "Education: "
                + " | ".join(
                    str(value)
                    for value in [
                        education.degree,
                        education.field_of_study,
                        education.institution,
                    ]
                    if value
                )
            )

    if candidate.certifications:
        parts.append(
            "Certifications: "
            + ", ".join(cert.name for cert in candidate.certifications)
        )

    if candidate.projects:
        for project in candidate.projects:
            parts.append(
                "Project: "
                + " | ".join(
                    str(value)
                    for value in [
                        project.name,
                        project.description,
                        ", ".join(project.technologies),
                    ]
                    if value
                )
            )

    return "\n".join(parts)


candidate_matcher = CandidateMatcher()
