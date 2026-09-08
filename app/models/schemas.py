# Pydantic schemas will go here.
from pydantic import BaseModel, Field


class Experience(BaseModel):
    """A single professional experience entry."""

    job_title: str
    company: str | None = None
    location: str | None = None

    start_date: str | None = None
    end_date: str | None = None

    description: list[str] = Field(default_factory=list)


class Education(BaseModel):
    """A single education entry."""

    degree: str
    institution: str | None = None

    start_year: int | None = None
    end_year: int | None = None


class Certification(BaseModel):
    """A professional certification."""

    name: str
    year: int | None = None


class Project(BaseModel):
    """A candidate project."""

    name: str
    description: str | None = None

    technologies: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    """
    Structured representation of a candidate.

    Protected attributes and direct identifiers are intentionally
    excluded from this model because this profile is intended
    for downstream screening and ranking.
    """

    skills: list[str] = Field(default_factory=list)

    experience: list[Experience] = Field(default_factory=list)

    education: list[Education] = Field(default_factory=list)

    certifications: list[Certification] = Field(default_factory=list)

    projects: list[Project] = Field(default_factory=list)

    total_experience_months: int = 0


class EducationRequirement(BaseModel):
    """A single education requirement extracted from a job description."""

    degree: str
    field_of_study: str | None = None
    required: bool = True


class JobDescription(BaseModel):
    """
    Structured representation of a job description.

    required_skills and preferred_skills are kept as separate fields
    (rather than one skills list with a flag) so that scoring weights
    can be applied independently and unambiguously downstream —
    required skills must carry more weight than preferred skills.
    """

    title: str
    raw_text: str

    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)

    min_experience_years: float | None = None

    education_requirements: list[EducationRequirement] = Field(default_factory=list)

    required_certifications: list[str] = Field(default_factory=list)

    other_requirements: list[str] = Field(default_factory=list)