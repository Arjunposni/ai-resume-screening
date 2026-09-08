from app.models.schemas import (
    CandidateProfile,
    Experience,
    Education,
    JobDescription,
    EducationRequirement,
)
from app.services.matcher import candidate_matcher


candidate = CandidateProfile(
    skills=["Python", "FastAPI", "Docker"],
    experience=[
        Experience(
            job_title="Software Engineer",
            start_date="Jun 2023",
            end_date="Aug 2026",
        )
    ],
    education=[
        Education(
            degree="B.S. Computer Science",
            institution="University of Texas",
            start_year=2019,
            end_year=2023,
        )
    ],
    total_experience_months=38,
)

job = JobDescription(
    title="Backend Engineer",
    raw_text="Backend Engineer. Python FastAPI PostgreSQL Docker. 3+ years experience. Bachelor's degree in Computer Science.",
    required_skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
    preferred_skills=["AWS", "Kubernetes"],
    min_experience_years=3,
    education_requirements=[
        EducationRequirement(
            degree="Bachelor's Degree",
            field_of_study="Computer Science",
            required=True,
        )
    ],
)

result = candidate_matcher.match(
    candidate,
    job,
    semantic_score=0.82,
)

print(result)