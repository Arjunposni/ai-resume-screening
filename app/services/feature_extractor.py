# Candidate/JD feature extraction.

import re
from datetime import datetime

from app.models.schemas import CandidateProfile


# Initial skill dictionary.
SKILL_DICTIONARY = {
    # Programming languages
    "python",
    "java",
    "javascript",
    "typescript",
    "c",
    "c++",
    "c#",
    "go",
    "rust",
    "kotlin",
    "swift",

    # Backend / APIs
    "fastapi",
    "flask",
    "django",
    "spring",
    "spring boot",
    "rest api",
    "graphql",

    # Frontend
    "react",
    "angular",
    "vue",
    "html",
    "css",
    "tailwind",

    # Data / AI
    "machine learning",
    "deep learning",
    "natural language processing",
    "nlp",
    "generative ai",
    "llm",
    "rag",
    "retrieval augmented generation",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "opencv",

    # Databases
    "sql",
    "mysql",
    "postgresql",
    "sqlite",
    "mongodb",
    "redis",
    "oracle",

    # Cloud / DevOps
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "github",
    "gitlab",
    "linux",
    "ci/cd",
    "jenkins",

    # Data engineering
    "apache spark",
    "hadoop",
    "kafka",
    "airflow",

    # Other
    "microservices",
    "system design",
    "data structures",
    "algorithms",
}


class FeatureExtractor:
    """Extract structured, job-relevant features from resume text."""

    # Example:
    # Software Engineer — Northstar Analytics | Austin, TX | Jun 2023 - Aug 2026
    EXPERIENCE_DATE_PATTERN = re.compile(
        r"(?P<start>"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s+\d{4}|\d{4})"
        r"\s*(?:[-–—]|to)\s*"
        r"(?P<end>"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s+\d{4}|\d{4}|Present|Current)",
        re.IGNORECASE,
    )
    EDUCATION_PATTERN = re.compile(
        r"^(?P<degree>"
        r"(?:B\.?S\.?|B\.?Tech|Bachelor(?:'s)?|M\.?S\.?|M\.?Tech|"
        r"Master(?:'s)?|Ph\.?D\.?|Doctor(?:ate)?|MBA|MCA|BCA)"
        r".*?)"
        r"\s*\|\s*"
        r"(?P<institution>.+?)"
        r"\s*\|\s*"
        r"(?P<start>\d{4})"
        r"\s*(?:[-–—]|to)\s*"
        r"(?P<end>\d{4})$",
        re.IGNORECASE,
    )
    CERTIFICATION_PATTERN = re.compile(
        r"^(?P<name>.+?)"
        r"\s*(?:\||[-–—])\s*"
        r"(?P<year>20\d{2})$",
        re.IGNORECASE,
    )
    PROJECT_PATTERN = re.compile(
        r"^(?P<name>.+?)"
        r"\s*\|\s*"
        r"(?P<technologies>.+)$",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> CandidateProfile:
        """Extract all structured candidate features."""

        if not text or not text.strip():
            return CandidateProfile()

        skills = self.extract_skills(text)
        experience = self.extract_experience(text)
        education = self.extract_education(text)
        certifications = self.extract_certifications(text)
        projects = self.extract_projects(text)

        total_experience_months = self.calculate_total_experience(
            experience
        )

        return CandidateProfile(
    skills=skills,
    experience=experience,
    education=education,
    certifications=certifications,
    projects=projects,
    total_experience_months=total_experience_months,
)
    def extract_projects(self, text: str) -> list[dict]:
        """Extract project names and technologies from resume lines."""

        projects = []

        project_keywords = (
            "project",
            "projects",
        )

        in_projects_section = False

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            normalized_line = line.lower()

            # Detect the beginning of the Projects section.
            if normalized_line in {
                "projects",
                "projects:",
                "academic projects",
                "personal projects",
            }:
                in_projects_section = True
                continue

            # Stop when another major section begins.
            if in_projects_section and normalized_line in {
                "experience",
                "work experience",
                "education",
                "certifications",
                "skills",
                "professional summary",
            }:
                in_projects_section = False
                continue

            if not in_projects_section:
                continue

            match = self.PROJECT_PATTERN.match(line)

            if not match:
                continue

            name = match.group("name").strip()
            technology_text = match.group("technologies").strip()

            technologies = [
                technology.strip()
                for technology in technology_text.split(",")
                if technology.strip()
            ]

            projects.append(
                {
                    "name": name,
                    "description": None,
                    "technologies": technologies,
                }
            )

        return projects
    def extract_certifications(self, text: str) -> list[dict]:
        """Extract certification records from resume lines."""

        certifications = []

        certification_keywords = (
            "certified",
            "certification",
            "certificate",
        )

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            normalized_line = line.lower()

            if normalized_line.rstrip(":") in {
    "certifications",
    "certification",
    "certificates",
    "licenses",
    "licenses and certifications",
}:
               continue

            if not any(
    keyword in normalized_line
    for keyword in certification_keywords
):
                continue

            # Look for a year anywhere in the certification line.
            year_match = re.search(r"\b(20\d{2})\b", line)

            if year_match:
                year = int(year_match.group(1))

                # Remove the year and common separators.
                name = re.sub(
                    r"\s*(?:\||[-–—])?\s*20\d{2}\s*$",
                    "",
                    line,
                ).strip()

                certifications.append(
                    {
                        "name": name,
                        "year": year,
                    }
                )
            else:
                certifications.append(
                    {
                        "name": line,
                        "year": None,
                    }
                )

        return certifications
    
    def extract_skills(self, text: str) -> list[str]:
        """Extract skills using the deterministic skill dictionary."""

        normalized_text = text.lower()
        detected_skills = []

        for skill in SKILL_DICTIONARY:
            pattern = self._build_skill_pattern(skill)

            if re.search(pattern, normalized_text):
                detected_skills.append(skill)

        return sorted(detected_skills)
    def extract_experience(self, text: str) -> list[dict]:
        """Extract employment records from the experience section."""

        experiences = []

        section_starts = {
            "experience",
            "work experience",
            "professional experience",
            "employment experience",
            "employment history",
        }

        section_stops = {
            "education",
            "certifications",
            "projects",
            "academic projects",
            "personal projects",
            "skills",
            "professional summary",
            "summary",
            "objective",
        }

        date_pattern = re.compile(
            r"(?P<start>"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}"
            r"|\d{4})"
            r"\s*(?:-|–|—|to)\s*"
            r"(?P<end>"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}"
            r"|\d{4}|Present|Current)"
            r"\s*$",
            re.IGNORECASE,
        )

        in_experience_section = False

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            normalized = line.lower().rstrip(":")

            if normalized in section_starts:
                in_experience_section = True
                continue

            if normalized in section_stops:
                in_experience_section = False
                continue

            if not in_experience_section:
                continue

            date_match = date_pattern.search(line)

            if not date_match:
                continue

            start_date = date_match.group("start").strip()
            end_date = date_match.group("end").strip()

            header = line[:date_match.start()].strip()
            header = re.sub(r"[\s|]+$", "", header)

            parts = re.split(
                r"\s*[—–-]\s*|\s*\|\s*",
                header,
            )

            parts = [
                part.strip()
                for part in parts
                if part.strip()
            ]

            if len(parts) < 2:
                continue

            job_title = parts[0]
            company = parts[1]

            location = None

            if len(parts) > 2:
                location = " | ".join(parts[2:])

            experiences.append(
                {
                    "job_title": job_title,
                    "company": company,
                    "location": location,
                    "start_date": start_date,
                    "end_date": end_date,
                    "description": [],
                }
            )

        return experiences

    def extract_projects(self, text: str) -> list[dict]:
        """Extract projects from the Projects section."""

        projects = []

        project_headers = {
            "projects",
            "academic projects",
            "personal projects",
            "technical projects",
        }

        stop_headers = {
            "experience",
            "work experience",
            "professional experience",
            "education",
            "certifications",
            "skills",
            "professional summary",
            "summary",
            "objective",
        }

        in_projects_section = False

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            normalized = line.lower().rstrip(":")

            # Start Projects section.
            if normalized in project_headers:
                in_projects_section = True
                continue

            # Stop at the next major section.
            if normalized in stop_headers:
                in_projects_section = False
                continue

            if not in_projects_section:
                continue

            # Project format:
            # Project Name | Python, FastAPI, Docker
            match = re.match(
                r"^(?P<name>[^|]+?)"
                r"\s*\|\s*"
                r"(?P<technologies>.+)$",
                line,
            )

            if not match:
                continue

            name = match.group("name").strip()
            technology_text = match.group("technologies").strip()

            technologies = [
                technology.strip().lower()
                for technology in technology_text.split(",")
                if technology.strip()
            ]

            projects.append(
                {
                    "name": name,
                    "description": None,
                    "technologies": technologies,
                }
            )

        return projects
    
    def extract_education(self, text: str) -> list[dict]:
        """Extract education records from structured resume lines."""

        education = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            match = self.EDUCATION_PATTERN.match(line)

            if not match:
                continue

            education.append(
                {
                    "degree": match.group("degree").strip(),
                    "institution": match.group("institution").strip(),
                    "start_year": int(match.group("start")),
                    "end_year": int(match.group("end")),
                }
            )

        return education

    @staticmethod
    def calculate_total_experience(experiences: list[dict]) -> int:
        """Calculate approximate total professional experience in months."""

        total_months = 0

        for experience in experiences:
            start = FeatureExtractor._parse_date(
                experience.get("start_date")
            )

            end = FeatureExtractor._parse_date(
                experience.get("end_date")
            )

            if start is None or end is None:
                continue

            months = (
                (end[0] - start[0]) * 12
                + (end[1] - start[1])
            )

            if months > 0:
                total_months += months

        return total_months

    @staticmethod
    def _parse_date(
        value: str | None,
    ) -> tuple[int, int] | None:
        """Convert common resume dates into (year, month)."""

        if not value:
            return None

        value = value.strip()

        if value.lower() in {"present", "current"}:
            today = datetime.now()
            return today.year, today.month

        month_names = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }

        match = re.match(
            r"(?i)^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\s+(\d{4})$",
            value,
        )

        if match:
            month = month_names[match.group(1).lower()]
            year = int(match.group(2))

            return year, month

        match = re.match(r"^(\d{4})$", value)

        if match:
            return int(match.group(1)), 1

        return None

    @staticmethod
    def _build_skill_pattern(skill: str) -> str:
        """Build a safe regex pattern for skill matching."""

        escaped_skill = re.escape(skill)

        return rf"(?<![a-z0-9]){escaped_skill}(?![a-z0-9])"


feature_extractor = FeatureExtractor()