# Candidate/JD feature extraction.

import re
from datetime import datetime

from app.models.schemas import CandidateProfile


# Skill dictionary. Kept as a set for O(1) membership; iteration order
# doesn't matter since extract_skills() sorts the result before returning.
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

# Punctuation characters that appear *inside* skill names (e.g. "c++",
# "c#", "ci/cd", "scikit-learn"). These need to be treated as "word"
# characters when building boundary lookarounds, otherwise a short skill
# like "c" will incorrectly match inside "c++" (since "+" is not
# alphanumeric and would otherwise satisfy a naive word boundary).
_SKILL_BOUNDARY_CHARS = "".join(
    sorted({ch for skill in SKILL_DICTIONARY for ch in skill if not ch.isalnum() and not ch.isspace()})
)
_SKILL_NONBOUNDARY_CLASS = re.escape(_SKILL_BOUNDARY_CHARS)


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
        r"\s+\d{4}|\d{4}|Present|Current)"
        r"\s*$",
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

    # "AWS Certified Solutions Architect | 2023" or "... - 2023"
    CERTIFICATION_PATTERN = re.compile(
        r"^(?P<name>.+?)"
        r"\s*(?:\||[-–—])\s*"
        r"(?P<year>20\d{2})$",
        re.IGNORECASE,
    )

    # "Project Name | Python, FastAPI, Docker"
    PROJECT_PATTERN = re.compile(
        r"^(?P<name>[^|]+?)"
        r"\s*\|\s*"
        r"(?P<technologies>.+)$",
        re.IGNORECASE,
    )

    SECTION_HEADERS = {
        "experience": {
            "experience",
            "work experience",
            "professional experience",
            "employment experience",
            "employment history",
        },
        "education": {"education"},
        "certifications": {
            "certifications",
            "certification",
            "certificates",
            "licenses",
            "licenses and certifications",
        },
        "projects": {
            "projects",
            "academic projects",
            "personal projects",
            "technical projects",
        },
        "skills": {"skills"},
        "summary": {"professional summary", "summary", "objective"},
    }

    # Any of these headers ends whichever section we're currently in.
    ALL_HEADERS = set().union(*SECTION_HEADERS.values())

    CERTIFICATION_KEYWORDS = ("certified", "certification", "certificate")

    def extract(self, text: str) -> CandidateProfile:
        """Extract all structured candidate features."""

        if not text or not text.strip():
            return CandidateProfile()

        skills = self.extract_skills(text)
        experience = self.extract_experience(text)
        education = self.extract_education(text)
        certifications = self.extract_certifications(text)
        projects = self.extract_projects(text)

        total_experience_months = self.calculate_total_experience(experience)

        return CandidateProfile(
            skills=skills,
            experience=experience,
            education=education,
            certifications=certifications,
            projects=projects,
            total_experience_months=total_experience_months,
        )

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
        in_experience_section = False

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            normalized = line.lower().rstrip(":")

            if normalized in self.SECTION_HEADERS["experience"]:
                in_experience_section = True
                continue

            if normalized in self.ALL_HEADERS:
                in_experience_section = False
                continue

            if not in_experience_section:
                continue

            date_match = self.EXPERIENCE_DATE_PATTERN.search(line)

            if not date_match:
                continue

            start_date = date_match.group("start").strip()
            end_date = date_match.group("end").strip()

            header = line[: date_match.start()].strip()
            header = re.sub(r"[\s|]+$", "", header)

            parts = self._split_header(header)

            if len(parts) < 2:
                continue

            job_title = parts[0]
            company = parts[1]
            location = " | ".join(parts[2:]) if len(parts) > 2 else None

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
        in_projects_section = False

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            normalized = line.lower().rstrip(":")

            if normalized in self.SECTION_HEADERS["projects"]:
                in_projects_section = True
                continue

            if normalized in self.ALL_HEADERS:
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

    def extract_certifications(self, text: str) -> list[dict]:
        """Extract certification records from the Certifications section.

        Falls back to scanning for certification keywords only while inside
        that section, so a bullet like "Led a certified process..." under
        Experience is never mistaken for a certification entry.
        """

        certifications = []
        in_certifications_section = False

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            normalized = line.lower().rstrip(":")

            if normalized in self.SECTION_HEADERS["certifications"]:
                in_certifications_section = True
                continue

            if normalized in self.ALL_HEADERS:
                in_certifications_section = False
                continue

            if not in_certifications_section:
                continue

            # Preferred structured form: "Name | 2023" or "Name - 2023".
            structured_match = self.CERTIFICATION_PATTERN.match(line)

            if structured_match:
                certifications.append(
                    {
                        "name": structured_match.group("name").strip(),
                        "year": int(structured_match.group("year")),
                    }
                )
                continue

            # Unstructured line: only accept if it actually reads like a
            # certification (avoids grabbing stray notes in the section).
            has_keyword = any(
                keyword in line.lower() for keyword in self.CERTIFICATION_KEYWORDS
            )
            year_match = re.search(r"\b(20\d{2})\b", line)

            if not has_keyword and not year_match:
                continue

            if year_match:
                year = int(year_match.group(1))
                name = line[: year_match.start()] + line[year_match.end() :]
                name = re.sub(r"^[\s|()\-–—]+|[\s|()\-–—]+$", "", name)
                name = re.sub(r"\s{2,}", " ", name).strip()
            else:
                year = None
                name = line

            certifications.append({"name": name or line, "year": year})

        return certifications

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
        """Calculate approximate total professional experience in months.

        Overlapping date ranges are merged before summing, so concurrent
        roles (e.g. a part-time job during a full-time one) aren't
        double-counted.
        """

        intervals = []

        for experience in experiences:
            start = FeatureExtractor._parse_date(experience.get("start_date"))
            end = FeatureExtractor._parse_date(experience.get("end_date"))

            if start is None or end is None:
                continue

            start_index = start[0] * 12 + start[1]
            end_index = end[0] * 12 + end[1]

            if end_index > start_index:
                intervals.append((start_index, end_index))

        if not intervals:
            return 0

        intervals.sort()
        merged = [intervals[0]]

        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged[-1]

            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        return sum(end - start for start, end in merged)

    @staticmethod
    def _parse_date(value: str | None) -> tuple[int, int] | None:
        """Convert common resume dates into (year, month)."""

        if not value:
            return None

        value = value.strip()

        if value.lower() in {"present", "current"}:
            today = datetime.now()
            return today.year, today.month

        month_names = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }

        match = re.match(
            r"(?i)^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})$",
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
    def _split_header(header: str) -> list[str]:
        """Split a resume line header into [title, company, location...].

        Headers show up in two shapes:
          "Title — Company | Location"   (em/en dash + pipes)
          "Title | Company | Location"   (pipes only)

        A plain hyphen split is unsafe because hyphens routinely appear
        inside words (e.g. "Full-Stack Developer"), so only an em/en dash
        (surrounded by whitespace) is treated as the title/company
        separator. When a dash is present, it splits off the title first
        and then splits the remainder on pipes, so mixed-delimiter headers
        don't get the company folded into the title.
        """

        dash_split = re.split(r"\s+[—–]\s+", header, maxsplit=1)

        if len(dash_split) == 2:
            title, rest = dash_split
            parts = [title] + rest.split("|")
        else:
            parts = header.split("|")

        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _build_skill_pattern(skill: str) -> str:
        """Build a safe regex pattern for skill matching.

        Boundaries exclude alphanumerics *and* any punctuation that occurs
        inside a dictionary skill (e.g. '+', '#', '/', '-'). Without this,
        a short skill like "c" would match inside "c++" because '+' is not
        alphanumeric and would otherwise satisfy a naive word boundary.
        """

        escaped_skill = re.escape(skill)
        boundary_class = f"[a-z0-9{_SKILL_NONBOUNDARY_CLASS}]"

        return rf"(?<!{boundary_class}){escaped_skill}(?!{boundary_class})"


feature_extractor = FeatureExtractor()