# Job description extraction.

from __future__ import annotations

import re

from app.models.schemas import EducationRequirement, JobDescription
from app.services.feature_extractor import feature_extractor


REQUIRED_SECTION_HEADERS = {
    "required skills",
    "required skill",
    "must have",
    "must-have skills",
    "mandatory skills",
    "requirements",
}

PREFERRED_SECTION_HEADERS = {
    "preferred skills",
    "preferred skill",
    "nice to have",
    "nice-to-have",
    "good to have",
    "desired skills",
    "bonus skills",
}

OTHER_SECTION_HEADERS = {
    "responsibilities",
    "education",
    "qualifications",
    "qualification",
    "about us",
    "about the role",
    "benefits",
    "experience",
    "work experience",
    "professional experience",
    "certifications",
    "certification",
    "projects",
    "skills",
    "technical skills",
}

# More specific patterns are checked before broader degree names.
DEGREE_KEYWORDS: dict[str, str] = {
    r"\bb\.?\s*tech\.?\b": "B.Tech",
    r"\bb\.?\s*e\.?\b": "B.E.",
    r"\bbachelor(?:'s)?(?:\s+degree)?\b": "Bachelor's Degree",
    r"\bbca\b": "BCA",
    r"\bb\.?\s*s\.?\b": "B.S.",
    r"\bm\.?\s*tech\.?\b": "M.Tech",
    r"\bm\.?\s*e\.?\b": "M.E.",
    r"\bmaster(?:'s)?(?:\s+degree)?\b": "Master's Degree",
    r"\bmca\b": "MCA",
    r"\bmba\b": "MBA",
    r"\bm\.?\s*s\.?\b": "M.S.",
    r"\bph\.?\s*d\.?\b": "Ph.D.",
    r"\bdoctor(?:ate)?\b": "Doctorate",
}


class JDParser:
    """
    Parse raw job-description text into a structured JobDescription.

    The parser is intentionally deterministic and dependency-light. It extracts:
      - title
      - required skills
      - preferred skills
      - minimum experience
      - education requirements

    Optional/missing sections never cause parsing to fail.
    """

    EXPERIENCE_PATTERN = re.compile(
    r"(?:minimum\s+(?:of\s+)?|at\s+least\s+|min\.?\s+)?"
    r"(?P<min_years>\d+(?:\.\d+)?)"
    r"(?:\s*[-–—]\s*(?P<max_years>\d+(?:\.\d+)?))?"
    r"\s*\+?\s*years?\s*(?:of\s+)?"
    r"(?:relevant\s+|professional\s+|hands[-\s]?on\s+)?"
    r"(?:[A-Za-z][A-Za-z/&,\- ]{0,80}?\s+)?"
    r"experience",
    re.IGNORECASE,
)

    FIELD_OF_STUDY_PATTERN = re.compile(
        r"\b(?:in|of)\s+"
        r"(?P<field>[A-Za-z][A-Za-z0-9&/.,'() +#-]*?)"
        r"(?=\s*(?:,|;|\.|\bor\b|\bpreferred\b|\brequired\b|"
        r"\bmandatory\b|\boptional\b|\bplus\b|$))",
        re.IGNORECASE,
    )

    RELATED_FIELD_SUFFIX = re.compile(
        r"\s+(?:or\s+)?(?:a\s+)?"
        r"(?:closely\s+)?related\s+fields?(?:\s+or\s+equivalent)?"
        r".*$",
        re.IGNORECASE,
    )

    EDUCATION_SECTION_ALIASES = {
        "education",
        "academic background",
        "educational qualifications",
        "educational qualification",
        "qualifications",
    }

    OPTIONAL_EDUCATION_MARKERS = (
        "preferred",
        "nice to have",
        "nice-to-have",
        "optional",
        "a plus",
        "plus",
        "desired",
        "preferred qualification",
        "preferred qualifications",
        "not required",
    )

    def parse(self, text: str) -> JobDescription:
        """Parse all supported JD fields safely."""

        if not text or not text.strip():
            return JobDescription(title="", raw_text=text or "")

        title = self.extract_title(text)
        required_skills, preferred_skills = (
            self.extract_required_and_preferred_skills(text)
        )
        min_experience_years = self.extract_min_experience(text)
        education_requirements = self.extract_education_requirements(text)

        return JobDescription(
            title=title,
            raw_text=text,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            min_experience_years=min_experience_years,
            education_requirements=education_requirements,
        )

    def extract_title(self, text: str) -> str:
        """Return the first meaningful line as the job title."""

        for raw_line in text.splitlines():
            line = self._clean_line(raw_line)
            if not line:
                continue

            line = re.sub(
                r"^(?:job\s+title|position|role|job)\s*[:\-–—]\s*",
                "",
                line,
                flags=re.IGNORECASE,
            ).strip()

            return line

        return ""

    def extract_required_and_preferred_skills(
        self,
        text: str,
    ) -> tuple[list[str], list[str]]:
        """
        Extract required/preferred skill sections.

        Supports:
            REQUIRED SKILLS
            Python, FastAPI, PostgreSQL

        and:
            REQUIRED SKILLS: Python, FastAPI, PostgreSQL

        The deterministic resume skill dictionary is reused so candidate/JD
        normalization stays consistent.
        """

        required_lines: list[str] = []
        preferred_lines: list[str] = []
        current_section: str | None = None

        for raw_line in text.splitlines():
            line = self._clean_line(raw_line)
            if not line:
                continue

            normalized = self._normalize_header(line)

            # Exact section header.
            if normalized in REQUIRED_SECTION_HEADERS:
                current_section = "required"
                continue

            if normalized in PREFERRED_SECTION_HEADERS:
                current_section = "preferred"
                continue

            if normalized in OTHER_SECTION_HEADERS:
                current_section = None
                continue

            # Header with inline content: "Required Skills: Python, FastAPI"
            header, separator, content = line.partition(":")
            normalized_header = self._normalize_header(header)

            if normalized_header in REQUIRED_SECTION_HEADERS:
                current_section = "required"
                if content.strip():
                    required_lines.append(content.strip())
                continue

            if normalized_header in PREFERRED_SECTION_HEADERS:
                current_section = "preferred"
                if content.strip():
                    preferred_lines.append(content.strip())
                continue

            if current_section == "required":
                required_lines.append(line)
            elif current_section == "preferred":
                preferred_lines.append(line)

        required_skills = feature_extractor.extract_skills(
            "\n".join(required_lines)
        )
        preferred_skills = feature_extractor.extract_skills(
            "\n".join(preferred_lines)
        )

        # If a skill is explicitly required, never count it as preferred.
        preferred_skills = [
            skill
            for skill in preferred_skills
            if skill not in required_skills
        ]

        return required_skills, preferred_skills

    def extract_min_experience(self, text: str) -> float | None:
        """
        Extract the lower bound from phrases such as:
            3+ years of experience
            3 years experience
            3-5 years of experience
            minimum of 3 years relevant experience
            at least 3 years of professional experience
        """

        if not text:
            return None

        matches = list(self.EXPERIENCE_PATTERN.finditer(text))
        if not matches:
            return None

        # Prefer the first explicit experience requirement.
        try:
            return float(matches[0].group("min_years"))
        except (TypeError, ValueError):
            return None

    def extract_education_requirements(
        self,
        text: str,
    ) -> list[EducationRequirement]:
        """
        Extract degree + field requirements.

        Important normalization:
            "Bachelor's Degree in Computer Science or a closely related field"
        becomes:
            degree = "Bachelor's Degree"
            field_of_study = "Computer Science"

        This prevents the matcher from treating the policy phrase
        "or a closely related field" as part of the actual field name.
        """

        requirements: list[EducationRequirement] = []
        seen: set[tuple[str, str | None, bool]] = set()

        for raw_line in text.splitlines():
            line = self._clean_line(raw_line)
            if not line:
                continue

            normalized = line.lower()

            degree_match = self._find_degree(line)
            if degree_match is None:
                continue

            canonical_degree, degree_start, degree_end = degree_match

            # Determine whether this is an optional/preferred qualification.
            required = not any(
                marker in normalized
                for marker in self.OPTIONAL_EDUCATION_MARKERS
            )

            field = self._extract_field_of_study(
                line,
                degree_end,
            )

            # Some JDs use "Bachelor's degree - Computer Science".
            if field is None:
                field = self._extract_field_after_degree(
                    line,
                    degree_end,
                )

            key = (canonical_degree, field, required)

            if key in seen:
                continue

            seen.add(key)

            requirements.append(
                EducationRequirement(
                    degree=canonical_degree,
                    field_of_study=field,
                    required=required,
                )
            )

        return requirements

    def _find_degree(
        self,
        line: str,
    ) -> tuple[str, int, int] | None:
        """Return canonical degree and its span."""

        matches: list[tuple[int, int, str]] = []

        for pattern, canonical_degree in DEGREE_KEYWORDS.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                matches.append(
                    (match.start(), match.end(), canonical_degree)
                )

        if not matches:
            return None

        # Prefer the earliest occurrence; if tied, prefer the longest span.
        matches.sort(
            key=lambda item: (
                item[0],
                -(item[1] - item[0]),
            )
        )

        start, end, canonical = matches[0]
        return canonical, start, end

    def _extract_field_of_study(
        self,
        line: str,
        degree_end: int,
    ) -> str | None:
        """Extract a clean field from an 'in/of <field>' expression."""

        # Start after the degree to avoid unrelated "in" phrases earlier in
        # the sentence.
        tail = line[degree_end:]

        match = self.FIELD_OF_STUDY_PATTERN.search(tail)
        if not match:
            return None

        field = match.group("field").strip(" ,;:-–—")

        # Remove policy phrases such as "or a closely related field".
        field = self.RELATED_FIELD_SUFFIX.sub("", field).strip(
            " ,;:-–—"
        )

        # Defensive cleanup for common trailing policy wording.
        field = re.sub(
            r"\s+or\s+(?:a\s+)?(?:closely\s+)?related\s+field.*$",
            "",
            field,
            flags=re.IGNORECASE,
        ).strip(" ,;:-–—")

        return field or None

    def _extract_field_after_degree(
        self,
        line: str,
        degree_end: int,
    ) -> str | None:
        """
        Handle:
            Bachelor's Degree - Computer Science
            Bachelor's Degree: Computer Science
            Bachelor's Degree, Computer Science
        """

        tail = line[degree_end:].strip()

        tail = re.sub(
            r"^[\s:,\-–—]+",
            "",
            tail,
        ).strip()

        if not tail:
            return None

        # Remove trailing requirement/policy text.
        tail = re.split(
            r"\b(?:required|preferred|mandatory|optional|desired)\b",
            tail,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" ,;:-–—")

        tail = self.RELATED_FIELD_SUFFIX.sub("", tail).strip(
            " ,;:-–—"
        )

        if not tail:
            return None

        # Don't accidentally treat a generic sentence as a field.
        if len(tail.split()) > 8:
            return None

        return tail

    @staticmethod
    def _clean_line(line: str) -> str:
        """Normalize whitespace and common bullet prefixes."""

        if not line:
            return ""

        line = line.replace("\u00a0", " ")
        line = re.sub(r"^\s*[-•▪◦*]+\s*", "", line)
        return re.sub(r"\s+", " ", line).strip()

    @staticmethod
    def _normalize_header(line: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            line.lower().rstrip(":").strip(),
        )


jd_parser = JDParser()
