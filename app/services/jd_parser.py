# Job description extraction.

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

# Headers that end whichever skills section we were in, without
# starting a new one (so we stop collecting lines as skills).
OTHER_SECTION_HEADERS = {
    "responsibilities",
    "education",
    "qualifications",
    "about us",
    "about the role",
    "benefits",
}

# Ordered so more specific abbreviations are checked before broader
# ones; each maps a regex fragment (searched case-insensitively) to
# the canonical degree label we store.
DEGREE_KEYWORDS: dict[str, str] = {
    r"b\.?\s?tech\.?": "B.Tech",
    r"b\.?\s?e\.?\b": "B.E.",
    r"bachelor'?s?(?:\s+degree)?": "Bachelor's Degree",
    r"bca": "BCA",
    r"b\.?\s?s\.?\b": "B.S.",
    r"m\.?\s?tech\.?": "M.Tech",
    r"master'?s?(?:\s+degree)?": "Master's Degree",
    r"mca": "MCA",
    r"mba": "MBA",
    r"m\.?\s?s\.?\b": "M.S.",
    r"ph\.?\s?d\.?": "Ph.D.",
}


class JDParser:
    """
    Parse raw job description text into a structured JobDescription.

    Built incrementally (Phase 4): this step only extracts the title.
    Skill, experience, and education extraction are added in the
    following milestones (M4.3-M4.5) so each piece can be tested in
    isolation before being combined into the full parse.
    """

    # Matches phrasing like "2+ years experience", "3-5 years of
    # experience", "at least 2 years experience", "minimum of 4 years
    # of relevant experience". The lower bound is what we keep.
    EXPERIENCE_PATTERN = re.compile(
        r"(?:minimum\s+(?:of\s+)?|at\s+least\s+|min\.?\s+)?"
        r"(?P<min_years>\d+(?:\.\d+)?)"
        r"(?:\s*[-\u2013\u2014]\s*(?P<max_years>\d+(?:\.\d+)?))?"
        r"\+?\s*years?\s*(?:of\s+)?"
        r"(?:relevant\s+|professional\s+|hands-on\s+)?experience",
        re.IGNORECASE,
    )

    # Matches "in <field>", used to pull a field of study out of a
    # degree mention like "Bachelor's degree in Computer Science".
    FIELD_OF_STUDY_PATTERN = re.compile(
        r"\bin\s+(?P<field>[A-Za-z][A-Za-z\s]*?)"
        r"(?=\s*(?:,|\.|;|\bor\b|\bpreferred\b|\brequired\b|$))",
        re.IGNORECASE,
    )

    def parse(self, text: str) -> JobDescription:
        """Extract everything currently implemented into a JobDescription."""

        if not text or not text.strip():
            return JobDescription(title="", raw_text=text or "")

        title = self.extract_title(text)
        required_skills, preferred_skills = self.extract_required_and_preferred_skills(text)
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
        """
        Use the first non-empty line as the job title.

        Job descriptions conventionally lead with the role title, e.g.:
            Backend Engineer
        or:
            Job Title: Backend Engineer

        A leading "Job Title:" / "Position:" / "Role:" label is stripped
        if present.
        """

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            line = re.sub(
                r"^(?:job\s*title|position|role)\s*[:\-]\s*",
                "",
                line,
                flags=re.IGNORECASE,
            )

            return line

        return ""

    def extract_required_and_preferred_skills(
        self, text: str
    ) -> tuple[list[str], list[str]]:
        """
        Split the JD into required-skill and preferred-skill segments
        based on section headers (either "Header: inline, list" or a
        multi-line section under a header), then run the same
        deterministic skill matching used for resumes
        (feature_extractor.extract_skills) on each segment separately.

        Keeping required and preferred as two separately-matched
        segments (rather than one skill list with a flag) means the
        M4.8 hybrid scorer can weight them independently without any
        ambiguity about which bucket a skill belongs to.
        """

        required_lines: list[str] = []
        preferred_lines: list[str] = []
        current_section: str | None = None

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            header, _, inline_content = line.partition(":")
            normalized_header = header.strip().lower()

            if normalized_header in REQUIRED_SECTION_HEADERS:
                current_section = "required"
                if inline_content.strip():
                    required_lines.append(inline_content.strip())
                continue

            if normalized_header in PREFERRED_SECTION_HEADERS:
                current_section = "preferred"
                if inline_content.strip():
                    preferred_lines.append(inline_content.strip())
                continue

            if normalized_header in OTHER_SECTION_HEADERS:
                current_section = None
                continue

            if current_section == "required":
                required_lines.append(line)
            elif current_section == "preferred":
                preferred_lines.append(line)

        required_skills = feature_extractor.extract_skills(" ".join(required_lines))
        preferred_skills = feature_extractor.extract_skills(" ".join(preferred_lines))

        # A skill explicitly required shouldn't also be counted as merely
        # preferred, even if it's mentioned again in the preferred section.
        preferred_skills = [
            skill for skill in preferred_skills if skill not in required_skills
        ]

        return required_skills, preferred_skills

    def extract_min_experience(self, text: str) -> float | None:
        """
        Find the minimum years of experience required, e.g.
        "2+ years experience" or "3-5 years of experience" (the lower
        bound is used). Returns the first match found anywhere in the
        text, or None if no such phrase is present.
        """

        match = self.EXPERIENCE_PATTERN.search(text)

        if not match:
            return None

        return float(match.group("min_years"))

    def extract_education_requirements(self, text: str) -> list[EducationRequirement]:
        """
        Detect degree requirements mentioned in the JD, e.g. "Bachelor's
        degree in Computer Science required" or "Master's degree
        preferred". A line containing "preferred" / "nice to have" /
        "optional" / "a plus" / "not required" marks that requirement
        as not required.
        """

        requirements: list[EducationRequirement] = []
        seen: set[tuple[str, str | None]] = set()

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            normalized = line.lower()

            for pattern, canonical_degree in DEGREE_KEYWORDS.items():
                if not re.search(pattern, normalized):
                    continue

                field = self._extract_field_of_study(line)
                required = not any(
                    marker in normalized
                    for marker in (
                        "preferred",
                        "nice to have",
                        "optional",
                        "a plus",
                        "not required",
                    )
                )

                key = (canonical_degree, field)

                if key not in seen:
                    seen.add(key)
                    requirements.append(
                        EducationRequirement(
                            degree=canonical_degree,
                            field_of_study=field,
                            required=required,
                        )
                    )

                break

        return requirements

    def _extract_field_of_study(self, line: str) -> str | None:
        """Pull "<field>" out of a "... in <field> ..." phrase, if present."""

        match = self.FIELD_OF_STUDY_PATTERN.search(line)

        if not match:
            return None

        field = match.group("field").strip()

        return field or None


jd_parser = JDParser()