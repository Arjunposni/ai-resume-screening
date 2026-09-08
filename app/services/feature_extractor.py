# Candidate/JD feature extraction.

import re
from datetime import datetime

from app.models.schemas import CandidateProfile


# Skill dictionary. Kept as a set for O(1) membership; iteration order
# doesn't matter since extract_skills() sorts the result before returning.
SKILL_DICTIONARY = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#",
    "go", "rust", "kotlin", "swift",

    # Backend / APIs
    "fastapi", "flask", "django", "spring", "spring boot", "rest api", "graphql",

    # Frontend
    "react", "angular", "vue", "html", "css", "tailwind",

    # Data / AI
    "machine learning", "deep learning", "natural language processing", "nlp",
    "generative ai", "llm", "rag", "retrieval augmented generation",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "opencv",

    # Databases
    "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "oracle",

    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "gitlab",
    "linux", "ci/cd", "jenkins",

    # Data engineering
    "apache spark", "hadoop", "kafka", "airflow",

    # Other
    "microservices", "system design", "data structures", "algorithms",
}

# Punctuation characters that appear *inside* skill names (e.g. "c++",
# "c#", "ci/cd", "scikit-learn"). These are treated as "word" characters
# in the boundary lookarounds, otherwise a short skill like "c" would
# incorrectly match inside "c++" (since "+" is not alphanumeric and would
# otherwise satisfy a naive word boundary).
_SKILL_BOUNDARY_CHARS = "".join(
    sorted({ch for skill in SKILL_DICTIONARY for ch in skill if not ch.isalnum() and not ch.isspace()})
)
_SKILL_NONBOUNDARY_CLASS = re.escape(_SKILL_BOUNDARY_CHARS)

# Canonical dash class: plain hyphen, en dash, em dash.
DASH_CLASS = r"[-–—]"
# Dash class WITHOUT the plain hyphen — used where splitting on a bare "-"
# would be wrong because hyphens legitimately occur inside words
# (e.g. "Full-Stack Developer", "Ph.D.-level").
DASH_CLASS_NO_HYPHEN = r"[—–]"


class FeatureExtractor:
    """Extract structured, job-relevant features from resume text."""

    EXPERIENCE_DATE_PATTERN = re.compile(
        r"(?P<start>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}|\d{4})"
        rf"\s*(?:{DASH_CLASS}|to)\s*"
        r"(?P<end>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}|\d{4}|Present|Current)"
        r"\s*$",
        re.IGNORECASE,
    )

    EDUCATION_PATTERN = re.compile(
        r"^(?P<degree>.+?)"
        r"\s*\|\s*"
        r"(?P<institution>.+?)"
        rf"(?:\s*\|\s*(?P<start>19\d{{2}}|20\d{{2}})"
        rf"\s*(?:{DASH_CLASS}|to)\s*"
        r"(?P<end>19\d{2}|20\d{2}))?$",
        re.IGNORECASE,
    )

    CERTIFICATION_PATTERN = re.compile(
        rf"^(?P<name>.+?)\s*(?:\||{DASH_CLASS})\s*(?P<year>20\d{{2}})$",
        re.IGNORECASE,
    )

    PROJECT_PATTERN = re.compile(
        r"^(?P<name>[^|]+?)\s*\|\s*(?P<technologies>.+)$",
        re.IGNORECASE,
    )

    SECTION_HEADERS = {
        "experience": {
            "experience", "work experience", "professional experience",
            "employment experience", "employment history",
        },
        "education": {"education", "academic background"},
        "certifications": {
            "certifications", "certification", "certificates",
            "licenses", "licenses and certifications",
        },
        "projects": {
            "projects", "academic projects", "personal projects",
            "technical projects",
        },
        "skills": {"skills", "technical skills"},
        "summary": {"professional summary", "summary", "objective"},
    }

    ALL_HEADERS = set().union(*SECTION_HEADERS.values())
    CERTIFICATION_KEYWORDS = ("certified", "certification", "certificate")

    DURATION_PATTERN = re.compile(
        r"(?P<years>\d+(?:\.\d+)?)\s*\+?\s*years?(?:\s+of)?\s+experience",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> CandidateProfile:
        if not text or not text.strip():
            return CandidateProfile()

        experiences = self.extract_experience(text)

        return CandidateProfile(
            skills=self.extract_skills(text),
            experience=experiences,
            education=self.extract_education(text),
            certifications=self.extract_certifications(text),
            projects=self.extract_projects(text),
            total_experience_months=self.calculate_total_experience(experiences),
        )

    def extract_skills(self, text: str) -> list[str]:
        normalized_text = text.lower()
        detected = []

        for skill in SKILL_DICTIONARY:
            if re.search(self._build_skill_pattern(skill), normalized_text):
                detected.append(skill)

        return sorted(set(detected))

    def extract_experience(self, text: str) -> list[dict]:
        """
        Extract work experience entries from a resume.

        Supports both date-range ("Jan 2021 - Dec 2024") and explicit
        duration ("3.2 years of experience") formats.
        """
        experiences = []
        in_section = False

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            normalized = line.lower().rstrip(":")

            if normalized in self.SECTION_HEADERS["experience"]:
                in_section = True
                continue

            if normalized in self.ALL_HEADERS:
                in_section = False
                continue

            if not in_section:
                continue

            # Explicit duration format:
            # Backend Engineer — Company — 3.2 years of experience.
            duration_match = self.DURATION_PATTERN.search(line)

            if duration_match:
                years = float(duration_match.group("years"))

                before_duration = line[: duration_match.start()].strip()
                before_duration = before_duration.rstrip("-|:").strip()

                parts = self._split_header(before_duration)

                if len(parts) >= 2:
                    experiences.append({
                        "job_title": parts[0],
                        "company": parts[1],
                        "location": " | ".join(parts[2:]) if len(parts) > 2 else None,
                        "start_date": None,
                        "end_date": None,
                        "duration_years": years,
                        "description": [],
                    })
                    continue

            # Date-range format:
            # Backend Engineer — Company — Jan 2021 - Dec 2024
            match = self.EXPERIENCE_DATE_PATTERN.search(line)

            if not match:
                continue

            header = re.sub(rf"[\s|]+$|\s*{DASH_CLASS}\s*$", "", line[: match.start()].strip())
            parts = self._split_header(header)

            if len(parts) < 2:
                continue

            experiences.append({
                "job_title": parts[0],
                "company": parts[1],
                "location": " | ".join(parts[2:]) if len(parts) > 2 else None,
                "start_date": match.group("start").strip(),
                "end_date": match.group("end").strip(),
                "duration_years": None,
                "description": [],
            })

        return experiences

    def extract_education(self, text: str) -> list[dict]:
        """
        Extract education records from common resume layouts.

        Supported examples include:
            B.S. Computer Science | University of Texas | 2019 - 2023
            Bachelor's Degree in Computer Science | University | 2019 - 2023
            Bachelor of Science in Computer Science | University | 2019 - 2023
            B.Tech - Computer Science, ABC University, 2020 - 2024
            B.Tech Computer Science — ABC University — 2020-2024
            Bachelor of Science, Computer Science
            University of Texas | 2019 - 2023
        """

        education: list[dict] = []
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # First pass: parse self-contained education lines.
        for index, line in enumerate(lines):
            if self._is_section_header(line):
                continue

            record = self._parse_education_line(line)

            if record is not None:
                education.append(record)

                if index + 1 < len(lines):
                    next_line = lines[index + 1]
                    self._enrich_education_record(record, next_line)

        # Second pass: parse multi-line education entries:
        #   Bachelor of Science in Computer Science
        #   University of Texas
        #   2019 - 2023
        for index, line in enumerate(lines):
            if not self._looks_like_degree(line):
                continue

            # Pipe-delimited lines are fully handled by the structured
            # (single-line) pass above — skip them here to avoid emitting
            # a duplicate/garbled record from the raw, unparsed line.
            if "|" in line:
                continue

            degree, field = self._split_degree_and_field(line)
            if not degree:
                continue

            existing = next(
                (
                    item for item in education
                    if self._normalize_compare(item.get("degree")) == self._normalize_compare(degree)
                    and self._normalize_compare(item.get("field_of_study")) == self._normalize_compare(field)
                ),
                None,
            )

            if existing is not None:
                continue

            institution = None
            start_year = None
            end_year = None

            for next_index in range(index + 1, min(index + 4, len(lines))):
                candidate_line = lines[next_index]

                year_range = self._extract_year_range(candidate_line)
                if year_range:
                    start_year, end_year = year_range
                    continue

                if (
                    institution is None
                    and not self._looks_like_degree(candidate_line)
                    and not self._is_section_header(candidate_line)
                    and not self._looks_like_resume_noise(candidate_line)
                ):
                    institution = self._clean_institution(candidate_line)

            education.append({
                "degree": degree,
                "field_of_study": field,
                "institution": institution,
                "start_year": start_year,
                "end_year": end_year,
            })

        # Remove duplicates while preserving order.
        unique: list[dict] = []
        seen: set[tuple] = set()

        for item in education:
            key = (
                self._normalize_compare(item.get("degree")),
                self._normalize_compare(item.get("field_of_study")),
                self._normalize_compare(item.get("institution")),
                item.get("start_year"),
                item.get("end_year"),
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(item)

        return unique

    def _parse_education_line(self, line: str) -> dict | None:
        """Parse a complete education entry appearing on one line."""

        if not self._looks_like_degree(line):
            return None

        normalized = re.sub(r"\s+", " ", line).strip()

        year_range = self._extract_year_range(normalized)
        if year_range:
            normalized_without_dates = re.sub(
                rf"(?:\b(?:19|20)\d{{2}}\b)\s*(?:{DASH_CLASS}|to)\s*(?:\b(?:19|20)\d{{2}}\b)",
                "",
                normalized,
                flags=re.IGNORECASE,
            ).strip(" |,-–—")
        else:
            normalized_without_dates = normalized

        # Structured pipe format: Degree/field | Institution | dates
        parts = [part.strip() for part in normalized_without_dates.split("|") if part.strip()]

        degree_text = parts[0] if parts else normalized_without_dates
        institution = None

        if len(parts) >= 2:
            institution = self._clean_institution(parts[1])
        else:
            degree_text, institution = self._split_institution_from_degree(degree_text)

        degree, field = self._split_degree_and_field(degree_text)

        if not field:
            field_match = re.search(
                r"(?:major|field(?:\s+of\s+study)?|speciali[sz]ation)\s*[:\-]\s*(.+?)(?:\s*\||$)",
                normalized,
                re.IGNORECASE,
            )
            if field_match:
                field = field_match.group(1).strip(" ,.-–—")

        if not degree or not self._looks_like_degree_level(degree):
            return None

        start_year, end_year = year_range or (None, None)

        return {
            "degree": degree,
            "field_of_study": field,
            "institution": institution,
            "start_year": start_year,
            "end_year": end_year,
        }

    def _enrich_education_record(self, record: dict, next_line: str) -> None:
        """Fill missing institution/date fields from an adjacent line."""

        if record.get("institution") is None:
            candidate = next_line.strip()
            if not self._looks_like_degree(candidate) and not self._is_section_header(candidate):
                candidate_range = self._extract_year_range(candidate)
                if candidate_range is None:
                    cleaned = self._clean_institution(candidate)
                    if cleaned:
                        record["institution"] = cleaned

        if record.get("start_year") is None:
            year_range = self._extract_year_range(next_line)
            if year_range:
                record["start_year"], record["end_year"] = year_range

    @staticmethod
    def _split_degree_and_field(degree_text: str) -> tuple[str, str | None]:
        """Split degree level from major/field of study."""

        text = re.sub(r"\s+", " ", degree_text).strip(" |,-–—")

        patterns = [
            r"^(Bachelor\s+of\s+Science|Bachelor\s+of\s+Arts|Bachelor\s+of\s+Engineering|"
            r"Bachelor\s+of\s+Technology|Bachelor\s+of\s+Computer\s+Science|"
            r"Master\s+of\s+Science|Master\s+of\s+Arts|Master\s+of\s+Engineering|"
            r"Master\s+of\s+Technology)\s+(?:in|of)\s+(.+)$",

            r"^(Bachelor(?:'s)?(?:\s+Degree)?|B\.?\s*S\.?|B\.?\s*Tech\.?|"
            r"B\.?\s*E\.?|BCA|MCA|MBA|Master(?:'s)?(?:\s+Degree)?|"
            r"M\.?\s*S\.?|M\.?\s*Tech\.?|M\.?\s*E\.?|"
            r"Ph\.?\s*D\.?|Doctor(?:ate)?)\s+(?:in|of)\s+(.+)$",

            r"^(Bachelor\s+of\s+(?:Science|Arts|Engineering|Technology)|"
            r"Master\s+of\s+(?:Science|Arts|Engineering|Technology))\s+(.+)$",

            r"^(B\.?\s*S\.?|B\.?\s*Tech\.?|B\.?\s*E\.?|BCA|MCA|MBA|"
            r"M\.?\s*S\.?|M\.?\s*Tech\.?|M\.?\s*E\.?|"
            r"Ph\.?\s*D\.?|Doctor(?:ate)?)\s*[-,:]\s*(.+)$",

            r"^(Bachelor(?:'s)?(?:\s+Degree)?|Master(?:'s)?(?:\s+Degree)?)\s+(.+)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                degree = re.sub(r"\s+", " ", match.group(1)).strip()
                field = re.sub(r"\s+", " ", match.group(2)).strip(" ,.-–—")
                return degree, field or None

        bare_patterns = [
            r"^Bachelor(?:'s)?(?:\s+Degree)?$",
            r"^B\.?\s*S\.?$",
            r"^B\.?\s*Tech\.?$",
            r"^B\.?\s*E\.?$",
            r"^BCA$",
            r"^Master(?:'s)?(?:\s+Degree)?$",
            r"^M\.?\s*S\.?$",
            r"^M\.?\s*Tech\.?$",
            r"^M\.?\s*E\.?$",
            r"^MCA$",
            r"^MBA$",
            r"^Ph\.?\s*D\.?$",
            r"^Doctor(?:ate)?$",
        ]

        for pattern in bare_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return text, None

        return text, None

    @staticmethod
    def _looks_like_degree(text: str) -> bool:
        if not text:
            return False

        return bool(
            re.search(
                r"\b(?:B\.?\s*S\.?|B\.?\s*Tech\.?|B\.?\s*E\.?|BCA|"
                r"M\.?\s*S\.?|M\.?\s*Tech\.?|M\.?\s*E\.?|MCA|MBA|"
                r"Ph\.?\s*D\.?|Bachelor(?:'s)?|Master(?:'s)?|Doctor(?:ate)?|"
                r"Bachelor\s+of|Master\s+of)\b",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _looks_like_degree_level(text: str) -> bool:
        return FeatureExtractor._looks_like_degree(text)

    @staticmethod
    def _extract_year_range(text: str) -> tuple[int, int] | None:
        if not text:
            return None

        match = re.search(
            rf"\b((?:19|20)\d{{2}})\s*(?:{DASH_CLASS}|to)\s*((?:19|20)\d{{2}})\b",
            text,
            re.IGNORECASE,
        )

        if not match:
            return None

        start = int(match.group(1))
        end = int(match.group(2))

        if start > end or end > datetime.now().year + 10:
            return None

        return start, end

    @staticmethod
    def _split_institution_from_degree(text: str) -> tuple[str, str | None]:
        """
        Handle forms such as:
            B.Tech Computer Science, ABC University
            B.S. Computer Science — University of Texas
        """

        match = re.match(rf"^(?P<degree>.+?)\s+{DASH_CLASS_NO_HYPHEN}\s+(?P<institution>.+)$", text)
        if match and not FeatureExtractor._looks_like_degree(match.group("institution")):
            return match.group("degree").strip(), match.group("institution").strip()

        parts = [part.strip() for part in text.split(",", maxsplit=1)]
        if len(parts) == 2:
            left, right = parts
            if right and not FeatureExtractor._looks_like_degree(right):
                return left, right

        return text, None

    @staticmethod
    def _clean_institution(value: str | None) -> str | None:
        if not value:
            return None

        cleaned = re.sub(r"\s+", " ", value).strip(" |,-–—")
        if not cleaned:
            return None

        if re.fullmatch(rf"(?:19|20)\d{{2}}(?:\s*{DASH_CLASS}\s*(?:19|20)\d{{2}})?", cleaned):
            return None

        return cleaned

    @staticmethod
    def _normalize_compare(value: object) -> str:
        if value is None:
            return ""
        return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

    def _is_section_header(self, line: str) -> bool:
        normalized = line.lower().rstrip(":").strip()
        return normalized in self.ALL_HEADERS

    @staticmethod
    def _looks_like_resume_noise(line: str) -> bool:
        normalized = line.lower()

        noise_terms = (
            "skills", "experience", "certification", "project",
            "summary", "objective", "responsibilities", "references",
        )

        return any(term == normalized.rstrip(":") for term in noise_terms)

    def extract_certifications(self, text: str) -> list[dict]:
        certifications = []
        in_section = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            normalized = line.lower().rstrip(":")

            if normalized in self.SECTION_HEADERS["certifications"]:
                in_section = True
                continue

            if normalized in self.ALL_HEADERS:
                in_section = False
                continue

            if not in_section:
                continue

            structured = self.CERTIFICATION_PATTERN.match(line)
            if structured:
                certifications.append({
                    "name": structured.group("name").strip(),
                    "year": int(structured.group("year")),
                })
                continue

            year_match = re.search(r"\b(20\d{2})\b", line)
            has_keyword = any(keyword in line.lower() for keyword in self.CERTIFICATION_KEYWORDS)

            if not has_keyword and not year_match:
                continue

            if year_match:
                year = int(year_match.group(1))
                name = line[: year_match.start()] + line[year_match.end():]
                name = re.sub(r"^[\s|()\-–—]+|[\s|()\-–—]+$", "", name).strip()
            else:
                year = None
                name = line

            certifications.append({"name": name or line, "year": year})

        return certifications

    def extract_projects(self, text: str) -> list[dict]:
        projects = []
        in_section = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            normalized = line.lower().rstrip(":")

            if normalized in self.SECTION_HEADERS["projects"]:
                in_section = True
                continue

            if normalized in self.ALL_HEADERS:
                in_section = False
                continue

            if not in_section:
                continue

            match = self.PROJECT_PATTERN.match(line)
            if not match:
                continue

            technologies = [
                item.strip().lower()
                for item in match.group("technologies").split(",")
                if item.strip()
            ]

            projects.append({
                "name": match.group("name").strip(),
                "description": None,
                "technologies": technologies,
            })

        return projects

    def calculate_total_experience(self, experiences: list[dict]) -> int:
        """
        Calculate total professional experience, in months.

        - Entries with an explicit duration ("3.2 years of experience")
          contribute duration_years * 12 months directly.
        - Date-range entries are converted to (start, end) month indices,
          overlapping ranges are merged so concurrent roles aren't
          double-counted, and the merged spans are summed.
        """

        total_months = 0.0
        intervals: list[tuple[int, int]] = []

        for experience in experiences:
            duration = experience.get("duration_years")

            if duration is not None:
                try:
                    total_months += float(duration) * 12
                except (TypeError, ValueError):
                    pass
                continue

            start = self._parse_date(experience.get("start_date"))
            end = self._parse_date(experience.get("end_date"))

            if start is None or end is None:
                continue

            start_index = start[0] * 12 + start[1]
            end_index = end[0] * 12 + end[1]

            if end_index > start_index:
                intervals.append((start_index, end_index))

        if intervals:
            intervals.sort()
            merged = [intervals[0]]

            for current_start, current_end in intervals[1:]:
                last_start, last_end = merged[-1]

                if current_start <= last_end:
                    merged[-1] = (last_start, max(last_end, current_end))
                else:
                    merged.append((current_start, current_end))

            total_months += sum(end - start for start, end in merged)

        return round(total_months)

    @staticmethod
    def _parse_date(value: str | None) -> tuple[int, int] | None:
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
            return int(match.group(2)), month_names[match.group(1).lower()]

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

        Only an em/en dash (never a plain hyphen) is treated as the
        title/company separator, since plain hyphens legitimately occur
        inside words (e.g. "Full-Stack Developer"). When a dash is
        present, it splits off the title first and then splits the
        remainder on pipes, so mixed-delimiter headers don't fold the
        company into the title.
        """

        dash_split = re.split(rf"\s+{DASH_CLASS_NO_HYPHEN}\s+", header, maxsplit=1)

        if len(dash_split) == 2:
            title, rest = dash_split
            parts = [title] + rest.split("|")
        else:
            parts = header.split("|")

        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _build_skill_pattern(skill: str) -> str:
        escaped_skill = re.escape(skill)
        boundary_class = f"[a-z0-9{_SKILL_NONBOUNDARY_CLASS}]"
        return rf"(?<!{boundary_class}){escaped_skill}(?!{boundary_class})"


feature_extractor = FeatureExtractor()