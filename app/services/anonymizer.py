import re
from dataclasses import dataclass

import spacy


@dataclass
class DetectedEntity:
    label: str
    original: str
    start: int
    end: int
    priority: int


class ResumeAnonymizer:
    """
    Hybrid PII anonymizer.

    Uses deterministic regex/rules as the primary detector and
    spaCy NER as a secondary contextual detector.

    Conservative by design: job-relevant information such as
    skills, companies, job titles, and employment dates should
    remain available to downstream screening.
    """

    def __init__(self) -> None:
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed."
            ) from exc

    def detect(self, text: str) -> list[DetectedEntity]:
        entities: list[DetectedEntity] = []
        # ---------------------------------------------------------
        # 0. Resume-header name detection
        # ---------------------------------------------------------

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if lines:
            first_line = lines[0]

            if self._looks_like_person_name(first_line):
                start = text.find(first_line)

                entities.append(
                    DetectedEntity(
                        label="PERSON",
                        original=first_line,
                        start=start,
                        end=start + len(first_line),
                        priority=1,
                    )
                )

        # ---------------------------------------------------------
        # 1. Deterministic PII / sensitive attribute rules
        # ---------------------------------------------------------

        regex_patterns = {
            "EMAIL": (
                r"\b[A-Za-z0-9._%+-]+"
                r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
            ),

            "PHONE": (
                r"(?<!\d)"
                r"(?:\+?\d{1,3}[\s.-]?)?"
                r"(?:\(?\d{3}\)?[\s.-]?)"
                r"\d{3}[\s.-]?\d{4}"
                r"(?!\d)"
            ),

            "URL": (
    r"(?i)\b(?:https?://|www\.)[^\s<>()]+"
    r"|\b(?:linkedin\.com|github\.com|gitlab\.com)"
    r"/[^\s<>()]+"
),

            "DOB": (
                r"(?i)\b(?:date\s+of\s+birth|"
                r"dob|birth\s+date)\s*[:\-]?"
                r"\s*[^\n|]+"
            ),

            "AGE": (
                r"(?i)\b(?:age|years?\s*old)"
                r"\s*[:\-]?\s*\d{1,3}\b"
            ),

            "GENDER": (
                r"(?i)\b(?:gender|sex)\s*[:\-]?"
                r"\s*(?:male|female|non[-\s]?binary|"
                r"other|prefer\s+not\s+to\s+say)\b"
            ),

            "NATIONALITY": (
                r"(?i)\bnationality\s*[:\-]?[^\n|]+"
            ),

            "MARITAL_STATUS": (
                r"(?i)\bmarital\s+status\s*[:\-]?[^\n|]+"
            ),

            "RELIGION": (
                r"(?i)\breligion\s*[:\-]?[^\n|]+"
            ),
        }

        for label, pattern in regex_patterns.items():
            for match in re.finditer(pattern, text):
                entities.append(
                    DetectedEntity(
                        label=label,
                        original=match.group(),
                        start=match.start(),
                        end=match.end(),
                        priority=1,
                    )
                )

        # ---------------------------------------------------------
        # 2. spaCy NER — conservative PERSON detection
        # ---------------------------------------------------------

        doc = self.nlp(text)

        for entity in doc.ents:
            if entity.label_ != "PERSON":
                continue

            candidate = entity.text.strip()

            if self._looks_like_person_name(candidate):
                entities.append(
                    DetectedEntity(
                        label="PERSON",
                        original=entity.text,
                        start=entity.start_char,
                        end=entity.end_char,
                        priority=2,
                    )
                )

        return entities

    @staticmethod
    def _looks_like_person_name(value: str) -> bool:
        """
        Conservative validation for spaCy PERSON predictions.

        Rejects multiline spans, long phrases, and strings
        containing obvious technical/resume vocabulary.
        """

        if not value:
            return False

        # Names should normally be a single line.
        if "\n" in value or "\r" in value:
            return False

        # Reject very long spans.
        words = value.split()

        if not (2 <= len(words) <= 4):
            return False

        # A person name should contain alphabetic characters.
        if not all(re.search(r"[A-Za-z]", word) for word in words):
            return False

        # Reject obvious technical/resume terms.
        blocked_terms = {
    # Resume sections
    "resume",
    "curriculum",
    "vitae",
    "professional",
    "profile",
    "summary",
    "objective",
    "contact",
    "education",
    "experience",
    "skills",
    "projects",
    "certifications",
    "coursework",
    "additional",
    "information",

    # Job-related terminology
    "software",
    "engineer",
    "developer",
    "architect",
    "analyst",
    "manager",
    "intern",
    "internship",
    "professional",

    # Technical terminology
    "python",
    "java",
    "javascript",
    "typescript",
    "machine",
    "learning",
    "data",
    "science",
    "docker",
    "kubernetes",
    "postgresql",
    "mysql",
    "sqlite",
    "fastapi",
    "flask",
    "pytorch",
    "tensorflow",
    "scikit",
    "nlp",
    "rag",
    "llm",
    "api",
    "cloud",
    "aws",
    "azure",
    "gcp",
    "linux",
    "git",
    "github",

    # Resume phrases
    "relevant",
    "coursework",
    "certified",
    "certification",
    "certificate",
    "practitioner",
    "bachelor",
    "master",
    "computer",
    "university",
}

        normalized = {word.lower() for word in words}

        if normalized & blocked_terms:
            return False

        return True

    @staticmethod
    def _remove_overlaps(
        entities: list[DetectedEntity],
    ) -> list[DetectedEntity]:
        """
        Resolve overlapping detections.

        Lower priority number = higher priority.
        """

        entities.sort(
            key=lambda entity: (
                entity.start,
                entity.priority,
                -(entity.end - entity.start),
            )
        )

        selected: list[DetectedEntity] = []

        for entity in entities:
            overlaps = any(
                entity.start < existing.end
                and entity.end > existing.start
                for existing in selected
            )

            if not overlaps:
                selected.append(entity)

        return selected

    def anonymize(
        self,
        text: str,
    ) -> tuple[str, list[DetectedEntity]]:
        """Return anonymized text and detected entities."""

        if not text or not text.strip():
            return text, []

        detected = self.detect(text)
        selected = self._remove_overlaps(detected)

        anonymized_text = text

        # Replace from right to left.
        for entity in sorted(
            selected,
            key=lambda item: item.start,
            reverse=True,
        ):
            placeholder = f"[{entity.label}]"

            anonymized_text = (
                anonymized_text[:entity.start]
                + placeholder
                + anonymized_text[entity.end:]
            )

        return anonymized_text, selected


anonymizer = ResumeAnonymizer()