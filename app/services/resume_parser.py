# PDF/DOCX/TXT parsing.
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


class ResumeParser:
    """Extract plain text from supported resume document formats."""

    def parse(self, file_path: str) -> str:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Resume not found: {file_path}")

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        if extension == ".pdf":
            return self._parse_pdf(path)

        if extension == ".docx":
            return self._parse_docx(path)

        raise ValueError(f"Unsupported file type: {extension}")

    def _parse_pdf(self, path: Path) -> str:
        """Extract text from every page of a PDF."""

        text_parts = []

        with fitz.open(path) as document:
            for page in document:
                text = page.get_text("text")

                if text.strip():
                    text_parts.append(text.strip())

        return "\n\n".join(text_parts)

    def _parse_docx(self, path: Path) -> str:
        """Extract text from paragraphs and tables in a DOCX file."""

        document = Document(path)

        text_parts = []

        # Normal paragraphs
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()

            if text:
                text_parts.append(text)

        # Tables
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]

                row_text = " | ".join(
                    cell for cell in cells if cell
                )

                if row_text:
                    text_parts.append(row_text)

        return "\n".join(text_parts)


resume_parser = ResumeParser()