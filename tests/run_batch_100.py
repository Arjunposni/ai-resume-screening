import requests
from pathlib import Path
import json

resume_dir = Path("tests/resumes")
files = sorted(resume_dir.glob("*.docx"))

job = """Backend Engineer

REQUIRED SKILLS
Python, FastAPI, PostgreSQL, Docker

PREFERRED SKILLS
AWS, Kubernetes, Machine Learning

EXPERIENCE
3+ years of backend software engineering experience.

EDUCATION
Bachelor's Degree in Computer Science or a closely related field.
"""

multipart_files = [
    ("files", (file.name, open(file, "rb"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
    for file in files
]

try:
    response = requests.post(
        "http://127.0.0.1:8000/screening/batch",
        data={"job": job},
        files=multipart_files,
        timeout=300,
    )

    print("HTTP STATUS:", response.status_code)

    response.raise_for_status()

    result = response.json()

    Path("data/evaluation").mkdir(parents=True, exist_ok=True)

    with open("data/evaluation/batch_100_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("STATUS: SUCCESS")
    print("SUBMITTED:", result.get("submitted_candidates"))
    print("PROCESSED:", result.get("processed_candidates"))
    print("FAILED:", result.get("failed_candidates"))
    print("SHORTLISTED:", result.get("shortlisted"))
    print("REVIEW:", result.get("needs_review"))
    print("AVERAGE SCORE:", result.get("average_score"))

finally:
    for _, (_, file_obj, _) in multipart_files:
        file_obj.close()
