# AI-Powered Resume Screening and Candidate Shortlisting

AI-assisted recruitment screening system that compares resumes against job descriptions, ranks candidates using job-relevant features, evaluates fairness, explains recommendations, and keeps the final decision with a human recruiter.

## Pipeline

Recruiter UI -> FastAPI -> JD/Resume Processing -> PII Anonymization -> Information Extraction -> Feature Engineering -> Hybrid Matching -> ML Classification -> Fairness Evaluation -> Ranking -> Explainability -> Human Recruiter

## Stack

Streamlit, FastAPI, PyMuPDF, python-docx, spaCy, Sentence Transformers, scikit-learn, Fairlearn, SHAP, SQLite/PostgreSQL, optional ChromaDB, Docker.

## Status

Architecture scaffold.
