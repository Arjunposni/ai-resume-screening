import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.resumes import router as resume_router
from app.api.jobs import router as jobs_router
from app.api.screening import router as screening_router
from app.services.database import database

load_dotenv()

app = FastAPI(
    title="AI-Powered Resume Screening API",
    version="0.1.0",
)


# ============================================================
# CORS
# ============================================================

frontend_url = os.getenv(
    "FRONTEND_URL",
    "http://localhost:8501",
).strip().rstrip("/")

allowed_origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

if frontend_url and frontend_url not in allowed_origins:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ROUTES
# ============================================================

app.include_router(resume_router)
app.include_router(jobs_router)
app.include_router(screening_router)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}