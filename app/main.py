from fastapi import FastAPI

from app.api.resumes import router as resume_router


app = FastAPI(
    title="AI-Powered Resume Screening API",
    version="0.1.0",
)


app.include_router(resume_router)


@app.get("/health")
def health():
    return {
        "status": "ok"
    }