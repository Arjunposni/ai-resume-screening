# Job description endpoints will go here.
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.jd_parser import jd_parser

router = APIRouter(prefix="/jobs", tags=["Jobs"])


class JobDescriptionRequest(BaseModel):
    text: str


@router.post("/parse")
def parse_job_description(request: JobDescriptionRequest):
    job_description = jd_parser.parse(request.text)
    return job_description.model_dump()