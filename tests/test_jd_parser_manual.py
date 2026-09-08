from app.services.jd_parser import jd_parser

jd = """Backend Engineer

Required Skills:
Python, FastAPI, PostgreSQL, Docker
3+ years of experience
Bachelor's degree in Computer Science required

Preferred Skills:
AWS, Kubernetes, Machine Learning
"""

result = jd_parser.parse(jd)

print(result.model_dump())