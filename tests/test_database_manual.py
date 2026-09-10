from app.services.database import Database

db = Database()
print("Schema initialized OK")

db.save_job("job-1", "Backend Engineer", "raw jd text", {"required_skills": ["python"]})
db.save_job("job-1", "Backend Engineer v2", "raw jd text updated", {"required_skills": ["python", "fastapi"]})
print("save_job (insert + upsert) OK")

db.save_candidate("cand-1", "resume.pdf", "stored_resume.pdf", "pdf", "/tmp/resume.pdf", "anonymized text", {"skills": ["python"]})
print("save_candidate OK")

db.create_screening_run("run-1", "job-1", candidate_count=1, status="running")
db.update_screening_run("run-1", status="completed", candidate_count=1)
print("screening_run create+update OK")

db.save_screening_result("run-1", "cand-1", 91.5, "SHORTLIST", {"required_skill_score": 100}, ml_prediction=1)
print("save_screening_result OK")

db.save_recruiter_decision("run-1", "cand-1", "shortlist", notes="looks strong")
print("save_recruiter_decision OK")

db.add_audit_event("evt-1", "screening_completed", run_id="run-1", candidate_id="cand-1", details={"note": "ok"})
print("add_audit_event OK")

print("get_screening_results ->", db.get_screening_results("run-1"))
print("get_latest_runs ->", db.get_latest_runs(limit=5))