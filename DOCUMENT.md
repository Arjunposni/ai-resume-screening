# AI-Powered Resume Screening — Development Record

**Repository:** `Arjunposni/ai-resume-screening`  
**Branch:** `streamlit-deployment`  
**Python:** 3.14.7  
**Primary UI:** Streamlit  
**Backend:** FastAPI  
**Purpose:** Compact record of implementation, errors, corrections, testing, fairness, and deployment.

---

## 1. Final System

```text
Recruiter
   ↓
Streamlit Dashboard
   ↓
JD Parser + Resume Parser
   ↓
PII Anonymization
   ↓
Feature Extraction
   ↓
Candidate ↔ JD Matching
   ↓
Hybrid Score + ML
   ↓
Fairness + SHAP
   ↓
Ranking + Explanation
   ↓
Human Review
```

The system is **AI-assisted decision support**, not an autonomous hiring system.

---

## 2. Stack

| Area | Technology |
|---|---|
| Language | Python 3.14 |
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| Packages | uv |
| Resume parsing | PyMuPDF, python-docx |
| NLP | spaCy + `en_core_web_sm` |
| Embeddings | `all-MiniLM-L6-v2` |
| ML | scikit-learn |
| Fairness | Fairlearn |
| Explainability | SHAP |
| Data | pandas, NumPy |
| Database | SQLite |
| Container | Docker |
| Deployment | Streamlit Community Cloud |

---

## 3. Main Components

```text
app/services/
├── anonymizer.py
├── database.py
├── explainer.py
├── fairness.py
├── feature_extractor.py
├── jd_parser.py
├── matcher.py
├── resume_parser.py
├── shap_explainer.py
├── screening_features.py
└── streamlit_screening.py
```

`frontend/dashboard.py` is the recruiter UI.

---

## 4. Screening Pipeline

### Resume

```text
PDF/DOCX
 → text extraction
 → PII anonymization
 → candidate features
 → matching
 → scoring
 → ML prediction
 → explanation
```

### Job Description

Extracts:

- Title
- Required skills
- Preferred skills
- Minimum experience
- Education

---

## 5. Scoring

Current weights:

| Feature | Weight |
|---|---:|
| Required skills | 40% |
| Experience | 25% |
| Semantic similarity | 15% |
| Education | 10% |
| Preferred skills | 10% |

```text
Final =
Required × .40
+ Experience × .25
+ Semantic × .15
+ Education × .10
+ Preferred × .10
```

Threshold:

```text
>= 75 → SHORTLIST
< 75  → REVIEW
```

The threshold is a recommendation, not an automatic rejection/hiring rule.

---

## 6. Important Bugs & Corrections

| Issue | Correction |
|---|---|
| `CandidateJDMatcher` was referenced | Actual class is `CandidateMatcher` |
| ML model loaded incorrectly in Streamlit | Use shared `fairness_model` instance |
| Explicit `3.2 years` experience needed better handling | Convert duration to months |
| Candidate 043 experience was inaccurate | Corrected to 38 months / 3.17 years |
| `float(None)` could break explanations | Added safe handling |
| SHAP lacked required test data in dashboard | Recreated deterministic evaluation split |
| Dashboard depended on separate FastAPI server | Added native `streamlit_screening.py` |
| Old UI said backend/FastAPI | Replaced with Streamlit-native wording |
| `fitz` warning appeared | Prefer current `pymupdf` API during future cleanup |
| `openapi-local.json` appeared untracked | Kept it out of deployment commits |

---

## 7. Experience Validation

Test JD:

```text
Backend Engineer
Required: Python, FastAPI, PostgreSQL, Docker
Preferred: AWS, Kubernetes, Machine Learning
Experience: 3+ years
Education: Bachelor's Degree in Computer Science
```

Candidate 043:

```text
3.2 years
```

Result:

```text
38 months
3.17 years
Required: 3.0 years
Experience score: 100.0
Meets requirement: True
Final score: 96.98
```

A candidate slightly below the experience requirement is not automatically rejected; the system can flag the case for human review.

---

## 8. Batch Test

100 synthetic resumes were processed:

| Result | Count |
|---|---:|
| Submitted | 100 |
| Processed | 100 |
| Failed | 0 |
| Shortlisted | 82 |
| Review | 18 |
| Average score | 81.38 |

Examples:

```text
Candidate 022 → 2.83 years → 95.87
Candidate 025 → 2.58 years → 94.09
```

These were treated as human-review cases despite being high scoring.

**All evaluation data/results are synthetic.**

---

## 9. ML & Fairness

Features:

```text
required_skill_score
experience_score
education_score
semantic_score
preferred_skill_score
```

Protected attributes are kept separate from ranking features.

Models:

```text
Logistic Regression
KNN (n_neighbors=5, weights=distance)
Fairlearn ExponentiatedGradient + DemographicParity
```

Synthetic held-out results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | DP Difference |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6000 | 0.5714 | 0.4444 | 0.5000 | 0.7904 | 0.0625 |
| KNN | 0.6250 | 0.6154 | 0.4444 | 0.5161 | 0.6894 | 0.0208 |
| Fairness-aware | 0.5750 | 0.5385 | 0.3889 | 0.4516 | — | 0.0208 |

Observation:

- KNN had the best accuracy/precision/F1 in this synthetic test.
- Logistic Regression had the best ROC-AUC.
- Fairness-aware learning reduced the measured demographic-parity gap compared with baseline.
- No model is universally best from this experiment.

---

## 10. Synthetic Evaluation Dataset

```text
Candidates: 100
Selected: 44
Not selected: 56

Group A: 60
Group B: 40

Selection rate:
A = 45.0%
B = 42.5%
Difference = 2.5 percentage points
```

Selected-vs-not-selected feature differences:

```text
Required skills: +16.91
Experience:       +6.84
Semantic:         +4.74
Preferred:        +1.92
Education:        +1.71
```

Correlation with `selected`:

```text
Required skills: 0.4888
Experience:      0.1904
Semantic:        0.1461
Education:       0.0577
Preferred:       0.0412
```

These numbers describe the synthetic evaluation only.

---

## 11. SHAP

Implemented with `shap.LinearExplainer`.

Example validated explanation:

```text
Prediction: not_selected
Selection probability: 0.324077
Base value: -0.796836
```

Contributions:

```text
experience_score        -0.830196
required_skill_score    +0.726543
semantic_score          +0.378503
education_score         -0.191468
preferred_skill_score   -0.021644
```

SHAP values are model explanation values, not direct percentages of the final screening score.

---

## 12. Explainability

Candidate explanations include:

```text
Score breakdown
Matched required skills
Missing required skills
Preferred skills
Experience
Education
Semantic similarity
ML prediction
Strengths
Concerns
Human-review flags
```

Purpose:

```text
Score + evidence + explanation
```

rather than an unexplained ranking.

---

## 13. Streamlit Migration

Originally:

```text
Streamlit → FastAPI → Services
```

Final primary deployment:

```text
Streamlit → Services
```

`app/services/streamlit_screening.py` now handles:

```text
parse_job_description()
process_resume()
screen_candidate()
```

This means the main Streamlit screening flow does not require a separately running FastAPI server.

FastAPI remains available for API-based use.

---

## 14. Local Setup

```powershell
git clone https://github.com/Arjunposni/ai-resume-screening.git
cd ai-resume-screening
uv venv --python 3.14
.venv\Scripts\Activate.ps1
uv sync
uv run python -m spacy download en_core_web_sm
uv run streamlit run frontend/dashboard.py
```

Open:

```text
http://localhost:8501
```

Dependency check:

```powershell
uv run python -c "import fastapi, streamlit, sklearn, fairlearn, shap, pymupdf; print('All deployment dependencies: OK')"
```

Verified output:

```text
All deployment dependencies: OK
```

spaCy check:

```powershell
uv run python -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy model: OK')"
```

---

## 15. FastAPI

Run:

```powershell
uv run uvicorn app.main:app --reload
```

Health:

```text
http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok"}
```

Main routes:

```text
GET  /health
POST /jobs/parse
POST /resumes/upload
POST /screening/match
POST /screening/batch
```

---

## 16. Docker

```powershell
docker build -t ai-resume-screening .
docker run -p 8501:8501 ai-resume-screening
```

Open:

```text
http://localhost:8501
```

---

## 17. Git & Deployment

Repository:

```text
Arjunposni/ai-resume-screening
```

Deployment branch:

```text
streamlit-deployment
```

Streamlit entry point:

```text
frontend/dashboard.py
```

The Streamlit deployment was prepared for:

```text
Streamlit Community Cloud
```

Configuration:

```text
Repository: Arjunposni/ai-resume-screening
Branch: streamlit-deployment
Main file: frontend/dashboard.py
```

---

## 18. README Merge Conflict

Observed:

```text
Auto-merging README.md
CONFLICT (content): Merge conflict in README.md
Automatic merge failed
```

Cause:

```text
Local README and remote README had conflicting changes.
```

Resolution used:

```powershell
git checkout --theirs README.md
git add README.md
git commit -m "Resolve README merge conflict"
git push origin streamlit-deployment
```

Important:

```text
Do not run another pull while the merge conflict is unresolved.
Resolve → add → commit → push.
```

---

## 19. Important Files

```text
frontend/dashboard.py       → Streamlit UI
app/main.py                 → FastAPI application
app/services/matcher.py     → Candidate-JD matching
app/services/fairness.py    → ML + fairness
app/services/shap_explainer.py → SHAP
app/services/explainer.py   → Candidate explanations
app/services/streamlit_screening.py → Native Streamlit pipeline
data/evaluation/candidates.csv → Synthetic evaluation data
models/*.joblib             → Trained model artifacts
pyproject.toml              → Dependencies/project config
uv.lock                     → Locked environment
Dockerfile                  → Container setup
```

---

## 20. Security & Production Notes

Never commit:

```text
.env
API keys
Passwords
Private resumes
Private candidate information
```

A production system additionally needs:

```text
Authentication
Authorization
HTTPS
Encryption
Secure file storage
Audit logs
Retention/deletion policies
Access controls
Legal/privacy review
Continuous fairness monitoring
```

The current project is a prototype/educational decision-support system.

---

## 21. Key Lessons

- Resume parsing needs realistic test cases.
- Explicit experience durations need dedicated parsing.
- Removing PII does not remove proxy bias.
- Fairness and predictive performance must be evaluated together.
- SHAP makes model behavior easier to inspect.
- Human review should remain part of hiring decisions.
- Native Streamlit processing simplifies free deployment.
- Git conflicts should be resolved before continuing with other Git operations.

---

## 22. Final Status

```text
[x] PDF/DOCX parsing
[x] PII anonymization
[x] Feature extraction
[x] JD parsing
[x] Skill matching
[x] Experience matching
[x] Education matching
[x] Semantic similarity
[x] Hybrid scoring
[x] Candidate ranking
[x] Logistic Regression
[x] KNN
[x] Fairness-aware model
[x] Fairness evaluation
[x] SHAP
[x] Explanations
[x] Human-review recommendations
[x] Streamlit dashboard
[x] Native Streamlit screening
[x] Batch processing
[x] Synthetic evaluation
[x] FastAPI
[x] Docker
[x] GitHub deployment branch
[x] Streamlit deployment preparation
```

---

## 23. Final Project Statement

The completed system combines:

```text
NLP
+ Information Extraction
+ Semantic Matching
+ Machine Learning
+ Fairness Evaluation
+ Explainable AI
+ Human Oversight
+ Web Deployment
```

The intended output is not simply:

```text
Candidate = 92
```

but:

```text
Candidate = 92
        +
Why?
        +
Which skills matched?
        +
What is missing?
        +
What did the model predict?
        +
Are there fairness concerns?
        +
Human recruiter review
```

**End of document.**
