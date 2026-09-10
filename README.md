# AI-Powered Resume Screening & Candidate Shortlisting

> **AI-assisted recruitment screening with fairness, explainability, and human oversight.**

A Streamlit-based application that helps recruiters compare multiple resumes against a job description, rank candidates, understand the reasons behind scores, and evaluate potential bias.

**Final hiring decisions always remain with a human recruiter.**

---

## 🚀 Features

- 📄 PDF and DOCX resume parsing
- 🔐 PII detection and anonymization
- 🧠 Skill, experience, education and project extraction
- 🎯 Required vs preferred skill matching
- 🔎 Semantic similarity using `all-MiniLM-L6-v2`
- 📊 Weighted candidate scoring and ranking
- 🤖 Logistic Regression + KNN
- ⚖️ Fairness evaluation with Fairlearn
- 💡 SHAP model explanations
- 👥 Human-review recommendations
- 📚 Single or batch resume processing
- 🖥️ Streamlit recruiter dashboard
- 🚀 Streamlit Community Cloud deployment
- 🐳 Docker support

---

## 🏗️ Architecture

```text
             Recruiter
                 │
                 ▼
        ┌─────────────────┐
        │ Streamlit UI    │
        └────────┬────────┘
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
 Job Description         Resumes
       │                   │
       ▼                   ▼
    JD Parser         Text Extraction
                           │
                           ▼
                    PII Anonymization
                           │
                           ▼
                    Feature Extraction
       │                   │
       └─────────┬─────────┘
                 ▼
          Candidate Matching
                 │
                 ▼
          Hybrid Scoring
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
      ML      Fairness    SHAP
       │      Analysis  Explanation
       └─────────┬─────────┘
                 ▼
          Ranked Candidates
                 │
                 ▼
           Human Review
```

---

## 🧰 Tech Stack

| Category | Technology |
|---|---|
| UI | Streamlit |
| API | FastAPI |
| Language | Python 3.14 |
| Package manager | uv |
| NLP | spaCy |
| Resume parsing | PyMuPDF, python-docx |
| Embeddings | sentence-transformers |
| Embedding model | all-MiniLM-L6-v2 |
| ML | scikit-learn |
| Fairness | Fairlearn |
| Explainability | SHAP |
| Data | pandas, NumPy |
| Database | SQLite |
| Container | Docker |

---

# ⚡ Quick Start

### 1. Clone

```powershell
git clone https://github.com/Arjunposni/ai-resume-screening.git
cd ai-resume-screening
```

### 2. Create environment

```powershell
uv venv --python 3.14
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
uv sync
```

### 4. Install spaCy model

```powershell
uv run python -m spacy download en_core_web_sm
```

### 5. Start the application

```powershell
uv run streamlit run frontend/dashboard.py
```

Open:

```text
http://localhost:8501
```

**That's it.**

---

# 👤 How to Use

### 1. Enter a Job Description

Example:

```text
Backend Engineer

Required:
Python, FastAPI, PostgreSQL, Docker

Preferred:
AWS, Kubernetes, Machine Learning

Experience:
3+ years

Education:
Bachelor's Degree in Computer Science
```

### 2. Upload resumes

Supported:

```text
.pdf
.docx
```

Upload one resume or many.

### 3. Process

The system:

```text
Extracts → Anonymizes → Extracts Features → Matches → Scores → Ranks
```

### 4. Review

For each candidate you can inspect:

- Final score
- Matched skills
- Missing skills
- Experience
- Education
- Semantic similarity
- ML prediction
- Explanation
- Human-review flags

---

# 📊 Scoring

Current scoring weights:

| Component | Weight |
|---|---:|
| Required Skills | 40% |
| Experience | 25% |
| Semantic Similarity | 15% |
| Education | 10% |
| Preferred Skills | 10% |

```text
Final Score =
  Required Skills × 0.40
+ Experience × 0.25
+ Semantic Similarity × 0.15
+ Education × 0.10
+ Preferred Skills × 0.10
```

Default shortlist threshold:

```text
75 / 100
```

```text
Score >= 75 → SHORTLIST
Score < 75  → REVIEW
```

This is a recommendation, **not an automatic hiring decision**.

---

# ⚖️ Fairness & Bias

Protected attributes are not used as ranking features.

The project evaluates fairness separately using:

- Demographic Parity Difference
- Logistic Regression
- KNN
- Fairlearn's fairness-aware learning

The project also recognizes that removing protected attributes does **not** guarantee a bias-free model because other features can act as proxies.

The evaluation dataset included in the repository is **synthetic** and is not real hiring data.

---

# 💡 Explainability

The system uses **SHAP** to show how model features contribute to predictions.

Example features:

```text
Required Skill Score
Experience Score
Education Score
Semantic Score
Preferred Skill Score
```

The purpose is to help recruiters understand model behavior rather than receiving only an unexplained score.

---

# 🤖 ML Models

### Logistic Regression

Used as the baseline classifier.

### KNN

Configured with:

```text
n_neighbors = 5
weights = distance
```

### Fairness-Aware Model

Uses:

```text
Fairlearn
ExponentiatedGradient
DemographicParity
```

The models are compared using both predictive performance and fairness metrics.

---

# 📁 Project Structure

```text
ai-resume-screening/
├── app/
│   ├── api/
│   ├── models/
│   └── services/
│       ├── anonymizer.py
│       ├── explainer.py
│       ├── fairness.py
│       ├── feature_extractor.py
│       ├── jd_parser.py
│       ├── matcher.py
│       ├── resume_parser.py
│       ├── shap_explainer.py
│       └── streamlit_screening.py
│
├── data/
│   └── evaluation/
│       ├── candidates.csv
│       └── fairness_results.json
│
├── frontend/
│   └── dashboard.py
│
├── models/
│   ├── baseline_model.joblib
│   ├── knn_model.joblib
│   ├── fairness_model.joblib
│   └── scaler.joblib
│
├── tests/
│   └── resumes/
│
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# 🔌 FastAPI

The project also contains a FastAPI backend.

Run it with:

```powershell
uv run uvicorn app.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

Main endpoints:

```text
GET  /health
POST /jobs/parse
POST /resumes/upload
POST /screening/match
POST /screening/batch
```

The current Streamlit-native workflow does **not** require FastAPI to be running separately.

---

# 🐳 Docker

Build:

```powershell
docker build -t ai-resume-screening .
```

Run:

```powershell
docker run -p 8501:8501 ai-resume-screening
```

Open:

```text
http://localhost:8501
```

---

# ☁️ Streamlit Cloud

The current Streamlit application can be deployed directly from GitHub.

Use:

```text
Repository: Arjunposni/ai-resume-screening
Branch:     streamlit-deployment
Main file:  frontend/dashboard.py
```

Deploy it through Streamlit Community Cloud.

No separate FastAPI server is required for the Streamlit-native screening workflow.

---

# 🧪 Testing

Verify dependencies:

```powershell
uv run python -c "import fastapi, streamlit, sklearn, fairlearn, shap, pymupdf; print('All deployment dependencies: OK')"
```

Verify spaCy:

```powershell
uv run python -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy model: OK')"
```

Then launch:

```powershell
uv run streamlit run frontend/dashboard.py
```

Test with:

1. A sample JD
2. One PDF/DOCX resume
3. Multiple resumes
4. Candidate explanations
5. Fairness page
6. System page

---

# 📈 Synthetic Evaluation

The included synthetic evaluation dataset contains:

```text
100 candidates
56 not selected
44 selected

Group A: 60
Group B: 40
```

A previous synthetic 100-resume batch test produced:

```text
Submitted:       100
Processed:       100
Failed:            0
Shortlisted:      82
Review:           18
Average score: 81.38
```

These are **test results only**, not real-world hiring statistics.

---

# 🔒 Privacy & Security

Do not commit:

```text
.env
API keys
Passwords
Private credentials
Private candidate resumes
Sensitive candidate information
```

For production use, add:

- Authentication
- Authorization
- HTTPS
- Encryption
- Secure storage
- Data retention/deletion policies
- Audit logging
- Strong access controls

---

# 🧠 Design Principles

### Human in the loop

```text
AI screening
     ↓
Evidence + score + explanation
     ↓
Recruiter review
     ↓
Final human decision
```

### Why hybrid matching?

Keyword matching can miss context.

Semantic matching can miss exact mandatory skills.

Therefore the system combines:

```text
Deterministic matching
+
Semantic similarity
+
Machine learning
```

### Why fairness evaluation?

A model can appear neutral while still learning biased patterns or proxy features.

Fairness is therefore evaluated separately instead of assuming that anonymization solves everything.

---

# ⚠️ Limitations

This is an **educational/prototype AI-assisted recruitment system**.

It should not be used as the sole basis for employment decisions.

Automated screening can reproduce or amplify biases present in data, features, models, or organizational processes.

Before real-world use, the system requires appropriate:

- Legal review
- Security controls
- Privacy controls
- Bias validation
- Accessibility testing
- Human oversight
- Continuous monitoring

---

# 📌 Project Status

- [x] Resume PDF parsing
- [x] Resume DOCX parsing
- [x] PII anonymization
- [x] Candidate feature extraction
- [x] JD parsing
- [x] Skill matching
- [x] Experience matching
- [x] Education matching
- [x] Semantic similarity
- [x] Hybrid scoring
- [x] Candidate ranking
- [x] Logistic Regression
- [x] KNN
- [x] Fairness-aware model
- [x] Fairness evaluation
- [x] SHAP explainability
- [x] Human-review recommendations
- [x] Streamlit dashboard
- [x] Batch processing
- [x] Synthetic evaluation
- [x] FastAPI backend
- [x] Docker support
- [x] Streamlit deployment

---

# 👨‍💻 Author

**Arjun Posni**

GitHub:

https://github.com/Arjunposni/ai-resume-screening

---

## ⭐ Quickest Possible Setup

Already have Git, Python 3.14 and uv?

```powershell
git clone https://github.com/Arjunposni/ai-resume-screening.git
cd ai-resume-screening
uv venv --python 3.14
.venv\Scripts\Activate.ps1
uv sync
uv run python -m spacy download en_core_web_sm
uv run streamlit run frontend/dashboard.py
```

Then open:

```text
http://localhost:8501
```

**Clone → Install → Run → Screen resumes. 🚀**
