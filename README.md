
# sre-to-llm-journey – Starter Kit

This ZIP contains a ready-to-drop **starter kit** for your 3‑month LLM plan. Unzip into the **root** of your repo and commit.

## What you get
- Structured folders for each week
- Starter code: preprocessing, HF pipelines, tokenization, fine‑tuning, and a Flask API
- Dockerfile + docker-compose for local deploy (with optional Redis cache)
- GitHub Actions CI to run tests and build the Docker image

## Quick start
```bash
# 1) (optional) Create and activate a virtual env
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Try the basic pipeline
python month_2/week_5/hf_pipelines.py

# 4) Run the API locally
docker compose up --build
# -> POST http://localhost:5000/predict {"text":"LLMs are amazing for SRE."}
```

## Suggested workflow
1. Copy all files into your repo root
2. Commit and push on a new branch, e.g. `setup/starter-kit`
3. Open a PR to track changes
4. Iterate weekly by filling each week folder

---
**Security note:** Never commit secrets. Use `.env` (not in git) or GitHub Actions Secrets.
