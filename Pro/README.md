# AI-Powered Resume Screening & Job Matching System

A final-year project that parses resumes, compares them against a job
description using three different techniques (TF-IDF, SBERT embeddings,
and LLM reasoning), and outputs a ranked, explainable candidate list.

## Project Structure
```
resume_screener/
├── resume_parser.py     # Extracts text + structured fields from resumes
├── matching_engine.py    # TF-IDF / SBERT / LLM scoring functions
├── app.py                 # Streamlit dashboard (the UI)
├── requirements.txt
└── README.md
```

## 1. Setup

### Install Python (3.9+ recommended) and create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### (Optional) Get an OpenAI API key
Only needed if you want the LLM-based scoring + explanations feature.
- Sign up at https://platform.openai.com
- Generate an API key
- You'll paste this into the app's sidebar when running (never hardcode it in code)

## 2. Run the app
```bash
streamlit run app.py
```
This opens a browser tab at `http://localhost:8501`.

## 3. How to use it
1. Paste a job description into the text box.
2. Upload one or more resumes (PDF or DOCX).
3. (Optional) Check "Use LLM scoring" in the sidebar and paste your API key.
4. Click **Rank Candidates**.
5. View the ranked table, and download results as CSV.

---

## How each scoring method works (for your report/viva)

### 1. TF-IDF + Cosine Similarity (`tfidf_score`)
- **Library:** `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`)
- **How it works:** Converts both texts into vectors based on word frequency
  (weighted by how rare/common each word is across the two documents), then
  measures the angle between the vectors.
- **Strength:** Fast, no model download, fully interpretable.
- **Weakness:** Purely keyword-based — misses synonyms
  (e.g., won't connect "ML" with "Machine Learning").

### 2. SBERT Semantic Similarity (`sbert_score`)
- **Library:** `sentence-transformers`, model: `all-MiniLM-L6-v2`
- **How it works:** A pretrained transformer model converts each text into a
  dense 384-dimension vector that captures *meaning*, not just words. Cosine
  similarity between these vectors reflects semantic closeness.
- **Strength:** Understands paraphrasing and synonyms — much better real-world matching.
- **Weakness:** Slightly slower, and it's a "black box" — no reasoning, just a number.
- **Note:** the model (~80MB) downloads automatically the first time you run it — needs internet on first run only, then it's cached locally.

### 3. LLM-Based Scoring (`llm_score`)
- **Library:** `openai` (swap for `anthropic` if using Claude)
- **How it works:** Sends the resume + JD to an LLM with a structured prompt
  asking it to output a JSON verdict: match score, matched/missing skills,
  and a plain-English explanation.
- **Strength:** This is the differentiator — it doesn't just give a number,
  it *reasons* about the match and tells you *why*, which is what a human
  recruiter actually wants.
- **Weakness:** Costs API tokens per resume, slower, and needs careful prompt
  design to keep output format consistent.

### Why show all three in your final report
This progression (baseline → better → best) is a strong story for your
viva/interviews: you can explain *when* each is appropriate, and demonstrate
that you understand tradeoffs — not just "I called an LLM API."

---

## 4. Suggested Upgrades (bonus points / stretch goals)

- **Better name/entity extraction:** Replace the regex-based name extractor
  in `resume_parser.py` with `spaCy`'s NER (`en_core_web_sm` model) for more
  robust name/organization/degree extraction.
- **Bias & fairness audit:** Test whether scores change when you swap gendered
  names, remove elite university mentions, or introduce employment gaps.
  Document your findings — this is a genuinely important, current topic in
  AI hiring tools.
- **Vector database for large-scale matching:** If matching against
  thousands of resumes, store embeddings in a vector DB (FAISS or ChromaDB)
  instead of comparing one at a time.
- **Fine-tuning:** Fine-tune a small classifier (e.g., logistic regression
  on top of SBERT embeddings) on labeled "good fit / not a good fit" data if
  you can get access to real hiring outcome data.

## 5. Deployment (for your demo/submission)
- **Docker:** Wrap the app in a `Dockerfile` (base image `python:3.11-slim`,
  copy code, `pip install -r requirements.txt`, `CMD ["streamlit", "run", "app.py"]`)
- **Free hosting:** Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud)
  (easiest — just connect your GitHub repo) or Hugging Face Spaces.

## 6. What to include in your final report
- Problem statement + motivation
- Architecture diagram (parser → scoring engine → dashboard)
- Comparison table: TF-IDF vs SBERT vs LLM (accuracy, speed, cost, explainability)
- Bias/fairness findings (if you did the audit)
- Screenshots of the working dashboard
- Limitations and future work
