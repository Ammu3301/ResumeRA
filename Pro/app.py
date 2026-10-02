"""
app.py
------
Streamlit dashboard: a recruiter uploads a job description + multiple
resumes, and gets back a ranked table of candidates with score
breakdowns (TF-IDF, SBERT, and optionally LLM explanation).

Run with:
    streamlit run app.py

Libraries used:
- streamlit -> quick web UI, no frontend code needed
- tempfile / os -> handle uploaded files on disk temporarily
"""

import os
import tempfile

import pandas as pd
import streamlit as st

from resume_parser import parse_resume
from matching_engine import tfidf_score, sbert_score, llm_score

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI-Powered Resume Screening & Job Matching")
st.caption("Upload a job description and resumes to get ranked, explainable candidate scores.")

# --- Sidebar settings ---
st.sidebar.header("Settings")
use_llm = st.sidebar.checkbox("Use LLM scoring (needs API key)", value=False)
api_key = ""
if use_llm:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# --- Inputs ---
jd_text = st.text_area("Paste the Job Description here:", height=200)

uploaded_files = st.file_uploader(
    "Upload candidate resumes (PDF or DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

run_button = st.button("Rank Candidates", type="primary")

if run_button:
    if not jd_text.strip():
        st.error("Please paste a job description first.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        llm_client = None
        if use_llm and api_key:
            from openai import OpenAI
            llm_client = OpenAI(api_key=api_key)

        results = []
        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):
            # Save uploaded file to a temp path so our parser can read it
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                parsed = parse_resume(tmp_path)
                resume_text = parsed["raw_text"]

                row = {
                    "File": uploaded_file.name,
                    "Name": parsed["name"],
                    "Email": parsed["email"],
                    "Years Exp": parsed["years_experience"],
                    "Skills Found": ", ".join(parsed["skills"]) or "-",
                    "TF-IDF Score": tfidf_score(resume_text, jd_text),
                    "SBERT Score": sbert_score(resume_text, jd_text),
                }

                if use_llm and llm_client:
                    llm_result = llm_score(resume_text, jd_text, client=llm_client)
                    row["LLM Match %"] = llm_result.get("match_score", "-")
                    row["Missing Skills"] = ", ".join(llm_result.get("missing_skills", []))
                    row["Explanation"] = llm_result.get("overall_explanation", "-")

                results.append(row)
            except Exception as e:
                st.warning(f"Could not process {uploaded_file.name}: {e}")
            finally:
                os.remove(tmp_path)

            progress.progress((i + 1) / len(uploaded_files))

        if results:
            df = pd.DataFrame(results)
            # Rank by SBERT score by default (semantic similarity is usually most reliable)
            df = df.sort_values(by="SBERT Score", ascending=False).reset_index(drop=True)
            df.index += 1

            st.success(f"Ranked {len(df)} candidates.")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results as CSV", csv, "ranked_candidates.csv", "text/csv")
