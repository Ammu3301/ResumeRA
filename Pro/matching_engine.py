"""
matching_engine.py
-------------------
Three approaches to score a resume against a job description, from
simplest to most sophisticated. Building all three lets you compare them
in your final report (great talking point in interviews/viva).

1. TF-IDF + Cosine Similarity   (classic, fast, keyword-based, baseline)
2. SBERT Sentence Embeddings    (semantic similarity - understands
                                  synonyms like "ML" == "Machine Learning")
3. LLM-based scoring            (most powerful - reasons about the match
                                  and explains WHY, not just a number)

Libraries used:
- scikit-learn         -> TfidfVectorizer, cosine_similarity
- sentence-transformers -> pretrained SBERT model for embeddings
- openai (or anthropic) -> LLM API call for reasoning-based scoring
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# 1. TF-IDF BASELINE
# ---------------------------------------------------------------------
def tfidf_score(resume_text: str, jd_text: str) -> float:
    """
    Classic keyword-overlap based similarity.
    Fast, no ML model download needed, but misses synonyms
    (e.g. won't know "ML" and "Machine Learning" are related).
    Returns a score between 0 and 1.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(score), 4)


# ---------------------------------------------------------------------
# 2. SBERT SEMANTIC SIMILARITY
# ---------------------------------------------------------------------
# NOTE: sentence-transformers downloads a small pretrained model (~80MB)
# the first time you run this. Needs internet access on first run.
_sbert_model = None


def _get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        # 'all-MiniLM-L6-v2' is small, fast, and good enough for this task.
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def sbert_score(resume_text: str, jd_text: str) -> float:
    """
    Embeds both texts into vectors using a pretrained transformer model,
    then measures cosine similarity between the vectors. This captures
    MEANING, not just exact word overlap - so "led a team of 5 engineers"
    will match "team leadership experience" even with zero shared words.
    """
    model = _get_sbert_model()
    embeddings = model.encode([resume_text, jd_text])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(score), 4)


# ---------------------------------------------------------------------
# 3. LLM-BASED SCORING + EXPLANATION  (the differentiator)
# ---------------------------------------------------------------------
# Uses the OpenAI API here as an example. You can swap this for the
# Anthropic API (Claude) or any other LLM provider - just change the
# client setup and the call below.

LLM_PROMPT_TEMPLATE = """You are an expert technical recruiter. Compare the RESUME against the
JOB DESCRIPTION and return a JSON object ONLY (no extra text) with this
exact structure:

{{
  "match_score": <integer 0-100>,
  "matched_skills": [<list of matched skills as strings>],
  "missing_skills": [<list of important missing skills as strings>],
  "experience_fit": "<one short sentence>",
  "overall_explanation": "<2-3 sentence explanation of the score>"
}}

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}
"""


def llm_score(resume_text: str, jd_text: str, client=None, model_name: str = "gpt-4o-mini") -> dict:
    """
    Sends the resume + JD to an LLM and asks it to reason about the match,
    not just compute a similarity number. This gives you an EXPLANATION,
    which is what makes this project stand out over a plain ML classifier.

    `client` should be an already-initialized OpenAI() client object.
    Pass your own client so this function stays provider-agnostic and
    testable without hardcoding API keys here.
    """
    if client is None:
        raise ValueError(
            "Pass an initialized API client, e.g.:\n"
            "  from openai import OpenAI\n"
            "  client = OpenAI(api_key='YOUR_KEY')\n"
            "  llm_score(resume_text, jd_text, client=client)"
        )

    prompt = LLM_PROMPT_TEMPLATE.format(jd_text=jd_text, resume_text=resume_text)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw_output = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model wraps JSON in ```json ... ```
    raw_output = raw_output.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "match_score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "experience_fit": "Could not parse LLM response",
            "overall_explanation": raw_output,
        }


# ---------------------------------------------------------------------
# COMBINED SCORING PIPELINE
# ---------------------------------------------------------------------
def combined_score(resume_text: str, jd_text: str, llm_client=None, use_llm: bool = False) -> dict:
    """
    Runs TF-IDF and SBERT always (fast, free), and optionally the LLM
    scoring (slower, costs API tokens) if use_llm=True and a client is
    provided. Combine results into a single report dict.
    """
    result = {
        "tfidf_similarity": tfidf_score(resume_text, jd_text),
        "sbert_similarity": sbert_score(resume_text, jd_text),
    }

    if use_llm and llm_client is not None:
        result["llm_analysis"] = llm_score(resume_text, jd_text, client=llm_client)

    return result


if __name__ == "__main__":
    sample_resume = "Experienced Python developer with 3 years in machine learning and NLP projects using PyTorch."
    sample_jd = "Looking for a Machine Learning engineer skilled in Python and deep learning frameworks."

    print("TF-IDF score:", tfidf_score(sample_resume, sample_jd))
    # SBERT line commented out by default since it needs to download a model
    # print("SBERT score:", sbert_score(sample_resume, sample_jd))
