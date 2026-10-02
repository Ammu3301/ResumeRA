"""
resume_parser.py
-----------------
Extracts raw text from resumes (PDF/DOCX) and pulls out structured
information: name, email, phone, skills, years of experience, education.

Libraries used:
- pdfplumber   -> extract text from PDF resumes
- python-docx  -> extract text from DOCX resumes
- re (built-in)-> regex based field extraction (email, phone, experience)

Why regex instead of a heavy NLP model here?
Resumes have fairly predictable patterns for emails/phone numbers, and a
skills list can be matched against a known vocabulary. This keeps the
pipeline fast and dependency-light. You can swap in spaCy NER later as
an upgrade (mentioned in README).
"""

import re
import pdfplumber
import docx


# A starter skills vocabulary. Expand this list with skills relevant to the
# job roles you're targeting (data science, web dev, etc.)
SKILLS_DB = [
    "python", "java", "c++", "c", "javascript", "typescript", "sql", "r",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy",
    "opencv", "django", "flask", "fastapi", "react", "node.js", "angular",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
    "html", "css", "mongodb", "mysql", "postgresql", "spark", "hadoop",
    "tableau", "power bi", "excel", "data analysis", "data visualization",
    "rest api", "microservices", "ci/cd", "agile", "scrum",
]


def extract_text_from_pdf(filepath: str) -> str:
    """Extract raw text from a PDF file."""
    text_chunks = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_text_from_docx(filepath: str) -> str:
    """Extract raw text from a DOCX file."""
    document = docx.Document(filepath)
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def extract_text(filepath: str) -> str:
    """Route to the correct extractor based on file extension."""
    lower = filepath.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(filepath)
    elif lower.endswith(".docx"):
        return extract_text_from_docx(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")


def extract_email(text: str) -> str:
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    # Matches common phone formats: +91 9876543210, (123) 456-7890, 123-456-7890 etc.
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}", text)
    return match.group(0).strip() if match else ""


def extract_name(text: str) -> str:
    """
    Naive heuristic: assume the candidate's name is on the first non-empty
    line of the resume (very common resume convention).
    For a more robust solution, use spaCy's NER (see README 'Upgrades' section).
    """
    for line in text.splitlines():
        line = line.strip()
        if line and len(line.split()) <= 4 and not extract_email(line):
            return line
    return "Unknown"


def extract_years_of_experience(text: str) -> float:
    """
    Looks for patterns like '3 years of experience', '5+ years', etc.
    Returns the largest number found (assume that's total experience).
    """
    matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", text, re.IGNORECASE)
    years = [float(m) for m in matches]
    return max(years) if years else 0.0


def extract_skills(text: str) -> list:
    """Match known skills vocabulary against resume text (case-insensitive)."""
    text_lower = text.lower()
    found = [skill for skill in SKILLS_DB if skill in text_lower]
    return sorted(set(found))


def extract_education(text: str) -> list:
    """
    Looks for common degree keywords. Extend this list for your use case.
    """
    degree_keywords = [
        "b.tech", "btech", "b.e", "bachelor", "m.tech", "mtech",
        "master", "phd", "diploma", "b.sc", "m.sc", "mba",
    ]
    text_lower = text.lower()
    found = [deg for deg in degree_keywords if deg in text_lower]
    return sorted(set(found))


def parse_resume(filepath: str) -> dict:
    """
    Main entry point: given a resume file path, return a structured dict.
    """
    text = extract_text(filepath)
    return {
        "filepath": filepath,
        "raw_text": text,
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "years_experience": extract_years_of_experience(text),
        "education": extract_education(text),
    }


if __name__ == "__main__":
    # Quick manual test - replace with a path to a real resume file
    import sys
    if len(sys.argv) > 1:
        result = parse_resume(sys.argv[1])
        import json
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Usage: python resume_parser.py <path_to_resume.pdf|docx>")
