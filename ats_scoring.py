import re
from typing import List, Optional

EXPERIENCE_KEYWORDS = {
    "experience", "years", "developed", "implemented", "designed", "managed", "led"
}

EDUCATION_KEYWORDS = {
    "bachelor", "master", "phd", "degree", "university", "college", "certification"
}

ATS_TERMS = {
    "api", "microservices", "cloud", "agile", "git", "ci/cd", "deployment", "testing"
}


def _keyword_density_score(text: str) -> float:
    words = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())
    if not words:
        return 0.0

    hits = sum(1 for w in words if w in ATS_TERMS)
    density = hits / len(words)

    # 5%+ density gives full marks for this sub-score.
    return min(1.0, density / 0.05)


def _keyword_presence_score(text: str, keywords: set) -> float:
    lowered = text.lower()
    if not keywords:
        return 0.0
    hit_count = sum(1 for kw in keywords if kw in lowered)
    return hit_count / len(keywords)


def compute_ats_score(
    resume_text: str,
    resume_skills: List[str],
    job_description: Optional[str] = None,
    job_skills: Optional[List[str]] = None,
) -> int:
    job_skills = job_skills or []

    # 1) Skill relevance (40)
    if job_skills:
        overlap = len(set(resume_skills).intersection(set(job_skills)))
        skill_relevance = overlap / max(1, len(set(job_skills)))
    else:
        skill_relevance = min(1.0, len(resume_skills) / 8)

    # 2) Keyword density (20)
    keyword_density = _keyword_density_score(resume_text)

    # 3) Experience keywords (15)
    experience_signal = _keyword_presence_score(resume_text, EXPERIENCE_KEYWORDS)

    # 4) Education keywords (15)
    education_signal = _keyword_presence_score(resume_text, EDUCATION_KEYWORDS)

    # 5) Job description match (10)
    if job_description:
        jd_words = set(re.findall(r"[a-zA-Z0-9+#.]+", job_description.lower()))
        resume_words = set(re.findall(r"[a-zA-Z0-9+#.]+", resume_text.lower()))
        jd_match = len(jd_words.intersection(resume_words)) / max(1, len(jd_words))
    else:
        jd_match = 0.5

    score = (
        skill_relevance * 40
        + keyword_density * 20
        + experience_signal * 15
        + education_signal * 15
        + jd_match * 10
    )

    return int(round(max(0, min(100, score))))
