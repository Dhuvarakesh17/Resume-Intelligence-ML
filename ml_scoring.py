from collections import Counter
import math
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from typing import Any, Dict, List, Optional, Tuple

from sentence_model import get_sentence_transformer

_EMBEDDER = None

_WORD_STOP_WORDS = set(ENGLISH_STOP_WORDS)
_ACTION_VERBS = {
    "accelerated", "achieved", "architected", "automated", "built", "created",
    "deployed", "designed", "developed", "drove", "engineered", "expanded",
    "implemented", "improved", "increased", "launched", "led", "managed",
    "migrated", "optimized", "orchestrated", "owned", "reduced", "scaled",
    "streamlined", "transformed"
}
_SECTION_KEYWORDS = {
    "summary": ["summary", "profile", "objective"],
    "experience": ["experience", "employment", "work history"],
    "education": ["education", "degree", "university", "college"],
    "skills": ["skills", "technical skills", "competencies", "technologies"],
    "projects": ["projects", "portfolio"],
}

_EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*years?\s+(?:of\s+)?experience",
    r"experience[:\s]+(\d+)\+?\s*years?",
    r"(\d+)\+?\s*yrs?\b",
]


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_text(text).lower()
    return re.findall(r"[a-z][a-z0-9.+#/-]{1,}", normalized)


def _extract_keywords(text: str, max_terms: int = 40) -> Counter:
    tokens = [
        token for token in _tokenize(text)
        if token not in _WORD_STOP_WORDS and not token.isdigit() and len(token) >= 3
    ]
    return Counter(dict(Counter(tokens).most_common(max_terms)))


def _normalize_skill(skill: str) -> str:
    lowered = skill.strip().lower()
    return re.sub(r"[^a-z0-9+#.]+", " ", lowered).strip()


def _order_skills_by_text_position(skills: List[str], source_text: str) -> List[str]:
    if not source_text:
        return skills

    lowered = source_text.lower()
    unique_skills = list(dict.fromkeys(skills))
    return sorted(
        unique_skills,
        key=lambda skill: (
            lowered.find(skill.lower()) if lowered.find(skill.lower()) >= 0 else 10**9,
            skill.lower(),
        ),
    )


def _token_set(value: str) -> set:
    return set(part for part in re.split(r"[^a-z0-9+#.]+", value) if part)


def _skills_match(job_skill: str, resume_skill: str) -> bool:
    if job_skill == resume_skill:
        return True

    if len(job_skill) >= 4 and (job_skill in resume_skill or resume_skill in job_skill):
        return True

    job_tokens = _token_set(job_skill)
    resume_tokens = _token_set(resume_skill)
    if not job_tokens or not resume_tokens:
        return False

    overlap = len(job_tokens.intersection(resume_tokens))
    return overlap > 0 and (overlap / max(len(job_tokens), len(resume_tokens))) >= 0.6


def _build_tfidf_vectorizer(analyzer: str = "word") -> TfidfVectorizer:
    if analyzer == "char_wb":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
        )

    return TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )


def _get_or_extract_skills(text: str, skills: Optional[List[str]]) -> List[str]:
    if skills:
        return skills

    if not text:
        return []

    try:
        from skill_extractor import extract_skills
        return extract_skills(text)
    except Exception:
        return []


def _compute_skill_alignment(resume_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
    normalized_resume = {}
    for skill in resume_skills:
        normalized = _normalize_skill(skill)
        if normalized:
            normalized_resume[normalized] = skill

    normalized_job = {}
    ordered_job_keys = []
    for skill in job_skills:
        normalized = _normalize_skill(skill)
        if normalized:
            normalized_job[normalized] = skill
            if normalized not in ordered_job_keys:
                ordered_job_keys.append(normalized)

    if not normalized_job:
        breadth_score = 0.0
        if normalized_resume:
            breadth_score = _clamp(math.log1p(len(normalized_resume)) / math.log1p(18))
        return {
            "score": breadth_score,
            "coverage": breadth_score,
            "precision": breadth_score,
            "matched_skills": sorted(normalized_resume.values()),
            "missing_skills": [],
            "exact_matches": [],
            "soft_matches": [],
            "critical_missing": [],
            "optional_missing": [],
            "must_have_score": breadth_score,
            "optional_score": breadth_score,
        }

    exact_matches = []
    soft_matches = []
    matched_resume_keys = set()

    for job_key, job_skill in normalized_job.items():
        if job_key in normalized_resume:
            exact_matches.append(job_skill)
            matched_resume_keys.add(job_key)
            continue

        for resume_key, resume_skill in normalized_resume.items():
            if resume_key in matched_resume_keys:
                continue
            if _skills_match(job_key, resume_key):
                soft_matches.append(job_skill)
                matched_resume_keys.add(resume_key)
                break

    matched_skills = sorted(set(exact_matches + soft_matches))
    missing_skills = sorted(skill for key, skill in normalized_job.items() if skill not in matched_skills)

    exact_weight = len(exact_matches)
    soft_weight = len(soft_matches) * 0.75
    coverage = (exact_weight + soft_weight) / max(1, len(normalized_job))
    precision = (exact_weight + soft_weight) / max(1, len(normalized_resume))

    must_have_keys = ordered_job_keys[: min(5, len(ordered_job_keys))]
    optional_keys = ordered_job_keys[len(must_have_keys):]
    matched_job_keys = {_normalize_skill(skill) for skill in matched_skills}

    must_have_matches = sum(1 for key in must_have_keys if key in matched_job_keys)
    optional_matches = sum(1 for key in optional_keys if key in matched_job_keys)
    must_have_score = must_have_matches / max(1, len(must_have_keys))
    optional_score = optional_matches / max(1, len(optional_keys)) if optional_keys else must_have_score
    tiered_coverage = must_have_score * 0.70 + optional_score * 0.30

    critical_missing = [normalized_job[key] for key in must_have_keys if key not in matched_job_keys]
    optional_missing = [normalized_job[key] for key in optional_keys if key not in matched_job_keys]

    return {
        "score": _clamp(tiered_coverage * 0.85 + precision * 0.15),
        "coverage": _clamp(coverage),
        "precision": _clamp(precision),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "exact_matches": sorted(exact_matches),
        "soft_matches": sorted(soft_matches),
        "critical_missing": critical_missing,
        "optional_missing": optional_missing,
        "must_have_score": _clamp(must_have_score),
        "optional_score": _clamp(optional_score),
    }


def _extract_years_experience(text: str) -> Optional[int]:
    lowered = text.lower()
    for pattern in _EXPERIENCE_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    return None


def _compute_experience_alignment(resume_text: str, job_description: str) -> Dict[str, Any]:
    resume_years = _extract_years_experience(resume_text)
    required_years = _extract_years_experience(job_description)

    if required_years is None:
        return {
            "score": 0.65 if resume_years is not None else 0.45,
            "resume_years": resume_years,
            "required_years": None,
        }

    if resume_years is None:
        return {
            "score": 0.25,
            "resume_years": None,
            "required_years": required_years,
        }

    if resume_years >= required_years:
        surplus = min(3, resume_years - required_years)
        return {
            "score": _clamp(0.85 + surplus * 0.05),
            "resume_years": resume_years,
            "required_years": required_years,
        }

    gap = required_years - resume_years
    return {
        "score": _clamp(1.0 - gap * 0.18),
        "resume_years": resume_years,
        "required_years": required_years,
    }


def _compute_quantification_signal(resume_text: str) -> Dict[str, Any]:
    patterns = {
        "percentages": r"\b\d+(?:\.\d+)?%",
        "currency": r"\$\s?\d+[\d,]*(?:\.\d+)?(?:\s?[kKmMbB])?",
        "scale": r"\b\d+(?:\.\d+)?\s?[xX]\b",
        "large_numbers": r"\b\d{2,}\b",
    }
    counts = {name: len(re.findall(pattern, resume_text)) for name, pattern in patterns.items()}
    total_hits = sum(counts.values())
    return {
        "score": min(1.0, total_hits / 6),
        "count": total_hits,
        "details": counts,
    }


def _compute_keyword_alignment(resume_text: str, job_description: str) -> Dict[str, float]:
    if not resume_text or not job_description:
        return {"score": 0.0, "precision": 0.0, "recall": 0.0}

    resume_keywords = _extract_keywords(resume_text)
    job_keywords = _extract_keywords(job_description)
    if not job_keywords:
        return {"score": 0.0, "precision": 0.0, "recall": 0.0}

    matched_weight = sum(min(job_keywords[word], resume_keywords.get(word, 0)) for word in job_keywords)
    recall = matched_weight / max(1, sum(job_keywords.values()))
    precision = matched_weight / max(1, sum(resume_keywords.values()))

    if precision + recall == 0:
        score = 0.0
    else:
        score = (2 * precision * recall) / (precision + recall)

    return {
        "score": _clamp(score),
        "precision": _clamp(precision),
        "recall": _clamp(recall),
    }


def _compute_tfidf_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0

    try:
        vectorizer = _build_tfidf_vectorizer()
        matrix = vectorizer.fit_transform([text1, text2])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except Exception:
        return 0.0


def _compute_char_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0

    try:
        vectorizer = _build_tfidf_vectorizer(analyzer="char_wb")
        matrix = vectorizer.fit_transform([text1, text2])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except Exception:
        return 0.0


def _compute_lexical_alignment(resume_text: str, job_description: str) -> Dict[str, float]:
    word_similarity = _compute_tfidf_similarity(resume_text, job_description)
    char_similarity = _compute_char_similarity(resume_text, job_description)
    score = _clamp(word_similarity * 0.75 + char_similarity * 0.25)
    return {
        "score": score,
        "word_similarity": _clamp(word_similarity),
        "char_similarity": _clamp(char_similarity),
    }


def _compute_semantic_alignment(resume_text: str, job_description: str, lexical_score: float) -> Dict[str, Any]:
    if not resume_text or not job_description:
        return {"score": 0.0, "source": "none"}

    embedder = _get_embedder()
    if embedder is None:
        return {"score": _clamp(lexical_score * 0.9), "source": "lexical_fallback"}

    try:
        embeddings = embedder.encode(
            [resume_text, job_description],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return {"score": _clamp(similarity), "source": "sentence_transformer"}
    except Exception:
        return {"score": _clamp(lexical_score * 0.9), "source": "lexical_fallback"}


def _compute_resume_quality(resume_text: str) -> Dict[str, float]:
    if not resume_text:
        return {
            "score": 0.0,
            "contact": 0.0,
            "sections": 0.0,
            "action_verbs": 0.0,
            "structure": 0.0,
            "length": 0.0,
            "readability": 0.0,
        }

    normalized = resume_text.lower()
    word_count = len(_tokenize(resume_text))

    contact = 1.0 if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", resume_text) or re.search(r"\+?\d[\d\s().-]{7,}\d", resume_text) else 0.0
    sections = sum(any(keyword in normalized for keyword in keywords) for keywords in _SECTION_KEYWORDS.values()) / len(_SECTION_KEYWORDS)
    action_verbs = min(1.0, sum(1 for verb in _ACTION_VERBS if verb in normalized) / 6)
    bullet_lines = len(re.findall(r"(?m)^\s*(?:[-*•▪])\s+", resume_text))
    structure = min(1.0, bullet_lines / 6)
    sentences = [segment.strip() for segment in re.split(r"[.!?]+", resume_text) if segment.strip()]
    avg_sentence_length = word_count / max(1, len(sentences))
    if 8 <= avg_sentence_length <= 24:
        readability = 1.0
    elif 5 <= avg_sentence_length < 8 or 24 < avg_sentence_length <= 32:
        readability = 0.75
    else:
        readability = 0.45

    if 250 <= word_count <= 900:
        length = 1.0
    elif 180 <= word_count < 250 or 900 < word_count <= 1100:
        length = 0.75
    elif 120 <= word_count < 180:
        length = 0.55
    else:
        length = 0.35 if word_count > 0 else 0.0

    score = _clamp(
        contact * 0.10
        + sections * 0.30
        + action_verbs * 0.20
        + structure * 0.15
        + readability * 0.10
        + length * 0.15
    )

    return {
        "score": score,
        "contact": contact,
        "sections": sections,
        "action_verbs": action_verbs,
        "structure": structure,
        "length": length,
        "readability": readability,
    }


def _resolve_component_weights(has_job_description: bool, has_job_skills: bool) -> Dict[str, float]:
    if has_job_description and has_job_skills:
        return {
            "keyword_alignment": 0.30,
            "semantic_alignment": 0.25,
            "skill_alignment": 0.20,
            "experience_alignment": 0.10,
            "resume_quality": 0.10,
            "quantification": 0.05,
        }

    if has_job_description:
        return {
            "keyword_alignment": 0.34,
            "semantic_alignment": 0.28,
            "skill_alignment": 0.10,
            "experience_alignment": 0.13,
            "resume_quality": 0.10,
            "quantification": 0.05,
        }

    if has_job_skills:
        return {
            "keyword_alignment": 0.0,
            "semantic_alignment": 0.0,
            "skill_alignment": 0.70,
            "experience_alignment": 0.10,
            "resume_quality": 0.15,
            "quantification": 0.05,
        }

    return {
        "keyword_alignment": 0.0,
        "semantic_alignment": 0.0,
        "skill_alignment": 0.45,
        "experience_alignment": 0.15,
        "resume_quality": 0.30,
        "quantification": 0.10,
    }


def _compute_confidence(
    resume_text: str,
    job_description: str,
    resume_skills: List[str],
    job_skills: List[str],
    semantic_source: str,
) -> float:
    factors = [min(1.0, len(_tokenize(resume_text)) / 250)]

    if job_description:
        factors.append(min(1.0, len(_tokenize(job_description)) / 150))
    else:
        factors.append(0.55)

    factors.append(min(1.0, len(resume_skills) / 12) if resume_skills else 0.35)
    factors.append(min(1.0, len(job_skills) / 10) if job_skills else 0.55)
    factors.append(1.0 if semantic_source == "sentence_transformer" else 0.75 if job_description else 0.6)

    return round(_clamp(sum(factors) / len(factors)) * 100, 2)


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    try:
        _EMBEDDER = get_sentence_transformer()
    except Exception:
        _EMBEDDER = None

    return _EMBEDDER


def compute_tfidf_similarity(text1: str, text2: str) -> float:
    """
    Compute TF-IDF based similarity between two texts using cosine similarity.
    Returns a score between 0 and 1.
    """
    return _compute_tfidf_similarity(text1, text2)


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity using sentence transformers embeddings.
    Returns a score between 0 and 1.
    """
    lexical_alignment = _compute_lexical_alignment(text1, text2)
    return _compute_semantic_alignment(text1, text2, lexical_alignment["score"])["score"]


def compute_ml_ats_details(
    resume_text: str,
    job_description: Optional[str] = None,
    resume_skills: Optional[List[str]] = None,
    job_skills: Optional[List[str]] = None,
) -> Dict[str, Any]:
    resume_text = _normalize_text(resume_text)
    job_description = _normalize_text(job_description)
    resume_skills = _get_or_extract_skills(resume_text, resume_skills)
    job_skills = _get_or_extract_skills(job_description, job_skills)
    job_skills = _order_skills_by_text_position(job_skills, job_description)

    skill_alignment = _compute_skill_alignment(resume_skills, job_skills)
    keyword_alignment = _compute_keyword_alignment(resume_text, job_description)
    lexical_alignment = _compute_lexical_alignment(resume_text, job_description)
    semantic_alignment = _compute_semantic_alignment(resume_text, job_description, lexical_alignment["score"])
    experience_alignment = _compute_experience_alignment(resume_text, job_description)
    quantification = _compute_quantification_signal(resume_text)
    resume_quality = _compute_resume_quality(resume_text)

    weights = _resolve_component_weights(bool(job_description), bool(job_skills))
    component_scores = {
        "skill_alignment": skill_alignment["score"],
        "keyword_alignment": keyword_alignment["score"],
        "semantic_alignment": semantic_alignment["score"],
        "experience_alignment": experience_alignment["score"],
        "resume_quality": resume_quality["score"],
        "quantification": quantification["score"],
    }

    final_score = sum(component_scores[name] * weight for name, weight in weights.items())
    confidence = _compute_confidence(
        resume_text=resume_text,
        job_description=job_description,
        resume_skills=resume_skills,
        job_skills=job_skills,
        semantic_source=semantic_alignment["source"],
    )

    return {
        "score": int(round(_clamp(final_score) * 100)),
        "score_float": round(_clamp(final_score) * 100, 2),
        "confidence": confidence,
        "matched_skills": skill_alignment["matched_skills"],
        "missing_skills": skill_alignment["missing_skills"],
        "critical_missing_skills": skill_alignment["critical_missing"],
        "optional_missing_skills": skill_alignment["optional_missing"],
        "components": {
            "skill_alignment": {
                "score": round(skill_alignment["score"] * 100, 2),
                "weight": weights["skill_alignment"],
                "coverage": round(skill_alignment["coverage"] * 100, 2),
                "precision": round(skill_alignment["precision"] * 100, 2),
                "must_have_score": round(skill_alignment["must_have_score"] * 100, 2),
                "optional_score": round(skill_alignment["optional_score"] * 100, 2),
                "exact_matches": skill_alignment["exact_matches"],
                "soft_matches": skill_alignment["soft_matches"],
                "critical_missing": skill_alignment["critical_missing"],
                "optional_missing": skill_alignment["optional_missing"],
            },
            "keyword_alignment": {
                "score": round(keyword_alignment["score"] * 100, 2),
                "weight": weights["keyword_alignment"],
                "precision": round(keyword_alignment["precision"] * 100, 2),
                "recall": round(keyword_alignment["recall"] * 100, 2),
            },
            "semantic_alignment": {
                "score": round(semantic_alignment["score"] * 100, 2),
                "weight": weights["semantic_alignment"],
                "source": semantic_alignment["source"],
                "lexical_backstop": round(lexical_alignment["score"] * 100, 2),
            },
            "experience_alignment": {
                "score": round(experience_alignment["score"] * 100, 2),
                "weight": weights["experience_alignment"],
                "resume_years": experience_alignment["resume_years"],
                "required_years": experience_alignment["required_years"],
            },
            "resume_quality": {
                "score": round(resume_quality["score"] * 100, 2),
                "weight": weights["resume_quality"],
                "contact": round(resume_quality["contact"] * 100, 2),
                "sections": round(resume_quality["sections"] * 100, 2),
                "action_verbs": round(resume_quality["action_verbs"] * 100, 2),
                "structure": round(resume_quality["structure"] * 100, 2),
                "length": round(resume_quality["length"] * 100, 2),
                "readability": round(resume_quality["readability"] * 100, 2),
            },
            "quantification": {
                "score": round(quantification["score"] * 100, 2),
                "weight": weights["quantification"],
                "count": quantification["count"],
                "details": quantification["details"],
            },
        },
    }


def compute_ml_ats_score(
    resume_text: str,
    job_description: Optional[str] = None,
    resume_skills: Optional[List[str]] = None,
    job_skills: Optional[List[str]] = None,
) -> int:
    """
    Compute ATS score using ML models (TF-IDF, semantic similarity, and skill matching).
    Returns a score between 0 and 100.
    """
    details = compute_ml_ats_details(
        resume_text=resume_text,
        job_description=job_description,
        resume_skills=resume_skills,
        job_skills=job_skills,
    )
    return details["score"]


def rank_resumes_by_job(
    resumes: List[str],
    job_description: str,
    resume_skills_list: Optional[List[List[str]]] = None,
) -> List[Tuple[int, float]]:
    """
    Rank a list of resumes against a job description.
    Returns list of (index, score) tuples sorted by score descending.
    """
    scores = []
    job_skills = _get_or_extract_skills(job_description, None) if job_description else []

    for idx, resume in enumerate(resumes):
        resume_skills = resume_skills_list[idx] if resume_skills_list else None

        score = compute_ml_ats_score(
            resume_text=resume,
            job_description=job_description,
            resume_skills=resume_skills,
            job_skills=job_skills,
        )
        scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
