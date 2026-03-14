from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sentence_model import get_sentence_transformer

ROLE_SKILL_MAP: Dict[str, List[str]] = {
    "Data Scientist": ["Python", "SQL", "Machine Learning", "Data Analysis"],
    "Machine Learning Engineer": [
        "Python", "Machine Learning", "Deep Learning", "TensorFlow", "Docker", "Kubernetes"
    ],
    "Data Analyst": ["SQL", "Python", "Data Analysis"],
    "Backend Developer": ["Java", "Spring Boot", "SQL", "Docker"],
    "AI Engineer": ["Python", "Machine Learning", "Deep Learning", "TensorFlow"],
    "Software Engineer": ["Python", "Java", "SQL", "Docker", "Kubernetes", "React"],
}

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    try:
        _MODEL = get_sentence_transformer()
    except Exception:
        _MODEL = None

    return _MODEL


def _keyword_role_scores(resume_skills: List[str]) -> List[Tuple[str, float]]:
    skill_set = set(resume_skills)
    scores = []

    for role, required in ROLE_SKILL_MAP.items():
        overlap = len(skill_set.intersection(set(required)))
        score = overlap / max(1, len(required))
        scores.append((role, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def recommend_roles(resume_text: str, resume_skills: List[str], top_k: int = 3) -> List[str]:
    model = _get_model()

    if model is None:
        ranked = _keyword_role_scores(resume_skills)
        return [role for role, score in ranked[:top_k] if score > 0] or ["Software Engineer"]

    resume_profile = resume_text + "\nSkills: " + ", ".join(resume_skills)
    role_profiles = [
        f"{role}. Key skills: {', '.join(skills)}"
        for role, skills in ROLE_SKILL_MAP.items()
    ]

    resume_vec = model.encode([resume_profile], convert_to_numpy=True)
    role_vecs = model.encode(role_profiles, convert_to_numpy=True)

    sim = cosine_similarity(np.array(resume_vec), np.array(role_vecs))[0]

    ranked_idx = np.argsort(sim)[::-1][:top_k]
    roles = list(ROLE_SKILL_MAP.keys())

    return [roles[i] for i in ranked_idx]
