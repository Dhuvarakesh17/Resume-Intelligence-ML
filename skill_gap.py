from typing import Dict, List


def analyze_skill_gap(resume_skills: List[str], job_skills: List[str]) -> Dict[str, object]:
    resume_set = set(resume_skills)
    job_set = set(job_skills)

    matching = sorted(resume_set.intersection(job_set))
    missing = sorted(job_set.difference(resume_set))

    if not job_set:
        pct = 0.0
    else:
        pct = round((len(matching) / len(job_set)) * 100, 2)

    return {
        "matching_skills": matching,
        "missing_skills": missing,
        "skill_match_percentage": pct,
    }
