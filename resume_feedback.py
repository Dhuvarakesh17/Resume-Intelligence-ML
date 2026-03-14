import re
from typing import List

ACTION_VERBS = {
    "built", "designed", "implemented", "developed", "optimized", "led", "improved", "automated"
}


def infer_experience_level(resume_text: str) -> str:
    years = [int(v) for v in re.findall(r"(\d+)\+?\s+years", resume_text.lower())]
    max_years = max(years) if years else 0

    if max_years >= 8:
        return "Senior"
    if max_years >= 3:
        return "Mid"
    if max_years > 0:
        return "Junior"
    return "Not specified"


def generate_resume_feedback(
    resume_text: str,
    missing_skills: List[str],
) -> List[str]:
    feedback: List[str] = []
    lowered = resume_text.lower()

    # Strong keyword recommendations
    if missing_skills:
        missing_str = ", ".join(missing_skills[:10])
        feedback.append(f"Add these high-impact keywords: {missing_str}")

    # Action verb improvement
    verb_hits = sum(1 for verb in ACTION_VERBS if verb in lowered)
    recommended_verbs = ["architected", "streamlined", "orchestrated", "amplified", "scaled", "accelerated"]
    if verb_hits < 3:
        verbs_hint = ", ".join(recommended_verbs[:3])
        feedback.append(
            f"Strengthen action verbs. Use: {verbs_hint} instead of basic terms."
        )

    # Metrics and quantifiable data
    metric_patterns = [
        (r"\b\d+%", "percentage metrics"),
        (r"\b\d+x\b", "multiple improvements"),
        (r"\$\d+", "revenue/cost data"),
        (r"\d+k\+?", "scale/reach metrics"),
    ]
    
    metrics_found = sum(1 for pattern, _ in metric_patterns if re.search(pattern, resume_text))
    if metrics_found < 2:
        feedback.append(
            f"Add quantifiable results: Use percentages (e.g., '50% faster'), multipliers (e.g., '3x improvement'), revenue impact, or scale metrics."
        )

    # Section checks
    sections_required = {
        "education": "Add Education section (degree, institution, graduation year).",
        "experience": "Add Experience/Work section with dates and company names.",
        "skills": "Add dedicated Skills section grouped by category (Languages, Frameworks, Tools).",
    }

    for section, recommendation in sections_required.items():
        if section not in lowered:
            feedback.append(recommendation)

    # Formatting check
    if "- " not in resume_text and "•" not in resume_text:
        feedback.append(
            "Use bullet points (- or •) for better readability and ATS parsing."
        )

    if not feedback:
        feedback.append("✓ Resume has strong ATS signals. Consider adding more industry certifications or speaking engagements for differentiation.")

    return feedback


def build_optimized_resume(
    resume_text: str,
    skills_found: List[str],
    missing_skills: List[str],
    recommended_roles: List[str],
) -> str:
    cleaned = re.sub(r"\s+", " ", resume_text).strip()

    summary_line = (
        f"Target Roles: {', '.join(recommended_roles)}. "
        f"Experienced professional focused on measurable delivery and production-ready solutions."
    )

    skills_line = "Core Skills: " + ", ".join(skills_found) if skills_found else "Core Skills: Add role-specific technical skills"

    keyword_line = (
        "Keywords to incorporate: " + ", ".join(missing_skills[:10])
        if missing_skills
        else "Keywords to incorporate: Resume already matches most target keywords"
    )

    optimized = (
        "PROFESSIONAL SUMMARY\n"
        + summary_line
        + "\n\n"
        + "TECHNICAL SKILLS\n"
        + skills_line
        + "\n\n"
        + "EXPERIENCE HIGHLIGHTS\n"
        + "- Built and delivered solutions aligned with business goals.\n"
        + "- Improved reliability, performance, and maintainability of core systems.\n"
        + "- Collaborated across teams to ship features on schedule.\n\n"
        + "ATS OPTIMIZATION NOTES\n"
        + keyword_line
        + "\n\n"
        + "ORIGINAL CONTENT (CONDENSED)\n"
        + cleaned
    )

    return optimized
