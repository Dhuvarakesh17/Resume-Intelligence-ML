"""
Resume Comparison Module
Provides side-by-side comparison of multiple resumes
"""
from typing import List, Dict, Any
from skill_extractor import extract_skills
from ml_scoring import compute_ml_ats_score
from job_recommender import recommend_roles
from resume_feedback import infer_experience_level
from industry_scoring import detect_industry, calculate_industry_score
import numpy as np


def extract_years_of_experience(resume_text: str) -> int:
    """
    Extract approximate years of experience from resume text
    """
    import re
    
    resume_lower = resume_text.lower()
    
    # Look for explicit year mentions
    patterns = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
        r'experience[:\s]+(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s+(?:in|with)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, resume_lower)
        if match:
            return int(match.group(1))
    
    # Infer from work history date ranges
    year_ranges = re.findall(r'(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|19\d{2}|present|current)', resume_lower)
    
    if year_ranges:
        total_years = 0
        current_year = 2026
        
        for start, end in year_ranges:
            start_year = int(start) if start.isdigit() else current_year
            end_year = current_year if end in ['present', 'current'] else int(end) if end.isdigit() else current_year
            total_years += max(0, end_year - start_year)
        
        return min(total_years, 50)  # Cap at 50 years
    
    # Default based on experience level
    level = infer_experience_level(resume_text)
    level_map = {
        "Junior": 1,
        "Mid": 4,
        "Senior": 8,
        "Not specified": 0
    }
    return level_map.get(level, 0)


def calculate_resume_metrics(resume_text: str, job_description: str = "") -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a single resume
    """
    # Extract skills
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description) if job_description else []
    
    # Calculate ATS score
    ats_score = compute_ml_ats_score(
        resume_text=resume_text,
        job_description=job_description,
        resume_skills=resume_skills,
        job_skills=job_skills
    )
    
    # Get role recommendations
    recommended_roles = recommend_roles(resume_text, resume_skills)
    predicted_role = recommended_roles[0] if recommended_roles else "General"
    
    # Experience analysis
    experience_level = infer_experience_level(resume_text)
    years_of_experience = extract_years_of_experience(resume_text)
    
    # Industry analysis
    industry_key = detect_industry(resume_text, resume_skills)
    industry_analysis = calculate_industry_score(resume_text, resume_skills, industry_key)
    
    # Calculate word count and density metrics
    word_count = len(resume_text.split())
    skill_density = (len(resume_skills) / word_count * 100) if word_count > 0 else 0
    
    # Check for important sections
    resume_lower = resume_text.lower()
    has_sections = {
        "education": any(keyword in resume_lower for keyword in ["education", "degree", "university", "college"]),
        "experience": any(keyword in resume_lower for keyword in ["experience", "work history", "employment"]),
        "projects": any(keyword in resume_lower for keyword in ["project", "portfolio", "built", "developed"]),
        "certifications": any(keyword in resume_lower for keyword in ["certification", "certified", "license"]),
        "achievements": any(keyword in resume_lower for keyword in ["achievement", "award", "recognition", "accomplishment"])
    }
    
    # Check for quantifiable impact
    has_metrics = any(char in resume_text for char in ['%', '$']) or \
                  any(word in resume_lower for word in ['increased', 'reduced', 'improved', 'saved', 'grew'])
    
    return {
        "ats_score": ats_score,
        "skills_found": resume_skills,
        "skill_count": len(resume_skills),
        "predicted_role": predicted_role,
        "recommended_roles": recommended_roles,
        "experience_level": experience_level,
        "years_of_experience": years_of_experience,
        "industry": industry_analysis["industry"],
        "industry_score": industry_analysis["industry_score"],
        "industry_confidence": industry_analysis["confidence"],
        "word_count": word_count,
        "skill_density": round(skill_density, 2),
        "sections_present": has_sections,
        "section_completeness": round(sum(has_sections.values()) / len(has_sections) * 100, 1),
        "has_quantifiable_metrics": has_metrics,
    }


def compare_resumes(
    resume_texts: List[str],
    job_description: str = "",
    names: List[str] = None
) -> Dict[str, Any]:
    """
    Compare multiple resumes side-by-side
    
    Args:
        resume_texts: List of resume text contents
        job_description: Optional job description for targeted comparison
        names: Optional list of names/identifiers for each resume
    
    Returns:
        Comprehensive comparison with rankings and insights
    """
    if not names:
        names = [f"Resume {i+1}" for i in range(len(resume_texts))]
    
    # Calculate metrics for each resume
    resume_metrics = []
    for i, resume_text in enumerate(resume_texts):
        metrics = calculate_resume_metrics(resume_text, job_description)
        metrics["name"] = names[i]
        metrics["index"] = i
        resume_metrics.append(metrics)
    
    # Rank by ATS score
    ranked_by_ats = sorted(resume_metrics, key=lambda x: x["ats_score"], reverse=True)
    
    # Calculate skill overlap matrix
    all_skills = set()
    for metrics in resume_metrics:
        all_skills.update(s.lower() for s in metrics["skills_found"])
    
    skill_overlap_matrix = []
    for metrics in resume_metrics:
        resume_skills = set(s.lower() for s in metrics["skills_found"])
        overlap_percentages = []
        for other_metrics in resume_metrics:
            other_skills = set(s.lower() for s in other_metrics["skills_found"])
            if resume_skills and other_skills:
                overlap = len(resume_skills.intersection(other_skills))
                percentage = (overlap / len(resume_skills.union(other_skills))) * 100
                overlap_percentages.append(round(percentage, 1))
            else:
                overlap_percentages.append(0.0)
        skill_overlap_matrix.append(overlap_percentages)
    
    # Find unique skills for each resume
    unique_skills = []
    for i, metrics in enumerate(resume_metrics):
        resume_skills = set(s.lower() for s in metrics["skills_found"])
        other_resumes_skills = set()
        for j, other_metrics in enumerate(resume_metrics):
            if i != j:
                other_resumes_skills.update(s.lower() for s in other_metrics["skills_found"])
        unique = resume_skills - other_resumes_skills
        unique_skills.append(list(unique))
    
    # Calculate statistics
    ats_scores = [m["ats_score"] for m in resume_metrics]
    skill_counts = [m["skill_count"] for m in resume_metrics]
    
    # Identify strengths and weaknesses
    comparisons = []
    for metrics in resume_metrics:
        strengths = []
        weaknesses = []
        
        # Check ATS score
        if metrics["ats_score"] >= 70:
            strengths.append(f"Strong ATS score ({metrics['ats_score']})")
        elif metrics["ats_score"] < 50:
            weaknesses.append(f"Low ATS score ({metrics['ats_score']})")
        
        # Check skill count
        if metrics["skill_count"] >= np.percentile(skill_counts, 75):
            strengths.append(f"Rich skill set ({metrics['skill_count']} skills)")
        elif metrics["skill_count"] <= np.percentile(skill_counts, 25):
            weaknesses.append(f"Limited skills ({metrics['skill_count']} skills)")
        
        # Check experience
        if metrics["years_of_experience"] >= 5:
            strengths.append(f"Experienced ({metrics['years_of_experience']} years)")
        elif metrics["years_of_experience"] < 2 and metrics["experience_level"] != "Not specified":
            weaknesses.append("Limited experience")
        
        # Check sections
        if metrics["section_completeness"] >= 80:
            strengths.append("Well-structured resume")
        elif metrics["section_completeness"] < 60:
            weaknesses.append("Missing key sections")
        
        # Check metrics
        if metrics["has_quantifiable_metrics"]:
            strengths.append("Includes quantifiable achievements")
        else:
            weaknesses.append("Lacks quantifiable metrics")
        
        comparisons.append({
            "name": metrics["name"],
            "strengths": strengths,
            "weaknesses": weaknesses,
            "unique_skills": unique_skills[metrics["index"]][:5]
        })
    
    # Generate recommendation
    top_resume = ranked_by_ats[0]
    recommendation = {
        "best_overall": top_resume["name"],
        "reason": f"Highest ATS score ({top_resume['ats_score']}) with {top_resume['skill_count']} relevant skills",
        "runner_up": ranked_by_ats[1]["name"] if len(ranked_by_ats) > 1 else None
    }
    
    return {
        "summary": {
            "total_resumes": len(resume_texts),
            "job_description_provided": bool(job_description),
            "total_unique_skills": len(all_skills),
            "average_ats_score": round(np.mean(ats_scores), 2),
            "ats_score_range": [int(min(ats_scores)), int(max(ats_scores))],
        },
        "rankings": {
            "by_ats_score": [
                {"rank": i+1, "name": m["name"], "score": m["ats_score"]}
                for i, m in enumerate(ranked_by_ats)
            ],
            "by_skill_count": [
                {"rank": i+1, "name": m["name"], "count": m["skill_count"]}
                for i, m in enumerate(sorted(resume_metrics, key=lambda x: x["skill_count"], reverse=True))
            ],
            "by_experience": [
                {"rank": i+1, "name": m["name"], "years": m["years_of_experience"]}
                for i, m in enumerate(sorted(resume_metrics, key=lambda x: x["years_of_experience"], reverse=True))
            ]
        },
        "detailed_metrics": resume_metrics,
        "skill_overlap_matrix": skill_overlap_matrix,
        "comparison_insights": comparisons,
        "recommendation": recommendation
    }


def find_best_candidate(
    resume_texts: List[str],
    job_description: str,
    criteria_weights: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Find the best candidate based on customizable criteria weights
    
    Args:
        resume_texts: List of resume texts
        job_description: Target job description
        criteria_weights: Custom weights for scoring criteria
            - ats_match: Weight for ATS/job match score
            - experience: Weight for years of experience
            - skills: Weight for skill count/diversity
            - completeness: Weight for resume completeness
    
    Returns:
        Best candidate analysis with detailed scoring breakdown
    """
    if criteria_weights is None:
        criteria_weights = {
            "ats_match": 0.40,
            "experience": 0.25,
            "skills": 0.20,
            "completeness": 0.15
        }
    
    # Normalize weights
    total_weight = sum(criteria_weights.values())
    criteria_weights = {k: v/total_weight for k, v in criteria_weights.items()}
    
    # Calculate metrics for all resumes
    all_metrics = [
        calculate_resume_metrics(resume_text, job_description)
        for resume_text in resume_texts
    ]
    
    # Normalize scores for fair comparison
    max_ats = max(m["ats_score"] for m in all_metrics) or 1
    max_exp = max(m["years_of_experience"] for m in all_metrics) or 1
    max_skills = max(m["skill_count"] for m in all_metrics) or 1
    
    # Calculate composite scores
    candidates = []
    for i, metrics in enumerate(all_metrics):
        ats_norm = (metrics["ats_score"] / max_ats) * 100
        exp_norm = (metrics["years_of_experience"] / max_exp) * 100
        skills_norm = (metrics["skill_count"] / max_skills) * 100
        completeness_norm = metrics["section_completeness"]
        
        composite_score = (
            ats_norm * criteria_weights["ats_match"] +
            exp_norm * criteria_weights["experience"] +
            skills_norm * criteria_weights["skills"] +
            completeness_norm * criteria_weights["completeness"]
        )
        
        candidates.append({
            "index": i,
            "composite_score": round(composite_score, 2),
            "breakdown": {
                "ats_match": round(ats_norm, 2),
                "experience": round(exp_norm, 2),
                "skills": round(skills_norm, 2),
                "completeness": round(completeness_norm, 2)
            },
            "metrics": metrics
        })
    
    # Sort by composite score
    candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    
    best = candidates[0]
    
    return {
        "best_candidate_index": best["index"],
        "composite_score": best["composite_score"],
        "score_breakdown": best["breakdown"],
        "all_candidates": candidates,
        "criteria_weights_used": criteria_weights,
        "key_differentiators": {
            "highest_ats": max(c["breakdown"]["ats_match"] for c in candidates),
            "most_experienced": max(c["breakdown"]["experience"] for c in candidates),
            "most_skills": max(c["breakdown"]["skills"] for c in candidates)
        }
    }
