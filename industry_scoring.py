"""
Industry-Specific Scoring Profiles
Provides customized scoring weights and requirements for different industries
"""
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class IndustryProfile:
    """Industry-specific scoring configuration"""
    name: str
    key_skills: List[str]
    critical_keywords: List[str]
    scoring_weights: Dict[str, float]
    min_experience_years: int
    preferred_certifications: List[str]
    industry_specific_metrics: List[str]


# Industry Profiles Database
INDUSTRY_PROFILES = {
    "software_engineering": IndustryProfile(
        name="Software Engineering",
        key_skills=[
            "python", "java", "javascript", "react", "node.js", "sql", 
            "git", "docker", "kubernetes", "aws", "azure", "api", "rest",
            "microservices", "agile", "ci/cd", "testing", "algorithms"
        ],
        critical_keywords=[
            "software development", "code", "programming", "development",
            "architecture", "deployment", "scalability", "performance"
        ],
        scoring_weights={
            "technical_skills": 0.40,
            "experience": 0.25,
            "projects": 0.20,
            "education": 0.10,
            "certifications": 0.05
        },
        min_experience_years=0,
        preferred_certifications=["AWS Certified", "Azure Certified", "Google Cloud"],
        industry_specific_metrics=["lines of code", "systems deployed", "users served", "latency reduction"]
    ),
    
    "data_science": IndustryProfile(
        name="Data Science & ML",
        key_skills=[
            "python", "r", "machine learning", "deep learning", "tensorflow", 
            "pytorch", "scikit-learn", "pandas", "numpy", "sql", "spark",
            "statistics", "data visualization", "tableau", "nlp", "computer vision",
            "feature engineering", "model deployment", "a/b testing"
        ],
        critical_keywords=[
            "data analysis", "predictive modeling", "statistical analysis",
            "machine learning", "ai", "data pipeline", "big data", "analytics"
        ],
        scoring_weights={
            "technical_skills": 0.35,
            "experience": 0.20,
            "projects": 0.25,
            "education": 0.15,
            "certifications": 0.05
        },
        min_experience_years=1,
        preferred_certifications=["AWS ML Specialty", "Google Data Engineer", "TensorFlow Developer"],
        industry_specific_metrics=["model accuracy", "precision/recall", "data processed", "insights delivered"]
    ),
    
    "devops": IndustryProfile(
        name="DevOps & SRE",
        key_skills=[
            "kubernetes", "docker", "jenkins", "ci/cd", "terraform", "ansible",
            "aws", "azure", "gcp", "monitoring", "prometheus", "grafana",
            "linux", "bash", "python", "git", "networking", "security"
        ],
        critical_keywords=[
            "infrastructure", "automation", "deployment", "orchestration",
            "monitoring", "reliability", "scalability", "uptime", "incident response"
        ],
        scoring_weights={
            "technical_skills": 0.40,
            "experience": 0.30,
            "projects": 0.15,
            "education": 0.05,
            "certifications": 0.10
        },
        min_experience_years=2,
        preferred_certifications=["CKA", "AWS DevOps", "HashiCorp Certified"],
        industry_specific_metrics=["uptime percentage", "deployment frequency", "mttr", "infrastructure cost savings"]
    ),
    
    "cybersecurity": IndustryProfile(
        name="Cybersecurity",
        key_skills=[
            "penetration testing", "vulnerability assessment", "network security",
            "cryptography", "siem", "firewall", "ids/ips", "security auditing",
            "compliance", "incident response", "malware analysis", "cloud security",
            "python", "linux", "wireshark", "metasploit", "kali linux"
        ],
        critical_keywords=[
            "security", "threat detection", "risk assessment", "compliance",
            "penetration testing", "vulnerability", "encryption", "authentication"
        ],
        scoring_weights={
            "technical_skills": 0.35,
            "experience": 0.25,
            "projects": 0.15,
            "education": 0.10,
            "certifications": 0.15
        },
        min_experience_years=1,
        preferred_certifications=["CISSP", "CEH", "OSCP", "Security+", "CISM"],
        industry_specific_metrics=["threats mitigated", "vulnerabilities patched", "compliance score", "incident response time"]
    ),
    
    "frontend": IndustryProfile(
        name="Frontend Development",
        key_skills=[
            "html", "css", "javascript", "typescript", "react", "vue", "angular",
            "redux", "webpack", "responsive design", "ui/ux", "accessibility",
            "performance optimization", "testing", "jest", "cypress", "sass"
        ],
        critical_keywords=[
            "web development", "user interface", "responsive", "frontend",
            "component", "spa", "pwa", "cross-browser", "mobile-first"
        ],
        scoring_weights={
            "technical_skills": 0.40,
            "experience": 0.20,
            "projects": 0.25,
            "education": 0.10,
            "certifications": 0.05
        },
        min_experience_years=0,
        preferred_certifications=["React Certified", "Google Mobile Web Specialist"],
        industry_specific_metrics=["page load time", "lighthouse score", "components built", "user engagement"]
    ),
    
    "backend": IndustryProfile(
        name="Backend Development",
        key_skills=[
            "python", "java", "node.js", "go", "rust", "sql", "nosql", "mongodb",
            "postgresql", "redis", "api design", "rest", "graphql", "microservices",
            "docker", "kubernetes", "aws", "authentication", "caching"
        ],
        critical_keywords=[
            "backend", "server-side", "api", "database", "scalability",
            "performance", "microservices", "architecture", "data modeling"
        ],
        scoring_weights={
            "technical_skills": 0.40,
            "experience": 0.25,
            "projects": 0.20,
            "education": 0.10,
            "certifications": 0.05
        },
        min_experience_years=1,
        preferred_certifications=["AWS Solutions Architect", "Oracle Certified"],
        industry_specific_metrics=["api response time", "throughput", "database optimization", "concurrent users"]
    ),
    
    "cloud_architecture": IndustryProfile(
        name="Cloud Architecture",
        key_skills=[
            "aws", "azure", "gcp", "cloud architecture", "serverless", "lambda",
            "s3", "ec2", "vpc", "iam", "cloudformation", "terraform",
            "cost optimization", "high availability", "disaster recovery"
        ],
        critical_keywords=[
            "cloud", "infrastructure", "scalability", "architecture", "migration",
            "cost optimization", "multi-cloud", "hybrid cloud", "cloud native"
        ],
        scoring_weights={
            "technical_skills": 0.35,
            "experience": 0.30,
            "projects": 0.20,
            "education": 0.05,
            "certifications": 0.10
        },
        min_experience_years=3,
        preferred_certifications=["AWS Solutions Architect", "Azure Architect", "GCP Professional"],
        industry_specific_metrics=["cost savings", "availability percentage", "migration success", "infrastructure as code"]
    ),
    
    "product_management": IndustryProfile(
        name="Product Management",
        key_skills=[
            "product strategy", "roadmap", "agile", "scrum", "jira", "user research",
            "data analysis", "stakeholder management", "a/b testing", "metrics",
            "wireframing", "user stories", "analytics", "sql", "market research"
        ],
        critical_keywords=[
            "product", "strategy", "roadmap", "features", "user experience",
            "metrics", "kpis", "stakeholder", "requirements", "prioritization"
        ],
        scoring_weights={
            "technical_skills": 0.20,
            "experience": 0.30,
            "projects": 0.25,
            "education": 0.15,
            "certifications": 0.10
        },
        min_experience_years=2,
        preferred_certifications=["CSPO", "PMP", "Pragmatic Marketing"],
        industry_specific_metrics=["products launched", "user growth", "revenue impact", "feature adoption"]
    ),
    
    "general": IndustryProfile(
        name="General/Tech",
        key_skills=[
            "communication", "problem-solving", "teamwork", "leadership",
            "project management", "analytical thinking", "adaptability"
        ],
        critical_keywords=[
            "experience", "projects", "achievements", "responsibilities", "results"
        ],
        scoring_weights={
            "technical_skills": 0.30,
            "experience": 0.25,
            "projects": 0.20,
            "education": 0.15,
            "certifications": 0.10
        },
        min_experience_years=0,
        preferred_certifications=[],
        industry_specific_metrics=["impact", "results", "achievements"]
    )
}


def detect_industry(resume_text: str, resume_skills: List[str]) -> str:
    """
    Automatically detect the most relevant industry from resume content
    Returns the industry key
    """
    resume_lower = resume_text.lower()
    skill_set = set(s.lower() for s in resume_skills)
    
    industry_scores = {}
    
    for industry_key, profile in INDUSTRY_PROFILES.items():
        if industry_key == "general":
            continue
            
        score = 0
        
        # Match key skills
        profile_skills = set(s.lower() for s in profile.key_skills)
        skill_matches = len(skill_set.intersection(profile_skills))
        score += skill_matches * 3
        
        # Match critical keywords
        for keyword in profile.critical_keywords:
            if keyword.lower() in resume_lower:
                score += 2
        
        # Check for certifications
        for cert in profile.preferred_certifications:
            if cert.lower() in resume_lower:
                score += 5
        
        industry_scores[industry_key] = score
    
    if not industry_scores or max(industry_scores.values()) < 5:
        return "general"
    
    return max(industry_scores, key=industry_scores.get)


def calculate_industry_score(
    resume_text: str,
    resume_skills: List[str],
    industry_key: Optional[str] = None,
    base_scores: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Calculate industry-specific score for a resume
    
    Args:
        resume_text: Full resume text
        resume_skills: Extracted skills list
        industry_key: Specific industry to score against (auto-detect if None)
        base_scores: Base scoring components (technical, experience, etc.)
    
    Returns:
        Dict with industry-specific analysis
    """
    # Auto-detect industry if not specified
    if industry_key is None:
        industry_key = detect_industry(resume_text, resume_skills)
    
    profile = INDUSTRY_PROFILES.get(industry_key, INDUSTRY_PROFILES["general"])
    
    resume_lower = resume_text.lower()
    skill_set = set(s.lower() for s in resume_skills)
    
    # Calculate skill coverage for this industry
    profile_skills = set(s.lower() for s in profile.key_skills)
    matching_skills = skill_set.intersection(profile_skills)
    skill_coverage = (len(matching_skills) / len(profile_skills) * 100) if profile_skills else 0
    
    # Check critical keywords
    keyword_matches = sum(1 for kw in profile.critical_keywords if kw.lower() in resume_lower)
    keyword_coverage = (keyword_matches / len(profile.critical_keywords) * 100) if profile.critical_keywords else 0
    
    # Check certifications
    cert_matches = [cert for cert in profile.preferred_certifications if cert.lower() in resume_lower]
    
    # Check industry-specific metrics
    metric_matches = sum(1 for metric in profile.industry_specific_metrics if metric.lower() in resume_lower)
    has_quantifiable_metrics = metric_matches > 0 or any(char in resume_text for char in ['%', '$'])
    
    # Calculate weighted industry score
    if base_scores:
        weighted_score = (
            base_scores.get("technical", 50) * profile.scoring_weights["technical_skills"] +
            base_scores.get("experience", 50) * profile.scoring_weights["experience"] +
            base_scores.get("projects", 50) * profile.scoring_weights["projects"] +
            base_scores.get("education", 50) * profile.scoring_weights["education"] +
            base_scores.get("certifications", 0) * profile.scoring_weights["certifications"]
        )
    else:
        # Use skill and keyword coverage as proxy
        weighted_score = (skill_coverage + keyword_coverage) / 2
    
    # Confidence score (how well does this resume match the industry)
    confidence = min(100, (skill_coverage * 0.6 + keyword_coverage * 0.3 + (len(cert_matches) * 3) * 0.1))
    
    return {
        "industry": profile.name,
        "industry_key": industry_key,
        "industry_score": round(weighted_score, 2),
        "confidence": round(confidence, 2),
        "skill_coverage": round(skill_coverage, 2),
        "keyword_coverage": round(keyword_coverage, 2),
        "matching_skills": list(matching_skills),
        "missing_key_skills": list(profile_skills - skill_set)[:10],
        "certifications_found": cert_matches,
        "recommended_certifications": profile.preferred_certifications[:3],
        "has_quantifiable_metrics": has_quantifiable_metrics,
        "scoring_weights": profile.scoring_weights,
        "industry_insights": {
            "min_experience_recommended": f"{profile.min_experience_years}+ years",
            "key_focus_areas": profile.critical_keywords[:5],
            "top_skills_to_add": list(profile_skills - skill_set)[:5]
        }
    }


def compare_across_industries(resume_text: str, resume_skills: List[str]) -> List[Dict]:
    """
    Compare resume against all industries and return ranked matches
    """
    results = []
    
    for industry_key in INDUSTRY_PROFILES.keys():
        if industry_key == "general":
            continue
        
        score_result = calculate_industry_score(
            resume_text=resume_text,
            resume_skills=resume_skills,
            industry_key=industry_key
        )
        results.append(score_result)
    
    # Sort by confidence (how well resume matches each industry)
    results.sort(key=lambda x: x["confidence"], reverse=True)
    
    return results
