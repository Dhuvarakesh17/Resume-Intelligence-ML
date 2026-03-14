from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml_scoring import compute_ml_ats_details, compute_ml_ats_score
from job_recommender import ROLE_SKILL_MAP, recommend_roles
from resume_feedback import (
    build_optimized_resume,
    generate_resume_feedback,
    infer_experience_level,
)
from skill_extractor import extract_skills, get_skill_dictionary
from word_quality_analyzer import analyze_word_quality, calculate_professionalism_score
from advanced_word_analyzer import analyze_with_ml

# Enhanced ML Service imports
from enhanced_ml_scoring import get_enhanced_scorer
from industry_scoring import (
    detect_industry,
    calculate_industry_score,
    compare_across_industries,
    INDUSTRY_PROFILES
)
from resume_comparison import compare_resumes, find_best_candidate
from batch_processing import get_batch_processor

app = FastAPI(
    title="Resume Analysis ML Service - Enhanced",
    version="2.0.0",
    description="Advanced ML-powered resume analysis with industry-specific scoring, batch processing, and confidence metrics"
)

# Enable CORS for frontend at localhost:8080
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)
    job_description: Optional[str] = ""


class AnalyzeResponse(BaseModel):
    # Java backend compatibility fields.
    overallScore: float
    semanticScore: float
    skillCoverage: float
    experienceScore: int
    predictedRole: str
    missingSkills: List[str]

    # Extended API fields.
    ats_score: int
    skills_found: List[str]
    matching_skills: List[str]
    missing_skills: List[str]
    critical_missing_skills: List[str]
    optional_missing_skills: List[str]
    skill_match_percentage: float
    recommended_roles: List[str]
    experience_level: str
    resume_feedback: List[str]
    recommended_keywords: List[str]
    optimized_resume: str
    
    # Word quality analysis fields (production-ready ML layer)
    professionalism_score: float
    word_quality_analysis: Dict[str, Any]
    ml_confidence: float
    ml_breakdown: Dict[str, Any]


def _experience_score_from_level(experience_level: str) -> int:
    mapping = {
        "Junior": 45,
        "Mid": 70,
        "Senior": 90,
        "Not specified": 30,
    }
    return mapping.get(experience_level, 30)


def _semantic_score_for_role(predicted_role: str, resume_skills: List[str]) -> float:
    required_skills = ROLE_SKILL_MAP.get(predicted_role, [])
    if not required_skills:
        return 0.0

    overlap = len(set(required_skills).intersection(set(resume_skills)))
    return round((overlap / len(required_skills)) * 100, 2)


def _generate_recommended_keywords(missing_skills: List[str], job_description: str, resume_text: str) -> List[str]:
    """Generate specific keywords to add to resume for improvement."""
    recommended = []

    # Add missing skills as keywords
    if missing_skills:
        recommended.extend(missing_skills[:5])

    # Extract industry keywords from job description
    job_keywords = ["cloud", "api", "database", "microservices", "testing", "deployment", "ci/cd", "agile", "scrum"]
    job_lower = job_description.lower()
    for keyword in job_keywords:
        if keyword in job_lower and keyword not in resume_text.lower():
            recommended.append(keyword)

    # Add soft skills if not present
    soft_skills = ["leadership", "communication", "collaboration", "problem-solving", "mentoring"]
    resume_lower = resume_text.lower()
    for skill in soft_skills:
        if skill not in resume_lower:
            recommended.append(skill)

    # Add impact metrics if not present
    if not any(metric in resume_text for metric in ["%", "x", "$", "k]"]):
        recommended.extend(["increased ROI", "reduced cost", "improved performance", "scaled infrastructure"])

    return list(dict.fromkeys(recommended))[:10]  # Remove duplicates and limit to 10


@app.get("/health")
def health() -> dict:
    skill_count = len(get_skill_dictionary())
    return {"status": "ok", "skills_loaded": skill_count}


@app.post("/skills/refresh")
def refresh_skills() -> dict:
    skills = get_skill_dictionary(force_refresh=True)
    return {
        "status": "refreshed",
        "skills_loaded": len(skills),
    }


class WordQualityRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)


@app.post("/word-quality")
def analyze_word_quality_endpoint(payload: WordQualityRequest) -> Dict[str, Any]:
    """Production-ready ML endpoint for analyzing resume word quality."""
    word_quality = analyze_word_quality(payload.resume_text)
    professionalism = calculate_professionalism_score(payload.resume_text)
    
    return {
        "professionalism_score": professionalism,
        "word_quality_analysis": word_quality,
        "status": "analyzed",
    }


class AdvancedAnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)
    job_description: Optional[str] = Field(None, description="Optional JD for context-aware analysis")


@app.post("/analyze-ml")
def analyze_with_advanced_ml(payload: AdvancedAnalyzeRequest) -> Dict[str, Any]:
    """
    Advanced ML Analysis Endpoint
    
    Uses state-of-the-art ML/NLP techniques:
    - Transformer embeddings for semantic understanding
    - NER for entity extraction
    - Context-aware weak word detection
    - Industry-specific vocabulary matching
    - Quantifiable achievement detection
    - Reinforcement learning-ready architecture
    """
    analysis = analyze_with_ml(payload.resume_text, payload.job_description)
    
    return {
        "status": "analyzed",
        "ml_analysis": analysis,
        "model_info": {
            "type": "Transformer-based (BERT embeddings) + spaCy NLP",
            "features": [
                "semantic_embeddings",
                "named_entity_recognition",
                "context_aware_detection",
                "vocabulary_richness_analysis",
                "verb_strength_ml_scoring",
                "quantification_detection",
                "industry_vocabulary_matching",
                "statistical_text_features"
            ]
        }
    }


class RankRequest(BaseModel):
    resumes: List[str] = Field(..., description="List of resume texts to rank")
    job_description: str = Field(..., description="Target job description")


class RankResult(BaseModel):
    resume_index: int
    score: int
    rank: int


class RankResponse(BaseModel):
    results: List[RankResult]
    top_score: int
    average_score: float


@app.post("/rank", response_model=RankResponse)
def rank_resumes(payload: RankRequest) -> RankResponse:
    """
    Rank multiple resumes against a job description using ML models.
    Returns ranked list with scores.
    """
    from ml_scoring import rank_resumes_by_job
    
    ranked = rank_resumes_by_job(payload.resumes, payload.job_description)
    
    results = []
    for rank, (idx, score) in enumerate(ranked, 1):
        results.append(
            RankResult(
                resume_index=idx,
                score=score,
                rank=rank,
            )
        )
    
    scores = [score for _, score in ranked]
    top_score = scores[0] if scores else 0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    return RankResponse(
        results=results,
        top_score=top_score,
        average_score=round(avg_score, 2),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    resume_text = payload.resume_text
    job_description = payload.job_description or ""

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description) if job_description else []

    ml_analysis = compute_ml_ats_details(
        resume_text=resume_text,
        job_description=job_description,
        resume_skills=resume_skills,
        job_skills=job_skills,
    )
    ats_score = ml_analysis["score"]
    matched_skills = ml_analysis["matched_skills"]
    missing_skills = ml_analysis["missing_skills"]

    recommended_roles = recommend_roles(resume_text, resume_skills)
    predicted_role = recommended_roles[0] if recommended_roles else "Software Engineer"

    experience_level = infer_experience_level(resume_text)
    experience_score = _experience_score_from_level(experience_level)

    semantic_score = ml_analysis["components"]["semantic_alignment"]["score"]

    if job_skills:
        skill_coverage = ml_analysis["components"]["skill_alignment"]["coverage"]
    else:
        skill_coverage = ml_analysis["components"]["skill_alignment"]["score"]

    word_quality = analyze_word_quality(resume_text)
    professionalism_score = calculate_professionalism_score(resume_text)

    overall_score = float(ats_score)

    resume_feedback = generate_resume_feedback(
        resume_text=resume_text,
        missing_skills=missing_skills,
    )
    
    recommended_keywords = _generate_recommended_keywords(
        missing_skills=missing_skills,
        job_description=job_description,
        resume_text=resume_text,
    )
    
    optimized_resume = build_optimized_resume(
        resume_text=resume_text,
        skills_found=resume_skills,
        missing_skills=missing_skills,
        recommended_roles=recommended_roles,
    )

    skill_match_percentage = round(
        ml_analysis["components"]["skill_alignment"].get("coverage", 0.0),
        2,
    ) if job_skills else round(skill_coverage, 2)
    
    return AnalyzeResponse(
        overallScore=overall_score,
        semanticScore=semantic_score,
        skillCoverage=skill_coverage,
        experienceScore=experience_score,
        predictedRole=predicted_role,
        missingSkills=missing_skills,
        ats_score=ats_score,
        skills_found=resume_skills,
        matching_skills=matched_skills,
        missing_skills=missing_skills,
        critical_missing_skills=ml_analysis["critical_missing_skills"],
        optional_missing_skills=ml_analysis["optional_missing_skills"],
        skill_match_percentage=skill_match_percentage,
        recommended_roles=recommended_roles,
        experience_level=experience_level,
        resume_feedback=resume_feedback,
        recommended_keywords=recommended_keywords,
        optimized_resume=optimized_resume,
        professionalism_score=professionalism_score,
        word_quality_analysis=word_quality,
        ml_confidence=ml_analysis["confidence"],
        ml_breakdown=ml_analysis["components"],
    )


# ==================== ENHANCED ML ENDPOINTS (V2.0) ====================

@app.post("/analyze-enhanced")
def analyze_enhanced(payload: AnalyzeRequest) -> Dict[str, Any]:
    """
    Enhanced ML Analysis with Confidence Scoring
    
    Provides comprehensive analysis with:
    - Confidence metrics for all scores
    - Detailed component breakdown
    - Industry-specific scoring
    - Actionable recommendations with priority
    """
    resume_text = payload.resume_text
    job_description = payload.job_description or ""
    
    # Extract skills
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description) if job_description else []
    
    # Enhanced ML scoring with confidence
    scorer = get_enhanced_scorer()
    enhanced_analysis = scorer.compute_comprehensive_score(
        resume_text=resume_text,
        job_description=job_description,
        resume_skills=resume_skills,
        job_skills=job_skills
    )
    
    # Industry analysis
    industry_key = detect_industry(resume_text, resume_skills)
    industry_analysis = calculate_industry_score(
        resume_text=resume_text,
        resume_skills=resume_skills,
        industry_key=industry_key
    )
    
    # Role recommendations
    recommended_roles = recommend_roles(resume_text, resume_skills)
    experience_level = infer_experience_level(resume_text)
    
    return {
        "status": "success",
        "final_score": enhanced_analysis["final_score"],
        "confidence": enhanced_analysis["confidence"],
        "interpretation": enhanced_analysis["interpretation"],
        "components": enhanced_analysis["components"],
        "recommendations": enhanced_analysis["top_recommendations"],
        "industry_analysis": industry_analysis,
        "predicted_role": recommended_roles[0] if recommended_roles else "General",
        "recommended_roles": recommended_roles,
        "experience_level": experience_level,
        "skills_found": resume_skills,
        "model_version": "2.0-enhanced"
    }


@app.post("/industry-score")
def analyze_industry_specific(payload: AnalyzeRequest) -> Dict[str, Any]:
    """
    Industry-Specific Resume Scoring
    
    Automatically detects the best-fitting industry and provides
    customized scoring with industry-specific requirements.
    """
    resume_text = payload.resume_text
    resume_skills = extract_skills(resume_text)
    
    # Detect industry or use specific one
    industry_key = detect_industry(resume_text, resume_skills)
    industry_analysis = calculate_industry_score(
        resume_text=resume_text,
        resume_skills=resume_skills,
        industry_key=industry_key
    )
    
    # Compare across all industries
    all_industries = compare_across_industries(resume_text, resume_skills)
    
    return {
        "status": "success",
        "best_fit_industry": industry_analysis,
        "all_industry_matches": all_industries[:5],  # Top 5 matches
        "available_industries": list(INDUSTRY_PROFILES.keys())
    }


class CompareResumesRequest(BaseModel):
    resumes: List[str] = Field(..., description="List of resume texts to compare")
    job_description: str = Field("", description="Optional job description for comparison")
    names: Optional[List[str]] = Field(None, description="Optional names/IDs for each resume")


@app.post("/compare-resumes")
def compare_multiple_resumes(payload: CompareResumesRequest) -> Dict[str, Any]:
    """
    Side-by-Side Resume Comparison
    
    Compare multiple resumes with:
    - Skill overlap analysis
    - Ranking by multiple criteria
    - Strengths and weaknesses for each
    - Best candidate recommendation
    """
    result = compare_resumes(
        resume_texts=payload.resumes,
        job_description=payload.job_description,
        names=payload.names
    )
    
    return result


class FindBestCandidateRequest(BaseModel):
    resumes: List[str] = Field(..., description="List of resume texts")
    job_description: str = Field(..., description="Job description to match against")
    criteria_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Custom weights: ats_match, experience, skills, completeness"
    )


@app.post("/find-best-candidate")
def find_best_candidate_endpoint(payload: FindBestCandidateRequest) -> Dict[str, Any]:
    """
    Find Best Candidate with Customizable Criteria
    
    Uses customizable weights to find the best candidate based on:
    - ATS/job match score
    - Years of experience
    - Skill count and diversity
    - Resume completeness
    """
    result = find_best_candidate(
        resume_texts=payload.resumes,
        job_description=payload.job_description,
        criteria_weights=payload.criteria_weights
    )
    
    return result


class BatchProcessRequest(BaseModel):
    resumes: List[Dict[str, str]] = Field(
        ...,
        description="List of resumes with 'text' and optional 'id' keys"
    )
    job_description: str = Field("", description="Optional job description")
    include_detailed_analysis: bool = Field(
        False,
        description="Include detailed ML analysis (slower but more comprehensive)"
    )
    max_workers: int = Field(4, ge=1, le=10, description="Parallel processing workers")


@app.post("/batch-process")
def batch_process_resumes(payload: BatchProcessRequest) -> Dict[str, Any]:
    """
    Batch Process Multiple Resumes
    
    High-performance parallel processing for multiple resumes with:
    - Concurrent processing (configurable workers)
    - Aggregate statistics
    - Error handling per resume
    - Performance metrics
    """
    processor = get_batch_processor(max_workers=payload.max_workers)
    
    result = processor.process_batch(
        resumes=payload.resumes,
        job_description=payload.job_description,
        include_detailed_analysis=payload.include_detailed_analysis
    )
    
    return result


class SmartRankRequest(BaseModel):
    resumes: List[Dict[str, str]] = Field(..., description="Resumes to rank")
    job_description: str = Field(..., description="Target job description")
    ranking_criteria: str = Field(
        "ats_score",
        description="Criteria: 'ats_score', 'enhanced_score', or 'industry_score'"
    )
    max_workers: int = Field(4, ge=1, le=10)


@app.post("/smart-rank")
def smart_rank_resumes(payload: SmartRankRequest) -> Dict[str, Any]:
    """
    Smart Resume Ranking
    
    Processes and ranks resumes by customizable criteria:
    - ats_score: Traditional ATS matching
    - enhanced_score: ML-enhanced comprehensive scoring
    - industry_score: Industry-specific scoring
    
    Returns detailed rankings with insights.
    """
    processor = get_batch_processor(max_workers=payload.max_workers)
    
    result = processor.smart_rank_resumes(
        resumes=payload.resumes,
        job_description=payload.job_description,
        ranking_criteria=payload.ranking_criteria
    )
    
    return result


class FilterRequest(BaseModel):
    resumes: List[Dict[str, str]] = Field(..., description="Resumes to filter")
    job_description: str = Field(..., description="Job description")
    filters: Dict[str, Any] = Field(
        ...,
        description="Filter criteria: min_ats_score, required_skills, industries, min_confidence"
    )
    max_workers: int = Field(4, ge=1, le=10)


@app.post("/filter-resumes")
def filter_resumes_endpoint(payload: FilterRequest) -> Dict[str, Any]:
    """
    Filter Resumes by Criteria
    
    Filter resumes based on:
    - Minimum ATS score threshold
    - Required skills (must have all)
    - Allowed industries
    - Minimum confidence threshold
    
    Returns passed and rejected resumes with reasons.
    """
    processor = get_batch_processor(max_workers=payload.max_workers)
    
    result = processor.filter_resumes(
        resumes=payload.resumes,
        job_description=payload.job_description,
        filters=payload.filters
    )
    
    return result


@app.get("/industries")
def list_industries() -> Dict[str, Any]:
    """
    List Available Industry Profiles
    
    Returns all available industry profiles with their characteristics.
    """
    industries_info = []
    
    for key, profile in INDUSTRY_PROFILES.items():
        industries_info.append({
            "key": key,
            "name": profile.name,
            "key_skills_count": len(profile.key_skills),
            "top_skills": profile.key_skills[:5],
            "min_experience_years": profile.min_experience_years,
            "scoring_weights": profile.scoring_weights
        })
    
    return {
        "total_industries": len(INDUSTRY_PROFILES),
        "industries": industries_info
    }


@app.get("/stats")
def service_stats() -> Dict[str, Any]:
    """
    Service Statistics and Capabilities
    
    Returns information about the service's ML models and capabilities.
    """
    return {
        "service_version": "2.0-enhanced",
        "capabilities": [
            "Enhanced ML scoring with confidence metrics",
            "Industry-specific scoring (8+ industries)",
            "Batch processing with parallelization",
            "Resume comparison and ranking",
            "Smart candidate filtering",
            "Transformer-based semantic analysis",
            "Advanced NER and entity extraction"
        ],
        "industries_supported": len(INDUSTRY_PROFILES),
        "skills_loaded": len(get_skill_dictionary()),
        "models": {
            "embeddings": "sentence-transformers",
            "nlp": "spaCy",
            "scoring": "enhanced_ml_v2",
            "confidence": "multi-factor_analysis"
        },
        "endpoints": {
            "legacy": ["/analyze", "/rank", "/word-quality", "/analyze-ml"],
            "enhanced_v2": [
                "/analyze-enhanced",
                "/industry-score",
                "/compare-resumes",
                "/find-best-candidate",
                "/batch-process",
                "/smart-rank",
                "/filter-resumes"
            ]
        }
    }
