"""
Batch Processing Module
High-performance batch processing for multiple resumes
"""
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from skill_extractor import extract_skills
from ml_scoring import compute_ml_ats_score
from enhanced_ml_scoring import get_enhanced_scorer
from industry_scoring import detect_industry, calculate_industry_score
from job_recommender import recommend_roles


class BatchProcessor:
    """
    Efficient batch processing for multiple resumes
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize batch processor
        
        Args:
            max_workers: Maximum parallel workers for processing
        """
        self.max_workers = max_workers
        self.scorer = get_enhanced_scorer()
    
    def process_resume(
        self,
        resume_text: str,
        job_description: str = "",
        include_detailed_analysis: bool = False,
        resume_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single resume with comprehensive analysis
        """
        start_time = time.time()
        
        # Extract skills
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description) if job_description else []
        
        # Basic scoring
        ats_score = compute_ml_ats_score(
            resume_text=resume_text,
            job_description=job_description,
            resume_skills=resume_skills,
            job_skills=job_skills
        )
        
        # Role recommendations
        recommended_roles = recommend_roles(resume_text, resume_skills)
        
        # Industry detection
        industry_key = detect_industry(resume_text, resume_skills)
        
        result = {
            "resume_id": resume_id,
            "ats_score": ats_score,
            "skills_found": resume_skills,
            "skill_count": len(resume_skills),
            "recommended_roles": recommended_roles,
            "primary_role": recommended_roles[0] if recommended_roles else "General",
            "detected_industry": industry_key,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        # Add detailed analysis if requested
        if include_detailed_analysis:
            # Enhanced ML scoring with confidence
            enhanced_analysis = self.scorer.compute_comprehensive_score(
                resume_text=resume_text,
                job_description=job_description,
                resume_skills=resume_skills,
                job_skills=job_skills
            )
            
            # Industry-specific scoring
            industry_analysis = calculate_industry_score(
                resume_text=resume_text,
                resume_skills=resume_skills,
                industry_key=industry_key
            )
            
            result["enhanced_score"] = enhanced_analysis["final_score"]
            result["confidence"] = enhanced_analysis["confidence"]
            result["interpretation"] = enhanced_analysis["interpretation"]
            result["recommendations"] = enhanced_analysis["top_recommendations"]
            result["industry_score"] = industry_analysis["industry_score"]
            result["industry_confidence"] = industry_analysis["confidence"]
        
        return result
    
    def process_batch(
        self,
        resumes: List[Dict[str, str]],
        job_description: str = "",
        include_detailed_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Process multiple resumes in parallel
        
        Args:
            resumes: List of dicts with 'text' and optional 'id' keys
            job_description: Target job description
            include_detailed_analysis: Whether to include detailed ML analysis
        
        Returns:
            Batch processing results with statistics
        """
        start_time = time.time()
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_resume = {}
            for i, resume_data in enumerate(resumes):
                resume_text = resume_data.get('text', '')
                resume_id = resume_data.get('id', f'resume_{i}')
                
                future = executor.submit(
                    self.process_resume,
                    resume_text,
                    job_description,
                    include_detailed_analysis,
                    resume_id
                )
                future_to_resume[future] = resume_id
            
            # Collect results as they complete
            for future in as_completed(future_to_resume):
                resume_id = future_to_resume[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append({
                        "resume_id": resume_id,
                        "error": str(e)
                    })
        
        # Sort results by resume_id
        results.sort(key=lambda x: x.get('resume_id', ''))
        
        # Calculate batch statistics
        total_time = time.time() - start_time
        successful_count = len(results)
        failed_count = len(errors)
        
        # Calculate aggregate metrics
        if results:
            ats_scores = [r["ats_score"] for r in results]
            avg_ats = sum(ats_scores) / len(ats_scores)
            max_ats = max(ats_scores)
            min_ats = min(ats_scores)
            
            # Top performer
            top_resume = max(results, key=lambda x: x["ats_score"])
            
            statistics = {
                "total_processed": successful_count,
                "failed": failed_count,
                "total_time_seconds": round(total_time, 2),
                "avg_processing_time_ms": round(sum(r["processing_time_ms"] for r in results) / len(results), 2),
                "throughput_per_second": round(successful_count / total_time, 2) if total_time > 0 else 0,
                "ats_statistics": {
                    "average": round(avg_ats, 2),
                    "max": max_ats,
                    "min": min_ats,
                    "range": max_ats - min_ats
                },
                "top_performer": {
                    "resume_id": top_resume["resume_id"],
                    "ats_score": top_resume["ats_score"],
                    "primary_role": top_resume["primary_role"]
                }
            }
        else:
            statistics = {
                "total_processed": 0,
                "failed": failed_count,
                "total_time_seconds": round(total_time, 2)
            }
        
        return {
            "status": "completed",
            "results": results,
            "errors": errors,
            "statistics": statistics,
            "batch_metadata": {
                "total_resumes": len(resumes),
                "job_description_provided": bool(job_description),
                "detailed_analysis_enabled": include_detailed_analysis,
                "parallelization_factor": self.max_workers
            }
        }
    
    def smart_rank_resumes(
        self,
        resumes: List[Dict[str, str]],
        job_description: str,
        ranking_criteria: str = "ats_score"
    ) -> Dict[str, Any]:
        """
        Process and rank resumes by specified criteria
        
        Args:
            resumes: List of resume dicts
            job_description: Target job description
            ranking_criteria: 'ats_score', 'enhanced_score', or 'industry_score'
        
        Returns:
            Ranked results with detailed comparison
        """
        # Process all resumes with detailed analysis
        batch_result = self.process_batch(
            resumes=resumes,
            job_description=job_description,
            include_detailed_analysis=True
        )
        
        if not batch_result["results"]:
            return batch_result
        
        results = batch_result["results"]
        
        # Rank by specified criteria
        if ranking_criteria == "enhanced_score":
            results.sort(key=lambda x: x.get("enhanced_score", 0), reverse=True)
        elif ranking_criteria == "industry_score":
            results.sort(key=lambda x: x.get("industry_score", 0), reverse=True)
        else:  # default to ats_score
            results.sort(key=lambda x: x["ats_score"], reverse=True)
        
        # Add rankings
        for i, result in enumerate(results):
            result["rank"] = i + 1
        
        # Generate insights
        top_3 = results[:3]
        insights = {
            "winner": {
                "resume_id": results[0]["resume_id"],
                "score": results[0].get(ranking_criteria, results[0]["ats_score"]),
                "confidence": results[0].get("confidence", "N/A"),
                "strengths": results[0].get("recommended_roles", [])[:2]
            },
            "top_3_resumes": [
                {
                    "rank": r["rank"],
                    "resume_id": r["resume_id"],
                    "score": r.get(ranking_criteria, r["ats_score"]),
                    "role": r["primary_role"]
                }
                for r in top_3
            ],
            "score_distribution": {
                "highest": results[0].get(ranking_criteria, results[0]["ats_score"]),
                "lowest": results[-1].get(ranking_criteria, results[-1]["ats_score"]),
                "median": results[len(results)//2].get(ranking_criteria, results[len(results)//2]["ats_score"])
            }
        }
        
        return {
            "status": "ranked",
            "ranking_criteria": ranking_criteria,
            "ranked_results": results,
            "insights": insights,
            "statistics": batch_result["statistics"],
            "errors": batch_result["errors"]
        }
    
    def filter_resumes(
        self,
        resumes: List[Dict[str, str]],
        job_description: str,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter resumes based on criteria
        
        Args:
            resumes: List of resume dicts
            job_description: Target job description
            filters: Filter criteria dict, e.g.:
                {
                    "min_ats_score": 60,
                    "required_skills": ["python", "aws"],
                    "industries": ["software_engineering", "data_science"],
                    "min_confidence": 50
                }
        
        Returns:
            Filtered results
        """
        # Process all resumes
        batch_result = self.process_batch(
            resumes=resumes,
            job_description=job_description,
            include_detailed_analysis=True
        )
        
        if not batch_result["results"]:
            return batch_result
        
        results = batch_result["results"]
        filtered_results = []
        rejected_results = []
        
        for result in results:
            passed = True
            rejection_reasons = []
            
            # Check ATS score threshold
            min_ats = filters.get("min_ats_score", 0)
            if result["ats_score"] < min_ats:
                passed = False
                rejection_reasons.append(f"ATS score {result['ats_score']} below threshold {min_ats}")
            
            # Check required skills
            required_skills = filters.get("required_skills", [])
            if required_skills:
                resume_skills_lower = set(s.lower() for s in result["skills_found"])
                missing_required = [s for s in required_skills if s.lower() not in resume_skills_lower]
                if missing_required:
                    passed = False
                    rejection_reasons.append(f"Missing required skills: {', '.join(missing_required)}")
            
            # Check industry
            allowed_industries = filters.get("industries", [])
            if allowed_industries:
                if result["detected_industry"] not in allowed_industries:
                    passed = False
                    rejection_reasons.append(f"Industry {result['detected_industry']} not in allowed list")
            
            # Check confidence threshold
            min_confidence = filters.get("min_confidence", 0)
            if min_confidence and result.get("confidence", 0) < min_confidence:
                passed = False
                rejection_reasons.append(f"Confidence {result.get('confidence', 0)} below threshold {min_confidence}")
            
            if passed:
                filtered_results.append(result)
            else:
                result["rejection_reasons"] = rejection_reasons
                rejected_results.append(result)
        
        return {
            "status": "filtered",
            "filters_applied": filters,
            "passed_count": len(filtered_results),
            "rejected_count": len(rejected_results),
            "passed_resumes": filtered_results,
            "rejected_resumes": rejected_results,
            "statistics": batch_result["statistics"]
        }


# Singleton instance
_batch_processor = None

def get_batch_processor(max_workers: int = 4) -> BatchProcessor:
    """Get or create singleton batch processor"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor(max_workers=max_workers)
    return _batch_processor
