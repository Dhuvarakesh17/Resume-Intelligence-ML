"""
Enhanced ML Scoring with Confidence Metrics
Provides advanced scoring with model confidence and explainability
"""
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class EnhancedMLScorer:
    """
    Advanced ML-based resume scoring with confidence metrics
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True
        )
    
    def calculate_semantic_similarity(
        self,
        resume_text: str,
        job_description: str
    ) -> Tuple[float, float]:
        """
        Calculate semantic similarity with confidence score
        
        Returns:
            (similarity_score, confidence)
        """
        if not job_description or len(job_description.strip()) < 50:
            return 0.0, 0.0
        
        try:
            vectors = self.vectorizer.fit_transform([resume_text, job_description])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            similarity_score = float(similarity * 100)
            
            # Calculate confidence based on text lengths and vocabulary overlap
            resume_words = set(resume_text.lower().split())
            job_words = set(job_description.lower().split())
            vocab_overlap = len(resume_words.intersection(job_words))
            
            # Confidence factors
            length_confidence = min(1.0, len(job_description) / 500) * 40  # Up to 40 points
            vocab_confidence = min(1.0, vocab_overlap / 50) * 40  # Up to 40 points
            similarity_confidence = similarity * 20  # Up to 20 points
            
            confidence = length_confidence + vocab_confidence + similarity_confidence
            confidence = min(100, max(0, confidence))
            
            return round(similarity_score, 2), round(confidence, 2)
        
        except Exception as e:
            print(f"Semantic similarity calculation error: {e}")
            return 0.0, 0.0
    
    def analyze_skill_matching(
        self,
        resume_skills: List[str],
        job_skills: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze skill matching with detailed breakdown and confidence
        """
        if not job_skills:
            return {
                "match_score": 0,
                "confidence": 0,
                "matching_skills": [],
                "missing_skills": [],
                "match_percentage": 0,
                "skill_categories": {}
            }
        
        resume_skills_lower = set(s.lower() for s in resume_skills)
        job_skills_lower = set(s.lower() for s in job_skills)
        
        # Direct matches
        direct_matches = resume_skills_lower.intersection(job_skills_lower)
        
        # Fuzzy matches (partial matching)
        fuzzy_matches = set()
        for job_skill in job_skills_lower:
            for resume_skill in resume_skills_lower:
                if job_skill in resume_skill or resume_skill in job_skill:
                    fuzzy_matches.add(job_skill)
        
        total_matches = direct_matches.union(fuzzy_matches)
        missing_skills = job_skills_lower - total_matches
        
        # Calculate match percentage
        match_percentage = (len(total_matches) / len(job_skills_lower)) * 100
        
        # Calculate match score (weighted towards must-have skills)
        match_score = min(100, match_percentage * 1.2)  # Boost good matches
        
        # Confidence based on number of skills and match quality
        skill_count_factor = min(1.0, len(resume_skills) / 10) * 50
        match_quality = (len(direct_matches) / len(total_matches) if total_matches else 0) * 50
        confidence = skill_count_factor + match_quality
        
        # Categorize skills
        skill_categories = self._categorize_skills(list(total_matches))
        
        return {
            "match_score": round(match_score, 2),
            "confidence": round(confidence, 2),
            "matching_skills": list(total_matches),
            "missing_skills": list(missing_skills),
            "match_percentage": round(match_percentage, 2),
            "direct_matches": len(direct_matches),
            "fuzzy_matches": len(fuzzy_matches),
            "skill_categories": skill_categories
        }
    
    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize skills into technical areas"""
        categories = {
            "languages": [],
            "frameworks": [],
            "databases": [],
            "cloud": [],
            "tools": [],
            "other": []
        }
        
        language_keywords = ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'ruby', 'php', 'typescript', 'kotlin', 'swift', 'r']
        framework_keywords = ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'node', 'fastapi', 'tensorflow', 'pytorch', 'keras']
        database_keywords = ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'dynamodb', 'cassandra', 'oracle', 'nosql']
        cloud_keywords = ['aws', 'azure', 'gcp', 'cloud', 'kubernetes', 'docker', 'serverless', 'lambda']
        tool_keywords = ['git', 'jenkins', 'jira', 'linux', 'ci/cd', 'terraform', 'ansible']
        
        for skill in skills:
            skill_lower = skill.lower()
            categorized = False
            
            if any(lang in skill_lower for lang in language_keywords):
                categories["languages"].append(skill)
                categorized = True
            if any(fw in skill_lower for fw in framework_keywords):
                categories["frameworks"].append(skill)
                categorized = True
            if any(db in skill_lower for db in database_keywords):
                categories["databases"].append(skill)
                categorized = True
            if any(cloud in skill_lower for cloud in cloud_keywords):
                categories["cloud"].append(skill)
                categorized = True
            if any(tool in skill_lower for tool in tool_keywords):
                categories["tools"].append(skill)
                categorized = True
            
            if not categorized:
                categories["other"].append(skill)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def analyze_resume_quality(self, resume_text: str) -> Dict[str, Any]:
        """
        Comprehensive resume quality analysis with confidence scoring
        """
        resume_lower = resume_text.lower()
        word_count = len(resume_text.split())
        
        # Check for key sections
        sections = {
            "summary": any(kw in resume_lower for kw in ['summary', 'objective', 'profile']),
            "experience": any(kw in resume_lower for kw in ['experience', 'work history', 'employment']),
            "education": any(kw in resume_lower for kw in ['education', 'degree', 'university']),
            "skills": any(kw in resume_lower for kw in ['skills', 'technical skills', 'competencies']),
            "projects": any(kw in resume_lower for kw in ['project', 'portfolio']),
            "certifications": any(kw in resume_lower for kw in ['certification', 'certified', 'license'])
        }
        
        section_score = (sum(sections.values()) / len(sections)) * 100
        
        # Check for action verbs
        action_verbs = [
            'developed', 'designed', 'implemented', 'built', 'created', 'managed',
            'led', 'coordinated', 'improved', 'optimized', 'increased', 'reduced',
            'achieved', 'delivered', 'launched', 'architected', 'engineered'
        ]
        action_verb_count = sum(1 for verb in action_verbs if verb in resume_lower)
        action_verb_score = min(100, (action_verb_count / 5) * 100)
        
        # Check for quantifiable metrics
        has_percentages = bool(re.search(r'\d+%', resume_text))
        has_numbers = bool(re.search(r'\d{1,3}[,\d]*', resume_text))
        has_currency = bool(re.search(r'\$\d', resume_text))
        has_time_metrics = any(word in resume_lower for word in ['week', 'month', 'year', 'day', 'hours'])
        
        metrics_count = sum([has_percentages, has_numbers, has_currency, has_time_metrics])
        metrics_score = (metrics_count / 4) * 100
        
        # Check for buzzwords vs substance
        buzzwords = ['synergy', 'rockstar', 'ninja', 'guru', 'dynamic', 'passionate', 'team player', 'hardworking']
        buzzword_count = sum(1 for word in buzzwords if word in resume_lower)
        buzzword_penalty = min(20, buzzword_count * 5)
        
        # Word count assessment
        if 300 <= word_count <= 800:
            length_score = 100
        elif 200 <= word_count < 300 or 800 < word_count <= 1000:
            length_score = 80
        elif word_count < 200:
            length_score = 50
        else:
            length_score = 60
        
        # Calculate overall quality score
        quality_score = (
            section_score * 0.30 +
            action_verb_score * 0.25 +
            metrics_score * 0.25 +
            length_score * 0.20
        ) - buzzword_penalty
        
        quality_score = max(0, min(100, quality_score))
        
        # Calculate confidence
        # Higher confidence if resume is complete and has good signals
        confidence_factors = [
            sum(sections.values()) >= 4,  # Has most sections
            action_verb_count >= 5,  # Good action verbs
            metrics_count >= 2,  # Has metrics
            300 <= word_count <= 1000,  # Good length
            buzzword_count <= 2  # Limited buzzwords
        ]
        
        confidence = (sum(confidence_factors) / len(confidence_factors)) * 100
        
        return {
            "quality_score": round(quality_score, 2),
            "confidence": round(confidence, 2),
            "breakdown": {
                "section_completeness": round(section_score, 2),
                "action_verb_usage": round(action_verb_score, 2),
                "quantifiable_metrics": round(metrics_score, 2),
                "length_appropriateness": round(length_score, 2),
                "buzzword_penalty": round(buzzword_penalty, 2)
            },
            "sections_present": sections,
            "word_count": word_count,
            "action_verb_count": action_verb_count,
            "has_metrics": metrics_count > 0,
            "buzzword_count": buzzword_count,
            "suggestions": self._generate_quality_suggestions(sections, action_verb_count, metrics_count, buzzword_count, word_count)
        }
    
    def _generate_quality_suggestions(
        self,
        sections: Dict[str, bool],
        action_verb_count: int,
        metrics_count: int,
        buzzword_count: int,
        word_count: int
    ) -> List[str]:
        """Generate actionable suggestions for improving resume quality"""
        suggestions = []
        
        # Missing sections
        missing = [k for k, v in sections.items() if not v]
        if missing:
            suggestions.append(f"Add {', '.join(missing[:2])} section(s)")
        
        # Action verbs
        if action_verb_count < 5:
            suggestions.append("Use more strong action verbs (developed, implemented, led, etc.)")
        
        # Metrics
        if metrics_count < 2:
            suggestions.append("Add quantifiable metrics (percentages, numbers, dollar amounts)")
        
        # Buzzwords
        if buzzword_count > 2:
            suggestions.append("Replace buzzwords with specific achievements and skills")
        
        # Length
        if word_count < 300:
            suggestions.append("Expand resume with more details about experience and achievements")
        elif word_count > 1000:
            suggestions.append("Consider condensing to focus on most relevant experience")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def compute_comprehensive_score(
        self,
        resume_text: str,
        job_description: str,
        resume_skills: List[str],
        job_skills: List[str]
    ) -> Dict[str, Any]:
        """
        Compute comprehensive ML score with confidence and explainability
        """
        # Semantic analysis
        semantic_score, semantic_confidence = self.calculate_semantic_similarity(
            resume_text, job_description
        )
        
        # Skill matching
        skill_analysis = self.analyze_skill_matching(resume_skills, job_skills)
        
        # Resume quality
        quality_analysis = self.analyze_resume_quality(resume_text)
        
        # Calculate weighted final score
        weights = {
            "semantic": 0.35,
            "skills": 0.40,
            "quality": 0.25
        }
        
        final_score = (
            semantic_score * weights["semantic"] +
            skill_analysis["match_score"] * weights["skills"] +
            quality_analysis["quality_score"] * weights["quality"]
        )
        
        # Overall confidence (weighted average of component confidences)
        overall_confidence = (
            semantic_confidence * weights["semantic"] +
            skill_analysis["confidence"] * weights["skills"] +
            quality_analysis["confidence"] * weights["quality"]
        )
        
        return {
            "final_score": round(final_score, 2),
            "confidence": round(overall_confidence, 2),
            "components": {
                "semantic": {
                    "score": semantic_score,
                    "confidence": semantic_confidence,
                    "weight": weights["semantic"]
                },
                "skills": {
                    "score": skill_analysis["match_score"],
                    "confidence": skill_analysis["confidence"],
                    "weight": weights["skills"],
                    "details": skill_analysis
                },
                "quality": {
                    "score": quality_analysis["quality_score"],
                    "confidence": quality_analysis["confidence"],
                    "weight": weights["quality"],
                    "details": quality_analysis
                }
            },
            "interpretation": self._interpret_score(final_score, overall_confidence),
            "top_recommendations": self._get_top_recommendations(
                skill_analysis, quality_analysis
            )
        }
    
    def _interpret_score(self, score: float, confidence: float) -> str:
        """Provide human-readable interpretation of the score"""
        if confidence < 40:
            reliability = "Low confidence - "
        elif confidence < 70:
            reliability = "Moderate confidence - "
        else:
            reliability = "High confidence - "
        
        if score >= 80:
            interpretation = "Excellent match. Resume aligns very well with requirements."
        elif score >= 65:
            interpretation = "Good match. Resume meets most requirements with minor gaps."
        elif score >= 50:
            interpretation = "Fair match. Resume shows potential but has notable gaps."
        else:
            interpretation = "Needs improvement. Resume has significant gaps."
        
        return reliability + interpretation
    
    def _get_top_recommendations(
        self,
        skill_analysis: Dict,
        quality_analysis: Dict
    ) -> List[str]:
        """Get prioritized recommendations for improvement"""
        recommendations = []
        
        # Skill-based recommendations
        if skill_analysis["match_percentage"] < 60:
            missing = skill_analysis["missing_skills"][:3]
            if missing:
                recommendations.append(f"Add missing skills: {', '.join(missing)}")
        
        # Quality-based recommendations
        recommendations.extend(quality_analysis["suggestions"][:2])
        
        return recommendations[:5]


# Singleton instance
_scorer = None

def get_enhanced_scorer() -> EnhancedMLScorer:
    """Get or create singleton scorer instance"""
    global _scorer
    if _scorer is None:
        _scorer = EnhancedMLScorer()
    return _scorer
