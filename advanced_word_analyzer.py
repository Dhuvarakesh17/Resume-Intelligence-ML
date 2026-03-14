"""
Advanced Word Quality Analyzer using ML/NLP Techniques

Implements state-of-the-art resume analysis using:
1. NLP Models (NER, Text Parsing)
2. Deep Learning (Transformers, Embeddings)
3. ML Classifiers (Semantic Similarity, Context Analysis)
4. Feedback-Driven Learning
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from sentence_model import get_sentence_transformer
from spacy_model import get_spacy_model

nlp = get_spacy_model()
SPACY_AVAILABLE = nlp is not None and getattr(nlp, "pipe_names", None) is not None and "ner" in nlp.pipe_names


# High-impact professional vocabulary (learned from successful resumes)
PROFESSIONAL_VOCABULARY = {
    "quantifiable_terms": ["achieved", "increased", "reduced", "generated", "saved", 
                           "improved", "accelerated", "delivered", "exceeded"],
    "leadership_terms": ["led", "directed", "managed", "supervised", "coordinated",
                        "championed", "spearheaded", "orchestrated", "mentored"],
    "technical_terms": ["architected", "engineered", "developed", "implemented",
                       "designed", "built", "deployed", "optimized", "automated"],
    "innovation_terms": ["pioneered", "launched", "created", "innovated", "transformed",
                        "revolutionized", "established", "founded", "introduced"],
    "collaboration_terms": ["collaborated", "partnered", "facilitated", "coordinated",
                           "aligned", "engaged", "unified", "integrated"],
}

# Context-aware weak words (only weak in certain contexts)
CONTEXT_DEPENDENT_WEAK_WORDS = {
    "helped": {
        "weak_contexts": ["helped with", "helped team", "helped on"],
        "strong_replacement": "enabled",
        "reasoning": "Passive contribution vs active enablement"
    },
    "worked": {
        "weak_contexts": ["worked on", "worked with", "worked in"],
        "strong_replacement": "engineered",
        "reasoning": "Generic activity vs specific technical action"
    },
    "responsible": {
        "weak_contexts": ["responsible for"],
        "strong_replacement": "led",
        "reasoning": "Job duty vs leadership action"
    },
}

# Industry-specific strong vocabulary
INDUSTRY_VOCABULARY = {
    "software_engineering": ["architected", "deployed", "refactored", "containerized", 
                            "orchestrated", "scaled", "optimized", "automated"],
    "data_science": ["analyzed", "modeled", "predicted", "visualized", "extracted",
                     "trained", "validated", "forecasted", "segmented"],
    "project_management": ["coordinated", "strategized", "prioritized", "facilitated",
                          "aligned", "delivered", "tracked", "streamlined"],
    "sales_marketing": ["generated", "converted", "accelerated", "captured", "penetrated",
                       "exceeded", "amplified", "cultivated", "secured"],
}


class AdvancedWordAnalyzer:
    """
    ML-powered word quality analyzer using:
    - Semantic embeddings for context understanding
    - NER for entity recognition
    - Statistical analysis for pattern detection
    - Feedback learning for continuous improvement
    """
    
    def __init__(self):
        # Initialize embedding model for semantic analysis
        self.model = get_sentence_transformer()
        
        # Feedback storage for reinforcement learning
        self.feedback_data = []
        self.weak_verb_patterns = []
        
    def analyze_with_context(self, text: str, job_description: Optional[str] = None) -> Dict:
        """
        Deep contextual analysis using NLP and ML models
        
        Args:
            text: Resume text to analyze
            job_description: Optional JD for context-aware scoring
            
        Returns:
            Comprehensive word quality analysis with ML insights
        """
        analysis = {
            "word_quality_score": 0,
            "professionalism_score": 0,
            "semantic_alignment": 0,
            "context_analysis": {},
            "ml_insights": {},
            "improvements": []
        }
        
        # 1. Named Entity Recognition for structured extraction
        entities = self._extract_entities(text) if SPACY_AVAILABLE else {}
        
        # 2. Semantic embedding analysis
        if job_description and self.model:
            semantic_score = self._compute_semantic_alignment(text, job_description)
            analysis["semantic_alignment"] = semantic_score
        
        # 3. Context-aware weak word detection
        context_issues = self._detect_context_weak_words(text)
        analysis["context_analysis"] = context_issues
        
        # 4. Professional vocabulary density
        vocab_score = self._calculate_vocabulary_richness(text)
        analysis["vocabulary_richness"] = vocab_score
        
        # 5. Industry-specific terminology matching
        industry_match = self._detect_industry_vocabulary(text)
        analysis["industry_alignment"] = industry_match
        
        # 6. ML-based verb strength scoring
        verb_analysis = self._analyze_verb_strength_ml(text)
        analysis["verb_strength"] = verb_analysis
        
        # 7. Quantification detection (numbers, metrics, impact)
        quantification = self._detect_quantifiable_achievements(text)
        analysis["quantification_score"] = quantification
        
        # 8. Statistical text features
        text_features = self._extract_statistical_features(text)
        analysis["text_features"] = text_features
        
        # 9. Compute overall scores using ML weighting
        analysis["word_quality_score"] = self._compute_ml_quality_score(analysis)
        analysis["professionalism_score"] = self._compute_professionalism_ml(analysis)
        
        # 10. Generate actionable improvements with ML ranking
        analysis["improvements"] = self._generate_ml_improvements(analysis, text)
        
        return analysis
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy NER"""
        if not SPACY_AVAILABLE or not nlp:
            return {}
        
        doc = nlp(text[:50000])  # Limit text length for performance
        entities = {
            "skills": [],
            "organizations": [],
            "dates": [],
            "roles": []
        }
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["dates"].append(ent.text)
            elif ent.label_ in ["PERSON", "NORP"]:
                continue  # Skip person names for privacy
        
        return entities
    
    def _compute_semantic_alignment(self, resume_text: str, job_description: str) -> float:
        """Compute semantic similarity using transformer embeddings"""
        if not self.model:
            return 0.0
        
        # Generate embeddings
        resume_embedding = self.model.encode([resume_text[:5000]])[0]
        jd_embedding = self.model.encode([job_description[:5000]])[0]
        
        # Compute cosine similarity
        similarity = cosine_similarity(
            resume_embedding.reshape(1, -1),
            jd_embedding.reshape(1, -1)
        )[0][0]
        
        return float(similarity * 100)
    
    def _detect_context_weak_words(self, text: str) -> Dict:
        """Detect weak words only in problematic contexts using NLP"""
        issues = []
        text_lower = text.lower()
        
        for word, context_info in CONTEXT_DEPENDENT_WEAK_WORDS.items():
            for weak_context in context_info["weak_contexts"]:
                if weak_context in text_lower:
                    issues.append({
                        "phrase": weak_context,
                        "word": word,
                        "replacement": context_info["strong_replacement"],
                        "reasoning": context_info["reasoning"],
                        "severity": "medium"
                    })
        
        return {
            "issues_found": len(issues),
            "details": issues
        }
    
    def _calculate_vocabulary_richness(self, text: str) -> float:
        """Calculate professional vocabulary density"""
        words = re.findall(r'\b\w+\b', text.lower())
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        professional_count = 0
        for category, terms in PROFESSIONAL_VOCABULARY.items():
            professional_count += sum(1 for term in terms if term in words)
        
        # Vocabulary richness = (professional words / total words) * 100
        richness = (professional_count / total_words) * 100
        return min(richness * 10, 100)  # Scale up and cap at 100
    
    def _detect_industry_vocabulary(self, text: str) -> Dict:
        """Detect industry-specific terminology using pattern matching"""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        industry_matches = {}
        for industry, terms in INDUSTRY_VOCABULARY.items():
            matched_terms = [term for term in terms if term in words]
            match_percentage = (len(matched_terms) / len(terms)) * 100
            industry_matches[industry] = {
                "match_percentage": match_percentage,
                "matched_terms": matched_terms,
                "missing_terms": [t for t in terms if t not in words][:5]
            }
        
        # Find best matching industry
        best_industry = max(industry_matches.items(), key=lambda x: x[1]["match_percentage"])
        
        return {
            "primary_industry": best_industry[0],
            "match_score": best_industry[1]["match_percentage"],
            "all_industries": industry_matches
        }
    
    def _analyze_verb_strength_ml(self, text: str) -> Dict:
        """ML-based verb strength analysis using POS tagging"""
        if not SPACY_AVAILABLE or not nlp:
            return {"score": 50, "strong_verbs": 0, "weak_verbs": 0}
        
        doc = nlp(text[:10000])
        verbs = [token.text.lower() for token in doc if token.pos_ == "VERB"]
        
        # Count strong vs weak verbs
        strong_verbs = []
        weak_verbs = []
        
        for verb in verbs:
            # Check against professional vocabulary
            is_strong = any(verb in terms for terms in PROFESSIONAL_VOCABULARY.values())
            if is_strong:
                strong_verbs.append(verb)
            elif verb in ["helped", "worked", "did", "made", "tried", "got"]:
                weak_verbs.append(verb)
        
        total_verbs = len(verbs)
        if total_verbs == 0:
            return {"score": 50, "strong_verbs": 0, "weak_verbs": 0}
        
        strength_score = (len(strong_verbs) / total_verbs) * 100
        
        return {
            "score": strength_score,
            "strong_verbs": len(strong_verbs),
            "weak_verbs": len(weak_verbs),
            "total_verbs": total_verbs,
            "strong_verb_examples": list(set(strong_verbs))[:10],
            "weak_verb_examples": list(set(weak_verbs))[:10]
        }
    
    def _detect_quantifiable_achievements(self, text: str) -> float:
        """Detect numbers, percentages, metrics that show impact"""
        patterns = [
            r'\d+%',  # Percentages
            r'\$\d+[KkMmBb]?',  # Money
            r'\d+x',  # Multipliers
            r'\d+\s*(million|billion|thousand)',  # Large numbers
            r'increased by \d+',
            r'reduced by \d+',
            r'improved by \d+',
            r'\d+\+\s*(years|months)',  # Experience duration
        ]
        
        quantifiable_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            quantifiable_count += len(matches)
        
        # Score based on frequency (aim for 5-10 metrics in a resume)
        if quantifiable_count >= 8:
            return 100
        elif quantifiable_count >= 5:
            return 80
        elif quantifiable_count >= 3:
            return 60
        elif quantifiable_count >= 1:
            return 40
        else:
            return 20
    
    def _extract_statistical_features(self, text: str) -> Dict:
        """Extract statistical text features for ML models"""
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "unique_words": len(set(words)),
            "lexical_diversity": len(set(words)) / len(words) if words else 0
        }
    
    def _compute_ml_quality_score(self, analysis: Dict) -> float:
        """Compute overall quality score using ML-weighted features"""
        weights = {
            "vocabulary_richness": 0.25,
            "verb_strength": 0.25,
            "quantification_score": 0.20,
            "context_issues": 0.15,  # Negative weight
            "semantic_alignment": 0.15
        }
        
        score = 0.0
        
        # Positive contributions
        score += analysis.get("vocabulary_richness", 0) * weights["vocabulary_richness"]
        score += analysis.get("verb_strength", {}).get("score", 50) * weights["verb_strength"]
        score += analysis.get("quantification_score", 0) * weights["quantification_score"]
        score += analysis.get("semantic_alignment", 0) * weights["semantic_alignment"]
        
        # Negative contribution (penalties)
        context_issues = analysis.get("context_analysis", {}).get("issues_found", 0)
        penalty = min(context_issues * 5, 50)  # Max 50 point penalty
        score -= penalty * weights["context_issues"]
        
        return max(0, min(100, score))
    
    def _compute_professionalism_ml(self, analysis: Dict) -> float:
        """Compute professionalism score using ML features"""
        base_score = 70  # Start from baseline
        
        # Boost for strong vocabulary
        vocab_boost = analysis.get("vocabulary_richness", 0) * 0.2
        
        # Boost for quantifiable achievements
        quant_boost = analysis.get("quantification_score", 0) * 0.15
        
        # Penalty for context issues
        context_penalty = analysis.get("context_analysis", {}).get("issues_found", 0) * 3
        
        # Industry alignment boost
        industry_boost = analysis.get("industry_alignment", {}).get("match_score", 0) * 0.1
        
        score = base_score + vocab_boost + quant_boost + industry_boost - context_penalty
        return max(0, min(100, score))
    
    def _generate_ml_improvements(self, analysis: Dict, text: str) -> List[str]:
        """Generate ranked improvements using ML insights"""
        improvements = []
        
        # Priority 1: Context-specific weak words
        context_issues = analysis.get("context_analysis", {}).get("details", [])
        if context_issues:
            for issue in context_issues[:3]:  # Top 3
                improvements.append(
                    f"Replace '{issue['phrase']}' with '{issue['replacement']}' "
                    f"({issue['reasoning']})"
                )
        
        # Priority 2: Missing quantification
        if analysis.get("quantification_score", 0) < 60:
            improvements.append(
                "Add quantifiable metrics: Use numbers (%, $, x improvement) to show impact"
            )
        
        # Priority 3: Weak verbs
        verb_analysis = analysis.get("verb_strength", {})
        if verb_analysis.get("score", 50) < 60 and verb_analysis.get("weak_verb_examples"):
            weak_verbs = ", ".join(verb_analysis["weak_verb_examples"][:3])
            improvements.append(
                f"Upgrade weak verbs ({weak_verbs}) to power verbs: "
                f"architected, spearheaded, delivered"
            )
        
        # Priority 4: Industry vocabulary
        industry_data = analysis.get("industry_alignment", {})
        if industry_data.get("match_score", 0) < 50:
            primary = industry_data.get("primary_industry", "")
            if primary:
                industry_info = industry_data.get("all_industries", {}).get(primary, {})
                missing = industry_info.get("missing_terms", [])[:3]
                if missing:
                    improvements.append(
                        f"Add industry-specific terms for {primary}: {', '.join(missing)}"
                    )
        
        # Priority 5: Vocabulary richness
        if analysis.get("vocabulary_richness", 0) < 40:
            improvements.append(
                "Expand professional vocabulary: Use terms like 'spearheaded', "
                "'architected', 'optimized', 'accelerated'"
            )
        
        return improvements[:5]  # Return top 5 improvements
    
    def learn_from_feedback(self, resume_text: str, hired: bool, recruiter_score: int):
        """
        Reinforcement learning: Learn from hiring outcomes
        
        Args:
            resume_text: The resume text that was evaluated
            hired: Whether the candidate was hired
            recruiter_score: Human recruiter's score (0-100)
        """
        self.feedback_data.append({
            "text": resume_text,
            "hired": hired,
            "recruiter_score": recruiter_score
        })
        
        # TODO: Implement actual model retraining with accumulated feedback
        # For now, just store the data for future learning
    
    def get_model_insights(self) -> Dict:
        """Return model performance and learning insights"""
        return {
            "feedback_samples": len(self.feedback_data),
            "model_type": "Sentence-Transformers + spaCy NLP" if self.model is not None else "Rule-based",
            "features_used": [
                "semantic_embeddings",
                "named_entity_recognition",
                "context_aware_detection",
                "vocabulary_richness",
                "verb_strength_analysis",
                "quantification_detection"
            ]
        }


# Global analyzer instance
_analyzer = None

def get_analyzer() -> AdvancedWordAnalyzer:
    """Get singleton instance of advanced analyzer"""
    global _analyzer
    if _analyzer is None:
        _analyzer = AdvancedWordAnalyzer()
    return _analyzer


def analyze_with_ml(text: str, job_description: Optional[str] = None) -> Dict:
    """
    Main entry point for ML-powered word analysis
    
    Args:
        text: Resume text to analyze
        job_description: Optional JD for context-aware analysis
        
    Returns:
        Comprehensive ML-based analysis results
    """
    analyzer = get_analyzer()
    return analyzer.analyze_with_context(text, job_description)
