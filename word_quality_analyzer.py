import re
from typing import Dict, List, Tuple

# Low-quality/generic words that weaken resumes
WEAK_WORDS = {
    "cheap": ("cost-effective", "budget-friendly", "optimized"),
    "simple": ("streamlined", "elegant", "efficient"),
    "good": ("excellent", "exceptional", "proven"),
    "bad": ("suboptimal", "ineffective", "unsustainable"),
    "help": ("enabled", "empowered", "facilitated"),
    "work": ("developed", "engineered", "architected"),
    "team": ("collaborated with", "partnered with", "coordinated"),
    "thing": ("solution", "initiative", "implementation"),
    "made": ("architected", "engineered", "developed"),
    "nice": ("exceptional", "outstanding", "compelling"),
    "try": ("established", "implemented", "accomplished"),
    "do": ("execute", "deliver", "drive"),
    "get": ("achieve", "attain", "obtain"),
    "give": ("provide", "deliver", "contribute"),
    "important": ("critical", "pivotal", "essential"),
    "interesting": ("compelling", "valuable", "innovative"),
    "involved": ("spearheaded", "orchestrated", "championed"),
    "job": ("role", "position", "engagement"),
    "just": ("omit", "remove"),
    "large": ("extensive", "comprehensive", "substantial"),
    "lot": ("significant", "substantial", "considerable"),
    "many": ("multiple", "numerous", "diverse"),
    "more": ("enhanced", "amplified", "accelerated"),
    "most": ("predominantly", "primarily", "primarily"),
    "never": ("avoid", "redesign context"),
    "new": ("innovative", "cutting-edge", "advanced"),
    "old": ("established", "proven", "legacy"),
    "only": ("solely", "exclusively", "uniquely"),
    "or": ("and/or", "or specifically"),
    "other": ("specific category"),
    "pretty": ("remarkably", "notably", "significantly"),
    "quick": ("rapid", "swift", "agile"),
    "really": ("genuinely", "actually", "quantify impact"),
    "saw": ("identified", "discovered", "recognized"),
    "should": ("must", "will", "can"),
    "small": ("focused", "targeted", "specialized"),
    "something": ("solution", "capability", "feature"),
    "sometimes": ("regularly", "consistently", "frequently"),
    "sort": ("category", "type", "classification"),
    "still": ("remains", "continues", "maintains"),
    "such": ("examples", "like", "including"),
    "take": ("require", "demand", "necessitate"),
    "than": ("compared to", "versus"),
    "thanks": ("due to", "leveraging", "utilizing"),
    "think": ("determined", "concluded", "assessed"),
    "this": ("specific noun", "concrete reference"),
    "though": ("however", "conversely", "alternatively"),
    "thought": ("concluded", "determined", "recognized"),
    "tried": ("implemented", "executed", "delivered"),
    "trying": ("addressing", "tackling", "resolving"),
    "use": ("leveraged", "utilized", "employed"),
    "used": ("leveraged", "implemented", "deployed"),
    "usually": ("typically", "generally", "predominantly"),
    "very": ("exceptionally", "remarkably", "significantly"),
    "want": ("designed", "engineered", "developed"),
    "way": ("approach", "methodology", "strategy"),
    "weird": ("unconventional", "novel", "distinctive"),
    "well": ("effectively", "successfully", "efficiently"),
    "went": ("progressed", "advanced", "evolved"),
    "were": ("active voice", "employ specific action"),
    "what": ("specific entity"),
    "when": ("date", "quarter", "timeframe"),
    "where": ("location", "context", "department"),
    "which": ("specify entity"),
    "while": ("while", "meanwhile", "during"),
    "who": ("team", "department", "stakeholder"),
    "why": ("impact", "value", "outcome"),
    "will": ("can", "will deliver", "will enable"),
    "with": ("and", "partnering"),
    "worse": ("declining", "deteriorating", "degrading"),
    "worst": ("most critical", "most urgent", "most challenging"),
}

STRONG_ACTION_VERBS = {
    "accelerated", "achieved", "adapted", "advanced", "amplified", "analyzed",
    "architected", "automated", "boosted", "built", "championed", "clarified",
    "collaborated", "consolidated", "coordinated", "created", "delivered",
    "designed", "diagnosed", "directed", "discovered", "driven", "earned",
    "engineered", "enhanced", "expanded", "expedited", "facilitated", "founded",
    "generated", "guided", "implemented", "improved", "innovated", "integrated",
    "led", "leveraged", "launched", "maximized", "mentored", "modernized",
    "optimized", "orchestrated", "pioneered", "redesigned", "reduced", "streamlined",
    "spearheaded", "strategized", "strengthened", "transformed", "triggered",
}

WEAK_ACTION_VERBS = {
    "helped", "worked", "did", "made", "tried", "went", "got", "made", "said",
    "took", "saw", "looked", "thought", "used", "want", "know",
}

GENERIC_PHRASES = {
    "responsible for": "led",
    "in charge of": "directed",
    "worked on": "engineered",
    "helped with": "facilitated",
    "involved in": "spearheaded",
    "was part of": "contributed to",
    "team player": "collaborative leader",
    "hard worker": "high performer",
    "fast learner": "quick to master",
    "good communication": "exceptional communicator",
    "problem solver": "strategic problem-solver",
    "multitasker": "adaptable professional",
}


def detect_weak_words(text: str) -> List[Dict[str, str]]:
    """Detect weak/cheap words in resume text and suggest replacements."""
    issues = []
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    
    for word in words:
        if word in WEAK_WORDS:
            suggestions = WEAK_WORDS[word]
            if isinstance(suggestions, tuple) and len(suggestions) > 0:
                suggestion = suggestions[0]
                issues.append({
                    "word": word,
                    "suggestion": suggestion,
                    "severity": "high" if word in ["cheap", "simple", "just"] else "medium",
                })
    
    return issues


def detect_weak_verbs(text: str) -> List[Dict[str, str]]:
    """Detect weak action verbs and suggest stronger alternatives."""
    issues = []
    words = re.findall(r"\b\w+\b", text.lower())
    
    for word in words:
        if word in WEAK_ACTION_VERBS:
            issues.append({
                "verb": word,
                "suggestion": list(STRONG_ACTION_VERBS)[0],  # Pick a strong alternative
                "type": "weak_verb",
            })
    
    return issues


def detect_generic_phrases(text: str) -> List[Dict[str, str]]:
    """Detect generic/cliché phrases and suggest improvements."""
    issues = []
    text_lower = text.lower()
    
    for phrase, replacement in GENERIC_PHRASES.items():
        if phrase in text_lower:
            issues.append({
                "phrase": phrase,
                "replacement": replacement,
                "type": "generic_phrase",
            })
    
    return issues


def analyze_word_quality(text: str) -> Dict:
    """Comprehensive word quality analysis."""
    weak_words = detect_weak_words(text)
    weak_verbs = detect_weak_verbs(text)
    generic_phrases = detect_generic_phrases(text)
    
    # Calculate scores
    weak_word_score = max(0, 100 - len(weak_words) * 5)
    verb_score = max(0, 100 - len(weak_verbs) * 3)
    generic_score = max(0, 100 - len(generic_phrases) * 4)
    
    word_quality_score = (weak_word_score + verb_score + generic_score) / 3
    
    return {
        "word_quality_score": round(word_quality_score, 2),
        "weak_words_found": len(weak_words),
        "weak_verbs_found": len(weak_verbs),
        "generic_phrases_found": len(generic_phrases),
        "weak_words": weak_words[:10],  # Top 10
        "weak_verbs": weak_verbs[:10],
        "generic_phrases": generic_phrases[:10],
        "improvements": _generate_word_improvements(weak_words, weak_verbs, generic_phrases),
    }


def _generate_word_improvements(weak_words: List, weak_verbs: List, generic_phrases: List) -> List[str]:
    """Generate actionable word improvement suggestions."""
    improvements = []
    
    if weak_words:
        top_weak = [w["word"] for w in weak_words[:3]]
        improvements.append(f"Replace weak words: {', '.join(top_weak)} with stronger alternatives")
    
    if weak_verbs:
        top_verbs = [v["verb"] for v in weak_verbs[:3]]
        improvements.append(f"Upgrade action verbs: {', '.join(top_verbs)} → use verbs like {', '.join(list(STRONG_ACTION_VERBS)[:3])}")
    
    if generic_phrases:
        top_phrases = [p["phrase"] for p in generic_phrases[:3]]
        improvements.append(f"Replace clichés: {', '.join(top_phrases)} with specific, impactful language")
    
    if not improvements:
        improvements.append("✓ Excellent word choice! Resume uses strong, professional language.")
    
    return improvements


def calculate_professionalism_score(text: str) -> float:
    """Calculate overall professionalism score based on language quality."""
    analysis = analyze_word_quality(text)
    
    # Weight components
    score = (
        analysis["word_quality_score"] * 0.4 +
        (100 - len(analysis["weak_verbs"]) * 5) * 0.3 +
        (100 - len(analysis["generic_phrases"]) * 4) * 0.3
    )
    
    return max(0, min(100, round(score, 2)))
