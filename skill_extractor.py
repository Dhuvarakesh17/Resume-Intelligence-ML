import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.error import URLError
from urllib.request import urlopen

try:
    import spacy
    from spacy.matcher import PhraseMatcher
except Exception:
    spacy = None
    PhraseMatcher = None

from spacy_model import get_spacy_model

LOCAL_SKILL_FILE = Path(__file__).with_name("skill_dictionary.json")
ONLINE_SKILL_SOURCE = (
    "https://api.stackexchange.com/2.3/tags?order=desc&sort=popular&site=stackoverflow&pagesize=200"
)
CACHE_TTL_SECONDS = 60 * 60 * 6

_NLP = None
_MATCHER = None
_MATCHER_VERSION: Optional[str] = None
_SKILL_CACHE: Optional[Dict[str, List[str]]] = None
_SKILL_CACHE_TS: float = 0.0
_SKILL_CACHE_VERSION: Optional[str] = None

_TAG_BLACKLIST = {
    "arrays", "string", "regex", "database", "performance", "algorithm", "list",
    "json", "html", "css", "excel", "linux", "windows", "vba",
}

_CANONICAL_OVERRIDES = {
    "reactjs": "React",
    "react.js": "React",
    "node.js": "Node.js",
    "python-3.x": "Python",
    "spring-boot": "Spring Boot",
    "k8s": "Kubernetes",
    "dotnet": ".NET",
    "asp.net": "ASP.NET",
    "asp.net-mvc": "ASP.NET MVC",
}


def _canonicalize_tag(tag: str) -> str:
    if tag in _CANONICAL_OVERRIDES:
        return _CANONICAL_OVERRIDES[tag]

    cleaned = tag.replace("-", " ").replace("_", " ").strip()
    if not cleaned:
        return tag

    return " ".join(part.upper() if len(part) <= 2 else part.title() for part in cleaned.split())


def _merge_skill(mapping: Dict[str, Set[str]], canonical: str, aliases: List[str]) -> None:
    values = mapping.setdefault(canonical, set())
    values.add(canonical.lower())
    for alias in aliases:
        alias_norm = alias.strip().lower()
        if alias_norm:
            values.add(alias_norm)


def _load_local_skills() -> Dict[str, Set[str]]:
    if not LOCAL_SKILL_FILE.exists():
        return {}

    with LOCAL_SKILL_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    loaded: Dict[str, Set[str]] = {}
    for canonical, aliases in payload.items():
        if isinstance(canonical, str) and isinstance(aliases, list):
            _merge_skill(loaded, canonical.strip(), [str(alias) for alias in aliases])

    return loaded


def _load_online_skills() -> Dict[str, Set[str]]:
    extracted: Dict[str, Set[str]] = {}
    try:
        with urlopen(ONLINE_SKILL_SOURCE, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        return extracted

    for item in payload.get("items", []):
        tag = str(item.get("name", "")).strip().lower()
        if not tag or tag in _TAG_BLACKLIST:
            continue
        if len(tag) < 2 or re.fullmatch(r"\d+(\.\d+)?", tag):
            continue

        canonical = _canonicalize_tag(tag)
        _merge_skill(extracted, canonical, [tag])

    return extracted


def _serialize_skills(mapping: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    return {canonical: sorted(values) for canonical, values in mapping.items()}


def get_skill_dictionary(force_refresh: bool = False) -> Dict[str, List[str]]:
    global _SKILL_CACHE, _SKILL_CACHE_TS, _SKILL_CACHE_VERSION

    cache_valid = (time.time() - _SKILL_CACHE_TS) < CACHE_TTL_SECONDS
    if not force_refresh and _SKILL_CACHE is not None and cache_valid:
        return _SKILL_CACHE

    merged = _load_local_skills()
    online_skills = _load_online_skills()
    for canonical, aliases in online_skills.items():
        _merge_skill(merged, canonical, list(aliases))

    _SKILL_CACHE = _serialize_skills(merged)
    _SKILL_CACHE_TS = time.time()
    _SKILL_CACHE_VERSION = f"{len(_SKILL_CACHE)}:{int(_SKILL_CACHE_TS)}"
    return _SKILL_CACHE


def _build_matcher(skill_dictionary: Dict[str, List[str]]):
    global _NLP, _MATCHER, _MATCHER_VERSION

    if spacy is None:
        return None, None

    target_version = _SKILL_CACHE_VERSION
    if _NLP is not None and _MATCHER is not None and _MATCHER_VERSION == target_version:
        return _NLP, _MATCHER

    _NLP = get_spacy_model(disable=["parser", "ner", "textcat"])
    if _NLP is None:
        return None, None

    _MATCHER = PhraseMatcher(_NLP.vocab, attr="LOWER")
    for canonical, aliases in skill_dictionary.items():
        terms = sorted(set([canonical, *aliases]))
        patterns = [_NLP.make_doc(term) for term in terms if term.strip()]
        if patterns:
            _MATCHER.add(canonical, patterns)

    _MATCHER_VERSION = target_version
    return _NLP, _MATCHER


def extract_skills(text: str, force_refresh_dictionary: bool = False) -> List[str]:
    if not text:
        return []

    skill_dictionary = get_skill_dictionary(force_refresh=force_refresh_dictionary)
    nlp, matcher = _build_matcher(skill_dictionary)
    found = set()

    if nlp is not None and matcher is not None:
        doc = nlp(text)
        for match_id, _, _ in matcher(doc):
            found.add(nlp.vocab.strings[match_id])
    else:
        lowered = text.lower()
        for canonical, aliases in skill_dictionary.items():
            terms = [canonical.lower(), *aliases]
            if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms):
                found.add(canonical)

    return sorted(found)
