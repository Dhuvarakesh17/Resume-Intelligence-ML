import os
from typing import Dict, Optional, Tuple

try:
    import spacy
except Exception:
    spacy = None


_NLP_CACHE: Dict[Tuple[str, str], object] = {}


def _candidate_models() -> list[str]:
    configured = os.getenv("RESUME_SPACY_MODEL", "en_core_web_trf,en_core_web_sm")
    return [name.strip() for name in configured.split(",") if name.strip()]


def get_spacy_model(disable: Optional[list[str]] = None, fallback_blank: bool = True):
    if spacy is None:
        return None

    disable = disable or []
    cache_key = ("|".join(_candidate_models()), "|".join(disable))
    if cache_key in _NLP_CACHE:
        return _NLP_CACHE[cache_key]

    for model_name in _candidate_models():
        try:
            nlp = spacy.load(model_name, disable=disable)
            _NLP_CACHE[cache_key] = nlp
            return nlp
        except Exception:
            continue

    if fallback_blank:
        try:
            nlp = spacy.blank("en")
            _NLP_CACHE[cache_key] = nlp
            return nlp
        except Exception:
            pass

    _NLP_CACHE[cache_key] = None
    return None