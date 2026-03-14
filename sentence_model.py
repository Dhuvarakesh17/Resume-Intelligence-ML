import os
from typing import Dict, Optional

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


_MODEL_CACHE: Dict[str, Optional[SentenceTransformer]] = {}


def get_sentence_transformer(model_name: Optional[str] = None):
    model_name = model_name or os.getenv("RESUME_SENTENCE_MODEL", "all-mpnet-base-v2")
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    if SentenceTransformer is None:
        _MODEL_CACHE[model_name] = None
        return None

    try:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    except Exception:
        _MODEL_CACHE[model_name] = None

    return _MODEL_CACHE[model_name]