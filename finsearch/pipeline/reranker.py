"""
Cross-Encoder Reranker — ms-marco-MiniLM-L-6-v2
Scores (query, chunk) pairs and returns top-K most relevant.
"""
from sentence_transformers import CrossEncoder
from config import CE_MODEL_NAME, RERANK_TOP_K

_ce_model = None


def _get_model() -> CrossEncoder:
    global _ce_model
    if _ce_model is None:
        _ce_model = CrossEncoder(CE_MODEL_NAME, max_length=512, device="cpu")
    return _ce_model


def rerank(query: str, candidates: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    Rerank candidate chunks using Cross-Encoder.
    Returns top_k chunks sorted by relevance score.
    """
    if not candidates:
        return []
    model  = _get_model()
    pairs  = [(query, c["text"]) for c in candidates]
    scores = _get_model().predict(pairs).tolist()
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top    = [c for _, c in ranked[:top_k]]
    # Attach rerank score for transparency
    for chunk, (score, _) in zip(top, ranked[:top_k]):
        chunk["rerank_score"] = round(score, 4)
    return top
