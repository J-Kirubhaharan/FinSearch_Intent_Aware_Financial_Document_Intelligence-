# coding: utf-8
"""
Hybrid Retrieval - FAISS (dense) + BM25 (sparse) with Alpha Fusion
Alpha=0.7 validated in hybrid experiments (best NDCG@10).
Formula: final = 0.7 * dense_norm + 0.3 * bm25_norm (min-max normalised)
"""
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from config import FT_MODEL_PATH, RETRIEVAL_TOP_K

ALPHA = 0.7  # dense weight; (1-ALPHA) = BM25 weight

_ft_model    = None
_faiss_index = None
_corpus_texts = None
_corpus_ids   = None
_corpus_cats  = None
_bm25         = None          # built once at init from corpus_texts


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Returns zeros if all equal."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-9:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


# ── Init ───────────────────────────────────────────────────────────────────────

def init_retrieval(faiss_index, corpus_texts, corpus_ids, corpus_cats):
    """Called once at app start. Builds BM25 index from corpus."""
    global _faiss_index, _corpus_texts, _corpus_ids, _corpus_cats, _bm25
    _faiss_index  = faiss_index
    _corpus_texts = corpus_texts
    _corpus_ids   = corpus_ids
    _corpus_cats  = corpus_cats
    tokenised     = [_tokenize(t) for t in corpus_texts]
    _bm25         = BM25Okapi(tokenised)
    print(f"[retrieval] BM25 index built over {len(corpus_texts)} chunks")


def _get_ft_model() -> SentenceTransformer:
    global _ft_model
    if _ft_model is None:
        _ft_model = SentenceTransformer(FT_MODEL_PATH, device="cpu")
    return _ft_model


# ── Retrieve ───────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """
    Hybrid retrieval: FAISS + BM25 fused with alpha=0.7.
    Returns list of {chunk_id, text, category, ret_score} sorted by blended score.
    """
    n = len(_corpus_texts)
    pool = min(top_k * 3, n)   # retrieve more candidates before fusion

    # ── Dense (FAISS) ──────────────────────────────────────────────────────────
    model = _get_ft_model()
    q_emb = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    faiss_scores_raw, faiss_indices = _faiss_index.search(q_emb, pool)
    faiss_scores_raw = faiss_scores_raw[0]
    faiss_indices    = faiss_indices[0]

    # Build dense score array over full corpus (0 for non-retrieved)
    dense_full = np.zeros(n, dtype=np.float32)
    for idx, score in zip(faiss_indices, faiss_scores_raw):
        if idx >= 0:
            dense_full[idx] = score

    # ── Sparse (BM25) ──────────────────────────────────────────────────────────
    bm25_full = np.array(_bm25.get_scores(_tokenize(query)), dtype=np.float32)

    # ── Alpha Fusion ───────────────────────────────────────────────────────────
    dense_norm  = _minmax(dense_full)
    bm25_norm   = _minmax(bm25_full)
    blended     = ALPHA * dense_norm + (1 - ALPHA) * bm25_norm

    # Top-k by blended score
    top_indices = np.argsort(blended)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if blended[idx] < 1e-9:
            continue
        results.append({
            "chunk_id" : _corpus_ids[idx],
            "text"     : _corpus_texts[idx],
            "category" : _corpus_cats[idx],
            "ret_score": float(blended[idx]),
        })
    return results
