"""
Embeds text chunks using the fine-tuned MiniLM model.
Only called when new PDFs are indexed — not on every app start.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from config import FT_MODEL_PATH

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(FT_MODEL_PATH)
    return _model


def embed_chunks(texts: list[str], batch_size: int = 128) -> np.ndarray:
    """
    Embed a list of text strings.
    Returns float32 numpy array of shape (n, embedding_dim).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return embeddings
