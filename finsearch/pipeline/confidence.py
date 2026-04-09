"""
DeBERTa NLI Confidence Scorer
Computes retrieval + faithfulness confidence and labels HIGH/MEDIUM/LOW.
"""
import numpy as np
from sentence_transformers import CrossEncoder
from config import NLI_MODEL_NAME, CONF_HIGH, CONF_MEDIUM

_nli_model = None


def _get_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder(
            NLI_MODEL_NAME,
            default_activation_function=None,
            device="cpu",
        )
    return _nli_model


def compute_confidence(
    answer: str,
    top_chunks: list[dict],
    ret_scores: list[float],
) -> dict:
    """
    Returns {retrieval_confidence, faithfulness_confidence, final_confidence, label}.
    Weights: retrieval 40%, faithfulness 60%.
    """
    # Retrieval confidence — normalised FAISS scores
    s_min, s_max   = min(ret_scores), max(ret_scores)
    norm           = [(s - s_min) / (s_max - s_min + 1e-9) for s in ret_scores]
    retrieval_conf = float(np.mean(norm))

    # Faithfulness — NLI entailment(chunk, answer)
    model = _get_model()
    entailment_scores = []
    for chunk in top_chunks:
        logits = model.predict([[chunk["text"][:500], answer]])
        probs  = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        entailment_scores.append(float(probs[0][2]))  # index 2 = entailment

    faithfulness_conf = float(np.mean(entailment_scores)) if entailment_scores else 0.0

    final_conf = round(0.4 * retrieval_conf + 0.6 * faithfulness_conf, 4)
    label = (
        "HIGH"   if final_conf >= CONF_HIGH   else
        "MEDIUM" if final_conf >= CONF_MEDIUM else
        "LOW"
    )

    return {
        "retrieval_confidence"  : round(retrieval_conf, 4),
        "faithfulness_confidence": round(faithfulness_conf, 4),
        "final_confidence"      : final_conf,
        "label"                 : label,
    }
