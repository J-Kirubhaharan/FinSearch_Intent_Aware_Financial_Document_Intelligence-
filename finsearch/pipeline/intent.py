"""
Intent Classifier — Fine-Tuned MiniLM
Classifies query into one of 4 categories or flags as OOS.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import INTENT_MODEL_PATH, ID2LABEL, OOS_CONFIDENCE_THRESHOLD

_tokenizer = None
_model     = None


def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
        _model     = AutoModelForSequenceClassification.from_pretrained(
            INTENT_MODEL_PATH
        ).to("cpu")
        _model.eval()


def classify_intent(query: str) -> tuple[str, float, bool]:
    """
    Returns (category, confidence, is_oos).
    is_oos=True means confidence < OOS_CONFIDENCE_THRESHOLD.
    """
    _load()
    inputs = _tokenizer(
        query, return_tensors="pt",
        truncation=True, max_length=128, padding=True
    )
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs      = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred_id    = int(np.argmax(probs))
    confidence = float(probs[pred_id])
    category   = ID2LABEL[pred_id]
    is_oos     = confidence < OOS_CONFIDENCE_THRESHOLD
    return category, confidence, is_oos
