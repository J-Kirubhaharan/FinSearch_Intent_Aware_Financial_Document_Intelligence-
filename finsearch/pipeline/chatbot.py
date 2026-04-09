"""
Master Chatbot Pipeline
Wires: Intent → RBAC check → Retrieve → Filter by RBAC → Rerank → Generate → Confidence
"""
import json, os
from datetime import datetime

from pipeline.intent     import classify_intent
from pipeline.retrieval  import retrieve
from pipeline.reranker   import rerank
from pipeline.generator  import (
    generate_answer, OOS_RESPONSE, RBAC_BLOCKED_RESPONSE,
    CONF_MEDIUM_PREFIX, CONF_LOW_PREFIX
)
from pipeline.confidence import compute_confidence
from rbac.access_control import filter_by_role, is_blocked_question
from config import INDEX_DIR


# ── Query Log ──────────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(INDEX_DIR, "query_log.jsonl")

def _log(entry: dict):
    """Append one query result to the JSONL log."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main Pipeline ──────────────────────────────────────────────────────────────
def chatbot(
    query: str,
    user_role: str,
    user_id: str = "anonymous",
    personal_context: str = "",
) -> dict:
    """
    Full pipeline with RBAC.

    Parameters
    ----------
    query            : user's question
    user_role        : one of customer / support_agent / fraud_investigator /
                       merchant / dispute_resolver
    user_id          : for logging
    personal_context : formatted string from CSV/DB for this user (optional)

    Returns
    -------
    dict with keys: response, intent, intent_conf, is_oos, confidence,
                    top_chunks, oos_handled, rbac_blocked
    """
    result = {
        "query"       : query,
        "user_role"   : user_role,
        "user_id"     : user_id,
        "timestamp"   : datetime.utcnow().isoformat(),
        "rbac_blocked": False,
        "oos_handled" : False,
    }

    # ── Step 1: Intent Classification ─────────────────────────────────────────
    category, intent_conf, is_oos = classify_intent(query)
    result.update({
        "intent"     : category,
        "intent_conf": round(intent_conf, 4),
        "is_oos"     : is_oos,
    })

    # ── Step 2: OOS Check ─────────────────────────────────────────────────────
    if is_oos:
        result.update({
            "response"   : OOS_RESPONSE,
            "top_chunks" : [],
            "confidence" : None,
            "oos_handled": True,
        })
        _log({**result, "response": "[OOS]"})
        return result

    # ── Step 3: RBAC — Block restricted questions ──────────────────────────────
    if is_blocked_question(query, user_role):
        result.update({
            "response"    : RBAC_BLOCKED_RESPONSE,
            "top_chunks"  : [],
            "confidence"  : None,
            "rbac_blocked": True,
        })
        _log({**result, "response": "[RBAC_BLOCKED]"})
        return result

    # ── Step 4: Retrieval ──────────────────────────────────────────────────────
    candidates = retrieve(query, top_k=20)

    # ── Step 5: RBAC — Filter chunks by role ──────────────────────────────────
    allowed = filter_by_role(candidates, user_role)

    if not allowed:
        result.update({
            "response"    : RBAC_BLOCKED_RESPONSE,
            "top_chunks"  : [],
            "confidence"  : None,
            "rbac_blocked": True,
        })
        _log({**result, "response": "[NO_ALLOWED_CHUNKS]"})
        return result

    # ── Step 6: Rerank ─────────────────────────────────────────────────────────
    top_chunks = rerank(query, allowed, top_k=3)

    # ── Step 6b: Drop irrelevant chunks only for personal account queries
    # Only applies when: query is personal ("my payment", "I was charged") AND
    # personal_context is available AND FAISS returned low-scoring chunks.
    # Policy queries ("refund policy", "what are my rights") always keep FAISS chunks
    # so all 5 user roles get proper document-backed answers.
    _q_lower = f" {query.lower()} "
    _personal_signals = {"my ", " i ", "i've", "i was", "i am", "i'm", "i have", "i did"}
    _is_personal_query = any(sig in _q_lower for sig in _personal_signals)

    if _is_personal_query and personal_context and top_chunks:
        best_ret = max(c["ret_score"] for c in top_chunks)
        if best_ret < 0.3:
            top_chunks = []

    # ── Step 7: Generate Answer ────────────────────────────────────────────────
    raw_answer, context = generate_answer(query, top_chunks, personal_context)

    # ── Step 8: Confidence Score ───────────────────────────────────────────────
    ret_scores = [c["ret_score"] for c in top_chunks]
    if ret_scores:
        conf = compute_confidence(raw_answer, top_chunks, ret_scores)
    else:
        # No document chunks used — answered purely from personal context
        conf = {
            "label"                  : "HIGH",
            "final_confidence"       : 1.0,
            "retrieval_confidence"   : 1.0,
            "faithfulness_confidence": 1.0,
        }

    # ── Step 9: Apply Confidence Prefix ───────────────────────────────────────
    if conf["label"] == "MEDIUM":
        final_response = CONF_MEDIUM_PREFIX + raw_answer
    elif conf["label"] == "LOW":
        final_response = CONF_LOW_PREFIX + raw_answer
    else:
        final_response = raw_answer

    result.update({
        "response"  : final_response,
        "raw_answer": raw_answer,
        "top_chunks": top_chunks,
        "context"   : context,
        "confidence": conf,
    })

    # ── Step 10: Log ───────────────────────────────────────────────────────────
    _log({
        "timestamp"   : result["timestamp"],
        "user_id"     : user_id,
        "user_role"   : user_role,
        "query"       : query,
        "intent"      : category,
        "intent_conf" : round(intent_conf, 4),
        "confidence"  : conf["label"],
        "final_score" : conf["final_confidence"],
        "sources"     : [c["chunk_id"].rsplit("_chunk_", 1)[0] for c in top_chunks],
        "is_oos"      : False,
        "rbac_blocked": False,
    })

    return result
