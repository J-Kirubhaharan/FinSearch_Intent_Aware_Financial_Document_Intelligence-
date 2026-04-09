"""
RBAC Access Control
- filter_by_role()      : keeps only chunks this role is allowed to see
- is_blocked_question() : returns True if the question pattern is restricted
"""
from rbac.roles import ROLES


def filter_by_role(candidates: list[dict], user_role: str) -> list[dict]:
    """
    Filter retrieved chunks to only those the role is allowed to access.
    Checks both category-level and document-level access.
    """
    role_cfg = ROLES.get(user_role)
    if not role_cfg:
        return []

    allowed_categories = set(role_cfg.get("allowed_categories") or [])
    allowed_docs       = role_cfg.get("allowed_docs")  # None = all docs in category

    filtered = []
    for chunk in candidates:
        # Category check
        if chunk.get("category") not in allowed_categories:
            continue
        # Document-level check (if specific docs are listed)
        if allowed_docs is not None:
            doc_name = chunk["chunk_id"].rsplit("_chunk_", 1)[0]
            if not any(doc_name.startswith(d) or d in doc_name for d in allowed_docs):
                continue
        filtered.append(chunk)

    return filtered


def is_blocked_question(query: str, user_role: str) -> bool:
    """
    Returns True if the query matches a blocked pattern for this role.
    """
    role_cfg = ROLES.get(user_role, {})
    blocked  = role_cfg.get("blocked_patterns", [])
    q_lower  = query.lower()
    return any(pattern in q_lower for pattern in blocked)


def get_allowed_summary(user_role: str) -> dict:
    """Return human-readable summary of what this role can access."""
    role_cfg = ROLES.get(user_role, {})
    return {
        "categories": role_cfg.get("allowed_categories", []),
        "docs"      : role_cfg.get("allowed_docs", "all"),
    }
