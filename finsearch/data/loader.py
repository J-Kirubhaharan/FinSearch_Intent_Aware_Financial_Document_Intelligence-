"""
Personal Data Loader
Loads users, transactions, disputes from CSV/DB.
Formats personal context as text for LLM prompt injection.
"""
import sqlite3
import pandas as pd
from functools import lru_cache
from config import USERS_DB_PATH, TRANSACTIONS_PATH, DISPUTES_PATH


# ── Raw Data Loaders (cached) ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_transactions() -> pd.DataFrame:
    return pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"])


@lru_cache(maxsize=1)
def load_disputes() -> pd.DataFrame:
    return pd.read_csv(
        DISPUTES_PATH,
        parse_dates=["date_opened", "date_resolved", "sla_deadline"]
    )


@lru_cache(maxsize=1)
def load_users() -> pd.DataFrame:
    conn = sqlite3.connect(USERS_DB_PATH)
    df   = pd.read_sql("SELECT * FROM users", conn)
    conn.close()
    return df


@lru_cache(maxsize=1)
def load_sessions() -> pd.DataFrame:
    conn = sqlite3.connect(USERS_DB_PATH)
    df   = pd.read_sql("SELECT * FROM sessions", conn)
    conn.close()
    return df


# ── Per-User Getters ───────────────────────────────────────────────────────────

def get_user(user_id: str) -> dict | None:
    users = load_users()
    row   = users[users["user_id"] == user_id]
    return row.iloc[0].to_dict() if not row.empty else None


def get_user_transactions(user_id: str, limit: int = 10) -> pd.DataFrame:
    txns = load_transactions()
    return txns[txns["user_id"] == user_id].sort_values("date", ascending=False).head(limit)


def get_user_disputes(user_id: str) -> pd.DataFrame:
    disp = load_disputes()
    return disp[disp["user_id"] == user_id].sort_values("date_opened", ascending=False)


def get_user_sessions(user_id: str, limit: int = 5) -> pd.DataFrame:
    sess = load_sessions()
    return sess[sess["user_id"] == user_id].tail(limit)


# ── Role-Specific Data Views ───────────────────────────────────────────────────

def get_fraud_disputes() -> pd.DataFrame:
    """All active fraud disputes — for Fraud Investigator."""
    disp = load_disputes()
    return disp[
        disp["dispute_type"].isin(["fraud_dispute", "unauthorized_charge"]) &
        (disp["status"] != "resolved")
    ].sort_values("priority", ascending=False)


def get_overdue_disputes() -> pd.DataFrame:
    """All open disputes sorted by SLA deadline — for Dispute Resolver."""
    import pandas as pd
    disp = load_disputes()
    open_disp = disp[disp["status"] != "resolved"].copy()
    open_disp["sla_deadline"] = pd.to_datetime(open_disp["sla_deadline"])
    return open_disp.sort_values("sla_deadline", ascending=True)


def get_merchant_summary() -> pd.DataFrame:
    """Transaction failure rates by country — for Merchant."""
    txns = load_transactions()
    summary = (
        txns.groupby("country")
        .agg(
            total    =("txn_id", "count"),
            failed   =("status", lambda x: (x.isin(["failed", "declined"])).sum()),
        )
        .reset_index()
    )
    summary["fail_rate"] = (summary["failed"] / summary["total"] * 100).round(1)
    return summary.sort_values("fail_rate", ascending=False)


def get_all_users_list() -> list[dict]:
    """Returns list of {user_id, name} for customer login dropdown."""
    users = load_users()
    return users[["user_id", "name"]].to_dict("records")


# ── Personal Context Formatter ─────────────────────────────────────────────────

def build_personal_context(user_id: str, user_role: str) -> str:
    """
    Formats relevant personal data as a text block for the LLM prompt.
    Only called for customer role (others use aggregate views).
    """
    if user_role != "customer":
        return ""

    user = get_user(user_id)
    if not user:
        return ""

    txns     = get_user_transactions(user_id, limit=5)
    disputes = get_user_disputes(user_id)
    open_disputes = disputes[disputes["status"] != "resolved"]

    lines = [f"User: {user['name']} (Account age: {user['account_age_days']} days)"]

    if not txns.empty:
        lines.append("\nRecent Transactions:")
        for _, row in txns.iterrows():
            status_icon = "✅" if row["status"] == "completed" else "❌"
            lines.append(
                f"  {status_icon} {row['merchant']} — {row['currency']}{row['amount']:.2f} "
                f"| {row['status']} | {row['reason']} | {str(row['date'])[:10]}"
            )

    if not open_disputes.empty:
        lines.append("\nOpen Disputes:")
        for _, row in open_disputes.iterrows():
            lines.append(
                f"  [{row['dispute_id']}] {row['merchant']} — {row['currency']}{row['amount']:.2f} "
                f"| Type: {row['dispute_type']} | Priority: {row['priority']} "
                f"| SLA Deadline: {str(row['sla_deadline'])[:10]}"
            )

    return "\n".join(lines)
