"""
Sidebar — Role-Specific Data Views
"""
import streamlit as st
import pandas as pd
from data.loader import (
    get_user, get_user_transactions, get_user_disputes,
    get_fraud_disputes, get_overdue_disputes, get_merchant_summary,
    load_disputes,
)
from rbac.roles import ROLES


def render_sidebar(user_role: str, user_id: str):
    with st.sidebar:
        role_cfg = ROLES[user_role]
        st.markdown(f"## {role_cfg['label']}")

        # Switch role button
        if st.button("← Switch Role", use_container_width=True):
            for key in ["user_role", "user_id", "user_name", "page",
                        "messages", "show_user_select"]:
                st.session_state.pop(key, None)
            st.rerun()

        st.markdown("---")

        if user_role == "customer":
            _sidebar_customer(user_id)
        elif user_role == "support_agent":
            _sidebar_support_agent()
        elif user_role == "fraud_investigator":
            _sidebar_fraud_investigator()
        elif user_role == "merchant":
            _sidebar_merchant()
        elif user_role == "dispute_resolver":
            _sidebar_dispute_resolver()

        st.markdown("---")
        _sidebar_preset_questions(user_role)


# ── Per-Role Sidebars ──────────────────────────────────────────────────────────

def _sidebar_customer(user_id: str):
    user = get_user(user_id)
    if user:
        st.markdown(f"**{user['name']}**")
        st.caption(f"Account age: {user['account_age_days']} days · {user['user_type'].title()} user")

    st.markdown("### My Transactions")
    txns = get_user_transactions(user_id, limit=8)
    if txns.empty:
        st.info("No transactions found.")
    else:
        for _, row in txns.iterrows():
            icon = "✅" if row["status"] == "completed" else "❌"
            st.markdown(
                f"{icon} **{row['merchant']}** — {row['currency']}{row['amount']:.2f}  \n"
                f"<small>{row['status']} · {row['reason']} · {str(row['date'])[:10]}</small>",
                unsafe_allow_html=True,
            )

    st.markdown("### My Disputes")
    disputes = get_user_disputes(user_id)
    open_d   = disputes[disputes["status"] != "resolved"]
    if open_d.empty:
        st.success("No open disputes.")
    else:
        for _, row in open_d.iterrows():
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(row["priority"], "⚪")
            st.markdown(
                f"{priority_color} **{row['dispute_id']}** — {row['merchant']}  \n"
                f"<small>{row['dispute_type']} · SLA: {str(row['sla_deadline'])[:10]}</small>",
                unsafe_allow_html=True,
            )


def _sidebar_support_agent():
    disp = load_disputes()
    open_d = disp[disp["status"] != "resolved"]

    col1, col2 = st.columns(2)
    col1.metric("Open Disputes", len(open_d))
    col2.metric("Escalated", len(open_d[open_d["status"] == "escalated"]))

    st.markdown("### Recent Complaints")
    recent = open_d.head(6)
    for _, row in recent.iterrows():
        st.markdown(
            f"**{row['dispute_id']}** — {row['merchant']}  \n"
            f"<small>{row['dispute_type']} · {row['status']}</small>",
            unsafe_allow_html=True,
        )


def _sidebar_fraud_investigator():
    fraud_disp = get_fraud_disputes()

    col1, col2 = st.columns(2)
    col1.metric("Active Fraud Cases", len(fraud_disp))
    total_amount = fraud_disp["amount"].sum()
    col2.metric("Total at Risk", f"£{total_amount:,.0f}")

    st.markdown("### Active Fraud Cases")
    for _, row in fraud_disp.head(8).iterrows():
        priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(row["priority"], "⚪")
        st.markdown(
            f"{priority_color} **{row['dispute_id']}** — {row['currency']}{row['amount']:.2f}  \n"
            f"<small>{row['dispute_type']} · {row['status']}</small>",
            unsafe_allow_html=True,
        )


def _sidebar_merchant():
    summary = get_merchant_summary()
    txns    = load_disputes()

    st.markdown("### Transaction Failure by Country")
    for _, row in summary.head(8).iterrows():
        color = "🔴" if row["fail_rate"] > 25 else ("🟡" if row["fail_rate"] > 10 else "🟢")
        st.markdown(
            f"{color} **{row['country']}** — {row['fail_rate']}% fail rate  \n"
            f"<small>{row['failed']} failed / {row['total']} total</small>",
            unsafe_allow_html=True,
        )


def _sidebar_dispute_resolver():
    overdue = get_overdue_disputes()
    today   = pd.Timestamp.now()

    past_sla  = overdue[overdue["sla_deadline"] < today]
    this_week = overdue[
        (overdue["sla_deadline"] >= today) &
        (overdue["sla_deadline"] <= today + pd.Timedelta(days=7))
    ]
    on_track  = overdue[overdue["sla_deadline"] > today + pd.Timedelta(days=7)]

    col1, col2, col3 = st.columns(3)
    col1.metric("⚠️ Overdue", len(past_sla))
    col2.metric("🔴 This Week", len(this_week))
    col3.metric("✅ On Track", len(on_track))

    if not past_sla.empty:
        st.markdown("### ⚠️ Overdue")
        for _, row in past_sla.head(4).iterrows():
            days_over = (today - row["sla_deadline"]).days
            st.markdown(
                f"🔴 **{row['dispute_id']}** — {row['merchant']}  \n"
                f"<small>{days_over} days overdue · {row['dispute_type']}</small>",
                unsafe_allow_html=True,
            )

    if not this_week.empty:
        st.markdown("### 🔴 Due This Week")
        for _, row in this_week.head(4).iterrows():
            days_left = (row["sla_deadline"] - today).days
            st.markdown(
                f"🟡 **{row['dispute_id']}** — {row['merchant']}  \n"
                f"<small>{days_left} days left · {row['dispute_type']}</small>",
                unsafe_allow_html=True,
            )


def _sidebar_preset_questions(user_role: str):
    """Clickable preset question buttons."""
    from rbac.roles import ROLES
    presets = ROLES[user_role].get("preset_questions", [])
    if not presets:
        return
    st.markdown("### 💬 Try Asking:")
    for q in presets:
        if st.button(q, key=f"preset_{q[:30]}", use_container_width=True):
            st.session_state["preset_query"] = q
            st.rerun()
