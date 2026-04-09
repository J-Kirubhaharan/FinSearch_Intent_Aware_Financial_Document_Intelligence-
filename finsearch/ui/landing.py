"""
Landing Page — Role Selection Screen
"""
import streamlit as st
from rbac.roles import ROLES
from data.loader import get_all_users_list


def render_landing():
    """Render role selection page. Sets st.session_state on selection."""
    st.markdown("""
        <div style='text-align:center; padding: 2rem 0 1rem 0;'>
            <h1 style='font-size:2.5rem; margin-bottom:0.2rem;'>💼 FinSearch</h1>
            <p style='font-size:1.1rem; color:#666;'>
                Intent-Aware Financial Document Intelligence
            </p>
            <p style='font-size:0.95rem; color:#999; margin-top:0.5rem;'>
                Powered by Fine-Tuned MiniLM · Cross-Encoder · LLaMA-3.3-70B · DeBERTa Confidence
            </p>
        </div>
        <hr style='margin-bottom:2rem;'>
        <h3 style='text-align:center; margin-bottom:1.5rem;'>Who are you today?</h3>
    """, unsafe_allow_html=True)

    # ── Role Buttons — 2 columns x 3 rows ──────────────────────────────────────
    role_keys   = list(ROLES.keys())
    cols_row1   = st.columns(2)
    cols_row2   = st.columns(2)
    cols_row3   = st.columns([1, 2, 1])

    role_layout = [
        (cols_row1[0], role_keys[0]),  # customer
        (cols_row1[1], role_keys[1]),  # support_agent
        (cols_row2[0], role_keys[2]),  # fraud_investigator
        (cols_row2[1], role_keys[3]),  # merchant
        (cols_row3[1], role_keys[4]),  # dispute_resolver
    ]

    for col, role_key in role_layout:
        role = ROLES[role_key]
        with col:
            with st.container(border=True):
                st.markdown(f"### {role['label']}")
                st.caption(role["description"])
                if st.button("Select", key=f"select_{role_key}", use_container_width=True):
                    st.session_state["user_role"] = role_key
                    if role_key == "customer":
                        st.session_state["show_user_select"] = True
                    else:
                        st.session_state["user_id"]   = role_key
                        st.session_state["user_name"] = role["label"]
                        st.session_state["page"]      = "chat"
                    st.rerun()

    # ── Customer sub-screen: pick account ──────────────────────────────────────
    if st.session_state.get("show_user_select"):
        st.markdown("---")
        st.subheader("Select your account")
        users = get_all_users_list()
        options = {f"{u['name']} ({u['user_id']})": u["user_id"] for u in users}
        selected_label = st.selectbox("Choose account:", list(options.keys()))
        if st.button("Continue →", use_container_width=False):
            st.session_state["user_id"]          = options[selected_label]
            st.session_state["user_name"]        = selected_label.split(" (")[0]
            st.session_state["page"]             = "chat"
            st.session_state["show_user_select"] = False
            st.rerun()
