"""
Chat Panel — Message rendering and confidence badges.
"""
import re
import streamlit as st


CONF_COLORS = {
    "HIGH"  : ("#d4edda", "#155724", "🟢 HIGH Confidence"),
    "MEDIUM": ("#fff3cd", "#856404", "🟡 MEDIUM Confidence"),
    "LOW"   : ("#f8d7da", "#721c24", "🔴 LOW Confidence"),
}


def render_chat_header(user_role: str, user_name: str):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 💼 FinSearch")
        st.caption(f"Logged in as: **{user_name}**")
    with col2:
        st.markdown("")


def render_message(role: str, content: str):
    """Render a single chat message bubble."""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="💼"):
            highlighted = _highlight_citations(content)
            st.markdown(highlighted, unsafe_allow_html=True)


def render_response_metadata(result: dict):
    """Show intent, confidence badge, and source chips below assistant message."""
    with st.expander("📊 Response Details", expanded=False):
        cols = st.columns(3)

        # Intent
        cols[0].markdown("**Intent Detected**")
        cols[0].markdown(
            f"`{result.get('intent', 'N/A')}`  \n"
            f"Confidence: `{result.get('intent_conf', 0):.2f}`"
        )

        # Confidence badge
        conf = result.get("confidence")
        if conf:
            label  = conf["label"]
            bg, fg, text = CONF_COLORS.get(label, ("#eee", "#333", label))
            cols[1].markdown("**Answer Confidence**")
            cols[1].markdown(
                f"<span style='background:{bg}; color:{fg}; padding:4px 10px; "
                f"border-radius:12px; font-weight:bold; font-size:0.9rem;'>{text}</span>  \n"
                f"<small>Retrieval: {conf['retrieval_confidence']} · "
                f"Faithfulness: {conf['faithfulness_confidence']}</small>",
                unsafe_allow_html=True,
            )

        # Sources
        top_chunks = result.get("top_chunks", [])
        if top_chunks:
            cols[2].markdown("**Sources Used**")
            for chunk in top_chunks:
                doc = chunk["chunk_id"].rsplit("_chunk_", 1)[0]
                cols[2].markdown(
                    f"<span style='background:#e9ecef; padding:2px 8px; "
                    f"border-radius:8px; font-size:0.8rem;'>📄 {doc}</span>",
                    unsafe_allow_html=True,
                )

        # RBAC / OOS flags
        if result.get("rbac_blocked"):
            st.warning("⚠️ This query was restricted by your access level.")
        if result.get("oos_handled"):
            st.info("ℹ️ This question was outside the financial domain — polite refusal sent.")


def _highlight_citations(text: str) -> str:
    """Wrap [Source: ...] in styled spans for visibility."""
    return re.sub(
        r"\[Source:\s*([^\]]+)\]",
        r"<span style='background:#e8f4f8; color:#0077b6; font-size:0.8rem; "
        r"padding:1px 6px; border-radius:4px; font-weight:500;'>"
        r"📄 Source: \1</span>",
        text,
    )


def render_input_bar() -> str | None:
    """Render chat input and return query string or None."""
    # Check for preset query injected from sidebar
    preset = st.session_state.pop("preset_query", None)
    return st.chat_input(
        "Ask a financial question...",
        key="chat_input",
    ) or preset
