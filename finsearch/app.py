"""
FinSearch — Main Streamlit App
Run: streamlit run app.py
"""
import os, sys
import streamlit as st

# Ensure finsearch/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indexing.index_manager import load, index_is_ready
from pipeline.retrieval     import init_retrieval
from pipeline.chatbot       import chatbot
from data.loader            import build_personal_context
from ui.landing             import render_landing
from ui.sidebar             import render_sidebar
from ui.chat                import (
    render_chat_header, render_message,
    render_response_metadata, render_input_bar,
)
from config import OPENROUTER_API_KEY

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSearch",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Index Once ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FinSearch models and index...")
def load_pipeline():
    """Load FAISS index + all models once. Cached across sessions."""
    import gc
    if not index_is_ready():
        return None

    # Step 1: Load FAISS index from disk
    faiss_index, corpus_texts, corpus_ids, corpus_cats = load()
    init_retrieval(faiss_index, corpus_texts, corpus_ids, corpus_cats)
    gc.collect()

    # Step 2: Pre-warm intent classifier
    from pipeline.intent import _load as load_intent
    load_intent()
    gc.collect()

    # Step 3: Pre-warm fine-tuned MiniLM retriever
    from pipeline.retrieval import _get_ft_model
    _get_ft_model()
    gc.collect()

    # Step 4: Pre-warm Cross-Encoder reranker
    from pipeline.reranker import _get_model as get_ce
    get_ce()
    gc.collect()

    # Step 5: Pre-warm DeBERTa confidence scorer (load last — heaviest)
    from pipeline.confidence import _get_model as get_nli
    get_nli()
    gc.collect()

    return True


# ── API Key Check ──────────────────────────────────────────────────────────────
def check_api_key():
    if not OPENROUTER_API_KEY:
        st.error(
            "OpenRouter API key not found. "
            "Set it as environment variable: `export OPENROUTER_API_KEY=your_key`"
        )
        st.stop()


# ── Index Not Ready Screen ─────────────────────────────────────────────────────
def show_setup_screen():
    st.title("💼 FinSearch — Setup Required")
    st.warning(
        "No FAISS index found. You need to build the index before running the app."
    )
    st.markdown("""
    **Steps to set up:**

    1. Copy your PDFs into the knowledge base folders:
    ```
    finsearch/knowledge_base/Regulatory/
    finsearch/knowledge_base/Consumer_Protection/
    finsearch/knowledge_base/Payment_Industry/
    finsearch/knowledge_base/Synthetic_Policies/
    ```

    2. Copy your model folders:
    ```
    finsearch/models/minilm_finetuned/
    finsearch/models/minilm_intent_classifier/
    ```

    3. Build the index (run once from the finsearch/ folder):
    ```bash
    python -m indexing.index_manager build
    ```

    4. Restart the app:
    ```bash
    streamlit run app.py
    ```
    """)

    st.markdown("---")
    st.markdown("### Or if you already have an index elsewhere:")
    if st.button("🔄 Check Again"):
        st.cache_resource.clear()
        st.rerun()


# ── Main App ───────────────────────────────────────────────────────────────────
def main():
    check_api_key()

    # Load pipeline
    pipeline_ready = load_pipeline()
    if not pipeline_ready:
        show_setup_screen()
        return

    # Initialise session state
    if "page"     not in st.session_state: st.session_state["page"]     = "landing"
    if "messages" not in st.session_state: st.session_state["messages"] = []

    # ── Landing Page ───────────────────────────────────────────────────────────
    if st.session_state["page"] == "landing":
        render_landing()
        return

    # ── Chat Page ──────────────────────────────────────────────────────────────
    user_role = st.session_state.get("user_role", "customer")
    user_id   = st.session_state.get("user_id", "anonymous")
    user_name = st.session_state.get("user_name", "User")

    # Sidebar
    render_sidebar(user_role, user_id)

    # Chat header
    render_chat_header(user_role, user_name)
    st.markdown("---")

    # Render existing messages
    for msg in st.session_state["messages"]:
        render_message(msg["role"], msg["content"])
        if msg["role"] == "assistant" and "result" in msg:
            render_response_metadata(msg["result"])

    # ── Handle new input ───────────────────────────────────────────────────────
    query = render_input_bar()

    if query:
        # Show user message immediately
        st.session_state["messages"].append({"role": "user", "content": query})
        render_message("user", query)

        # Build personal context (only meaningful for customer role)
        personal_ctx = build_personal_context(user_id, user_role)

        # Run pipeline
        with st.spinner("Searching knowledge base..."):
            result = chatbot(
                query=query,
                user_role=user_role,
                user_id=user_id,
                personal_context=personal_ctx,
            )

        # Show assistant response
        render_message("assistant", result["response"])
        render_response_metadata(result)

        # Save to session
        st.session_state["messages"].append({
            "role"   : "assistant",
            "content": result["response"],
            "result" : result,
        })
        st.rerun()


if __name__ == "__main__":
    main()
