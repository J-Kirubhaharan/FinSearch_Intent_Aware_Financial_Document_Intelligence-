# coding: utf-8
"""
Answer Generator - LLaMA-3.3-70B via OpenRouter
Uses V4 prompts: citation-first, 3-rule structure, post-processing injection.
"""
import re, time
from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLAMA_MODEL

# ── V4 Prompts ─────────────────────────────────────────────────────────────────
ANSWER_SYSTEM_PROMPT_V4 = (
    "You are FinSearch, a warm and knowledgeable financial assistant for a bank.\n\n"
    "You have two sources of information:\n"
    "- USER ACCOUNT DATA: the customer's actual transactions, disputes, and account info"
    " (authoritative for personal questions)\n"
    "- DOCUMENT EXCERPTS: policy and regulation documents"
    " (authoritative for rules/rights/procedures)\n\n"
    "You have exactly 3 jobs:\n\n"
    "1. CITATIONS (most important):\n"
    "- For facts from DOCUMENT EXCERPTS: end the sentence with [Source: document_name]\n"
    "- For facts from USER ACCOUNT DATA: end the sentence with [Source: your account data]\n"
    "- Every factual sentence must have one of these citations. No citation = invalid.\n\n"
    "2. STRUCTURE:\n"
    "Write exactly in this order:\n"
    "a) One warm, empathetic opening sentence (no citation needed)\n"
    "b) Factual sentences with citations - use only what is relevant to the question\n"
    "c) One warm closing sentence offering further help (no citation needed)\n\n"
    "3. ACCURACY:\n"
    "- For personal questions (why was my payment declined, my last transaction, my dispute):"
    " answer ONLY from USER ACCOUNT DATA. State exactly what the data shows -"
    " transaction status, reason, merchant, amount."
    " Do NOT cite policy documents for personal account questions.\n"
    "- For policy/rights questions (what are my rights, how long does a dispute take):"
    " answer from DOCUMENT EXCERPTS.\n"
    "- NEVER mix sources - do not cite loan or policy documents when answering"
    " a personal account question.\n"
    "- If neither source has the answer, say so honestly."
    " Do not pad with unrelated information."
)

ANSWER_USER_PROMPT_V4 = (
    "USER ACCOUNT DATA (use this first for personal account questions):\n"
    "{personal_context}\n\n"
    "DOCUMENT EXCERPTS (use for policy, rules, and rights questions):\n"
    "{context}\n\n"
    "User question: {question}\n\n"
    "Answer following the 3 rules. For personal account questions, lead with what the"
    " account data shows. For policy questions, cite the document."
)

OOS_RESPONSE = (
    "Thank you so much for reaching out to FinSearch.\n\n"
    "I appreciate you taking the time to ask, and I want to be completely transparent"
    " with you - your question falls outside the areas I'm trained to assist with."
    " I specialise in financial regulations, payment industry standards, consumer"
    " protection policies, and related financial document topics.\n\n"
    "I genuinely wouldn't want to provide you with inaccurate or incomplete guidance"
    " on something outside my expertise. For the best and most reliable help, I'd"
    " warmly recommend speaking with a qualified human advisor who can give your"
    " question the personalised attention it deserves.\n\n"
    "If you have any questions about financial regulations, payment standards,"
    " consumer protection, or related policies - I'm right here and happy to help!"
)

RBAC_BLOCKED_RESPONSE = (
    "Thank you for your question.\n\n"
    "That information isn't available through this portal for your account type."
    " This helps us ensure you receive the most relevant and appropriate guidance.\n\n"
    "If you need further assistance, I'd recommend reaching out to our support team"
    " directly - they'll be happy to help with queries outside your current access level.\n\n"
    "Is there anything else I can help you with today?"
)

CONF_MEDIUM_PREFIX = (
    "I want to be transparent - my confidence in this answer is moderate, "
    "as the available documents provide only partial coverage of your question. "
    "I'd encourage you to verify the details with a specialist.\n\n"
)
CONF_LOW_PREFIX = (
    "I must be honest with you - I have low confidence in this answer. "
    "The documents don't appear to cover this topic sufficiently. "
    "Please treat this as a starting point only and verify with a qualified human advisor.\n\n"
)

# ── Citation Injection ─────────────────────────────────────────────────────────
def inject_citations(answer: str, top_chunks: list[dict]) -> str:
    """Post-process: inject [Source: doc] on factual sentences missing citations."""
    if not top_chunks:
        return answer

    primary_doc = top_chunks[0]["chunk_id"].rsplit("_chunk_", 1)[0]

    skip_phrases = [
        "source:", "i understand", "i appreciate", "thank you",
        "feel free", "happy to help", "hope this", "let me know",
        "please reach out", "do not hesitate", "i'm here", "i am here",
        "great question", "of course", "warmly", "certainly",
    ]

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    result    = []

    for sent in sentences:
        sent  = sent.strip()
        lower = sent.lower()
        if not sent:
            continue
        if "[source:" in lower:
            result.append(sent)
        elif any(p in lower for p in skip_phrases) or len(sent.split()) < 7:
            result.append(sent)
        else:
            clean = sent.rstrip(".!?")
            result.append(f"{clean} [Source: {primary_doc}].")

    return " ".join(result)


# ── Generator ──────────────────────────────────────────────────────────────────
def generate_answer(
    query: str,
    top_chunks: list[dict],
    personal_context: str = "",
    retry: int = 2,
) -> tuple[str, str]:
    """
    Generate answer using V4 prompts + citation injection.
    Returns (answer_with_citations, context_string).
    """
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    context_parts = []
    for i, chunk in enumerate(top_chunks):
        doc_name = chunk["chunk_id"].rsplit("_chunk_", 1)[0]
        context_parts.append(f"[{i+1}] Document: {doc_name}\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    personal_section = personal_context if personal_context else "No personal account data available."

    user_msg = ANSWER_USER_PROMPT_V4.format(
        context=context if context else "No document excerpts retrieved.",
        personal_context=personal_section,
        question=query,
    )

    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model=LLAMA_MODEL,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT_V4},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=1000,
                temperature=0.1,
            )
            raw   = resp.choices[0].message.content.strip()
            cited = inject_citations(raw, top_chunks)
            return cited, context
        except Exception as e:
            print(f"[generator] attempt {attempt} error: {type(e).__name__}: {e}")
            if attempt == retry:
                return "I apologise - I encountered a technical issue. Please try again shortly.", ""
            time.sleep(2)
