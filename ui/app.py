"""
app.py — Streamlit frontend for legal & Financial Document Intelligence.

Features:
  • Sidebar: PDF upload, processed-files list
  • Tab 1: Chat interface with CRAG pipeline, citations, and raw chunks
  • Tab 2: Structured clause extraction with risk-level badges
"""



import streamlit as st
import httpx
import json
import time

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚖️ Legal & Financial Document Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"
TIMEOUT = 120.0  # seconds — LLM calls can be slow

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Chat bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .assistant-bubble {
        background: rgba(255, 255, 255, 0.08);
        color: #e0e0e0;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 0;
        max-width: 85%;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.95rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-success { background: #00c853; color: #000; }
    .badge-warning { background: #ffd600; color: #000; }
    .badge-danger  { background: #ff1744; color: #fff; }
    .badge-info    { background: #448aff; color: #fff; }

    /* Risk badges */
    .risk-low    { background: rgba(0, 200, 83, 0.2); color: #00c853; border: 1px solid #00c853; }
    .risk-medium { background: rgba(255, 214, 0, 0.2); color: #ffd600; border: 1px solid #ffd600; }
    .risk-high   { background: rgba(255, 23, 68, 0.2); color: #ff1744; border: 1px solid #ff1744; }

    /* Heading styles */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Citation box */
    .citation-box {
        background: rgba(68, 138, 255, 0.1);
        border-left: 3px solid #448aff;
        padding: 8px 14px;
        border-radius: 0 8px 8px 0;
        margin: 4px 0;
        font-size: 0.85rem;
        color: #b0bec5;
    }

    /* Metrics row */
    .metric-card {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session State Initialization ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "last_raw_chunks" not in st.session_state:
    st.session_state.last_raw_chunks = []


# ── Helper Functions ─────────────────────────────────────────────────────────

def risk_badge(level: str) -> str:
    """Return an HTML badge span for a risk level."""
    emoji_map = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    css_class = f"risk-{level.lower()}"
    emoji = emoji_map.get(level.lower(), "⚪")
    return f'<span class="badge {css_class}">{emoji} {level.upper()}</span>'


def api_post(endpoint: str, **kwargs) -> httpx.Response:
    """Make a POST request to the FastAPI backend."""
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            return client.post(f"{API_BASE}{endpoint}", **kwargs)
    except httpx.ConnectError:
        st.error(
            "⚠️ Cannot connect to the API server. "
            "Make sure the backend is running: `uvicorn api.main:app --reload`"
        )
        st.stop()
    except httpx.TimeoutException:
        st.error("⏱️ Request timed out. The server may be overloaded.")
        st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown("## ⚖️ DocIntel")
    st.markdown(
        '<p style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">'
        "Legal & Financial Document Intelligence<br>"
        "Powered by Corrective RAG</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── File Uploader ────────────────────────────────────────────────────
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a legal or financial PDF document for analysis.",
    )

    if uploaded_file is not None:
        if st.button("🚀 Ingest Document", use_container_width=True):
            with st.spinner("Parsing, chunking, and embedding…"):
                response = api_post(
                    "/upload_document",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                )

            if response.status_code == 200:
                data = response.json()
                st.success(
                    f"✅ **{data['filename']}** ingested — "
                    f"**{data['chunks_stored']}** chunks stored"
                )
                if data["filename"] not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(data["filename"])
            else:
                detail = response.json().get("detail", "Unknown error")
                st.error(f"❌ Upload failed: {detail}")

    st.markdown("---")

    # ── Uploaded Documents List ──────────────────────────────────────────
    st.markdown("### 📚 Processed Documents")
    if st.session_state.uploaded_files:
        for fn in st.session_state.uploaded_files:
            st.markdown(f"📎 `{fn}`")
    else:
        st.markdown(
            '<p style="color: rgba(255,255,255,0.35); font-size: 0.85rem;">'
            "No documents uploaded yet.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<p style="color: rgba(255,255,255,0.25); font-size: 0.75rem; text-align: center;">'
        "Built with LlamaParse · LangGraph · Groq · ChromaDB</p>",
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN AREA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<div class="main-title">⚖️ Document Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Ask questions, extract clauses, and analyze legal &amp; financial documents with AI-powered Corrective RAG.</div>',
    unsafe_allow_html=True,
)

tab_chat, tab_clauses = st.tabs(["💬 Document Chat", "📋 Extract Clauses"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: DOCUMENT CHAT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_chat:
    # ── Chat History Display ─────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="assistant-bubble">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                # Show citations if available
                if msg.get("citations"):
                    with st.expander("📎 Sources & Citations", expanded=False):
                        for citation in msg["citations"]:
                            st.markdown(
                                f'<div class="citation-box">{citation}</div>',
                                unsafe_allow_html=True,
                            )
                # Show rewrite badge
                if msg.get("loop_count", 0) > 0:
                    st.markdown(
                        f'<span class="badge badge-info">🔄 Query rewrites used: '
                        f'{msg["loop_count"]}</span>',
                        unsafe_allow_html=True,
                    )

    # ── Chat Input ───────────────────────────────────────────────────────
    st.markdown("---")
    user_query = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., What are the payment terms in the contract?",
        key="chat_input",
    )

    col_send, col_clear = st.columns([1, 5])
    with col_send:
        send_clicked = st.button("🔍 Ask", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_raw_chunks = []
            st.rerun()

    if send_clicked and user_query.strip():
        # Add user message
        st.session_state.chat_history.append(
            {"role": "user", "content": user_query.strip()}
        )

        with st.spinner("🔍 Running Corrective RAG pipeline…"):
            start_time = time.time()
            response = api_post(
                "/chat_query",
                json={"query": user_query.strip(), "top_k": 5},
            )
            elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()

            # Add assistant message
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": data["answer"],
                    "citations": data.get("citations", []),
                    "loop_count": data.get("loop_count", 0),
                    "chunks_used": data.get("chunks_used", 0),
                }
            )

            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">'
                    f'{data.get("chunks_used", 0)}</div>'
                    f'<div class="metric-label">Chunks Used</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">'
                    f'{data.get("loop_count", 0)}</div>'
                    f'<div class="metric-label">Rewrites</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">'
                    f'{elapsed:.1f}s</div>'
                    f'<div class="metric-label">Latency</div></div>',
                    unsafe_allow_html=True,
                )

            st.rerun()
        else:
            detail = response.json().get("detail", "Unknown error")
            st.error(f"❌ Query failed: {detail}")

    # ── Transparency Panel — Raw Retrieved Chunks ────────────────────────
    st.markdown("---")
    with st.expander("🔍 Raw Retrieved Chunks (Transparency / Auditability)", expanded=False):
        if st.session_state.chat_history:
            # Find the last assistant message that has chunk data
            st.markdown(
                '<p style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">'
                "Showing the raw chunks and metadata that the system retrieved "
                "to generate the most recent answer. This enables full "
                "auditability of the RAG pipeline.</p>",
                unsafe_allow_html=True,
            )
            last_assistant = None
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "assistant":
                    last_assistant = msg
                    break
            if last_assistant and last_assistant.get("citations"):
                for i, c in enumerate(last_assistant["citations"], 1):
                    st.markdown(f"**Chunk {i}:** {c}")
            else:
                st.info("No chunk data available for the latest response.")
        else:
            st.info("Submit a query to see raw retrieved chunks here.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: CLAUSE EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_clauses:
    st.markdown(
        '<div class="glass-card">'
        "<h3>📋 Automated Clause Extraction</h3>"
        "<p style='color: rgba(255,255,255,0.6);'>"
        "Extract and classify all important legal and financial clauses from "
        "your uploaded documents. Each clause is assessed for risk level.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    clause_query = st.text_input(
        "Custom extraction instruction (optional):",
        value="Extract all important clauses",
        key="clause_query",
    )

    if st.button("⚡ Extract Key Clauses", use_container_width=True, key="extract_btn"):
        with st.spinner("🔎 Analyzing document for clauses…"):
            response = api_post(
                "/extract_clauses",
                json={"query": clause_query},
            )

        if response.status_code == 200:
            data = response.json()
            clauses = data.get("clauses", [])

            if not clauses:
                st.warning("No clauses found. Make sure a document has been uploaded and ingested.")
            else:
                # Summary metrics
                st.markdown(
                    f'<div class="glass-card">'
                    f"<strong>Source:</strong> {data.get('filename', 'N/A')} &nbsp;|&nbsp; "
                    f"<strong>Clauses Found:</strong> {data.get('total_found', 0)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Risk summary
                risk_counts = {"high": 0, "medium": 0, "low": 0}
                for c in clauses:
                    rl = c.get("risk_level", "medium").lower()
                    risk_counts[rl] = risk_counts.get(rl, 0) + 1

                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value" style="color: #ff1744;">'
                        f'{risk_counts["high"]}</div>'
                        f'<div class="metric-label">🔴 High Risk</div></div>',
                        unsafe_allow_html=True,
                    )
                with rc2:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value" style="color: #ffd600;">'
                        f'{risk_counts["medium"]}</div>'
                        f'<div class="metric-label">🟡 Medium Risk</div></div>',
                        unsafe_allow_html=True,
                    )
                with rc3:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value" style="color: #00c853;">'
                        f'{risk_counts["low"]}</div>'
                        f'<div class="metric-label">🟢 Low Risk</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                # Clause table
                for i, clause in enumerate(clauses, 1):
                    rl = clause.get("risk_level", "medium").lower()
                    badge = risk_badge(rl)

                    st.markdown(
                        f'<div class="glass-card">'
                        f"<strong>{i}. {clause.get('clause_type', 'Unknown')}</strong> "
                        f"&nbsp;{badge} &nbsp; "
                        f'<span style="color: rgba(255,255,255,0.4); font-size: 0.8rem;">'
                        f"Page {clause.get('page_number', '?')}</span><br><br>"
                        f'<p style="color: rgba(255,255,255,0.75); font-size: 0.9rem; '
                        f'line-height: 1.6;">{clause.get("clause_text", "")}</p>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            detail = response.json().get("detail", "Unknown error")
            st.error(f"❌ Clause extraction failed: {detail}")
