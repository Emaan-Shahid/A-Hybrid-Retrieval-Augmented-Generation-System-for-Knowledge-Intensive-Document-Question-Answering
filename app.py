import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from pdf_loader import load_documents
from rag_pipeline import create_vectorstore, get_embeddings_model
from tools.router import select_tool
import chat_store

load_dotenv()
chat_store.init_db()

# Explicitly set env var so langchain-google-genai always finds it
_key = os.getenv("GOOGLE_API_KEY", "")
if _key:
    os.environ["GOOGLE_API_KEY"] = _key

# PAGE CONFIG

st.set_page_config(
    page_title="FlowMind AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# ✅ FIX 3: Cache the LLM so it is NOT rebuilt on every question
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        st.error("GOOGLE_API_KEY missing — check your .env file.")
        st.stop()
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=api_key,
    )

# ─────────────────────────────────────────────
# CSS  – Aurora / glassmorphism light theme
# ─────────────────────────────────────────────

st.markdown("""
<style>
/* ── System font stack — NO network request, loads instantly ── */
/* ── Reset & root ───────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

#MainMenu, footer, header, .stDeployButton, .stStatusWidget { visibility: hidden; }

html, body, .stApp {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui,
                 'Helvetica Neue', Arial, sans-serif;
    background: #F0F4FF;
    color: #1A1D2E;
}

/* Animated aurora background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(167,139,250,0.25) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 90%, rgba(99,179,237,0.20) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 60% 40%, rgba(236,72,153,0.10) 0%, transparent 60%),
        linear-gradient(160deg, #EEF2FF 0%, #F0F4FF 50%, #EFF6FF 100%);
    animation: auroraShift 18s ease-in-out infinite alternate;
    pointer-events: none;
}

@keyframes auroraShift {
    0%   { filter: hue-rotate(0deg) brightness(1); }
    50%  { filter: hue-rotate(12deg) brightness(1.04); }
    100% { filter: hue-rotate(-8deg) brightness(0.98); }
}

/* ── Layout ─────────────────────────────────── */
.main .block-container {
    position: relative;
    z-index: 1;
    max-width: 820px;
    padding: 2rem 1.5rem 3rem;
}

/* ── Hero ───────────────────────────────────── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 100px;
    padding: 5px 16px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7C3AED;
    margin-bottom: 18px;
    box-shadow: 0 2px 12px rgba(124,58,237,0.08);
}

.hero-title {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    font-size: clamp(42px, 8vw, 64px);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #5B21B6 0%, #7C3AED 40%, #2563EB 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 12px;
}

.hero-title span {
    background: linear-gradient(135deg, #EC4899 0%, #8B5CF6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 15px;
    color: #6B7280;
    font-weight: 400;
    line-height: 1.6;
    max-width: 420px;
    margin: 0 auto;
}

/* ── Glass card ─────────────────────────────── */
.glass {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.85);
    border-radius: 20px;
    padding: 22px 24px;
    margin-bottom: 16px;
    box-shadow:
        0 1px 1px rgba(0,0,0,0.02),
        0 4px 8px rgba(124,58,237,0.04),
        0 12px 32px rgba(37,99,235,0.04),
        inset 0 1px 0 rgba(255,255,255,0.9);
    transition: transform 0.25s cubic-bezier(.22,.68,0,1.2),
                box-shadow 0.25s ease;
}

.glass:hover {
    transform: translateY(-3px);
    box-shadow:
        0 1px 1px rgba(0,0,0,0.02),
        0 8px 20px rgba(124,58,237,0.07),
        0 24px 48px rgba(37,99,235,0.06),
        inset 0 1px 0 rgba(255,255,255,0.9);
}

.card-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.card-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #E5E7EB, transparent);
}

/* ── Metrics ────────────────────────────────── */
.metric-row { display: flex; gap: 12px; margin-bottom: 16px; }

.metric {
    flex: 1;
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 16px;
    padding: 16px 12px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease;
}

.metric:hover { transform: translateY(-2px); }

.metric-num {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #7C3AED, #2563EB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 4px;
}

.metric-lbl {
    font-size: 11px;
    color: #9CA3AF;
    font-weight: 500;
}

/* ── File chips ─────────────────────────────── */
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; }

.chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 100px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 500;
    color: #374151;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: all 0.15s ease;
}

.chip:hover {
    border-color: #8B5CF6;
    color: #7C3AED;
    transform: translateY(-1px);
}

/* ── Answer bubble ──────────────────────────── */
.answer-bubble {
    background: white;
    border-radius: 0 18px 18px 18px;
    padding: 18px 22px;
    box-shadow:
        0 2px 4px rgba(0,0,0,0.02),
        0 8px 24px rgba(124,58,237,0.06);
    border-left: 3px solid transparent;
    border-image: linear-gradient(180deg, #7C3AED, #2563EB) 1;
    line-height: 1.75;
    font-size: 14.5px;
    color: #1A1D2E;
    margin: 4px 0;
}

/* ── Source card ────────────────────────────── */
.source-box {
    background: #F8F9FF;
    border: 1px solid #EEF2FF;
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 13px;
    line-height: 1.65;
    color: #374151;
}

/* ── Streamlit overrides ────────────────────── */
/* Buttons */
.stButton > button {
    border-radius: 12px !important;
    background: rgba(255,255,255,0.7) !important;
    border: 1px solid rgba(229,231,235,0.8) !important;
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: all 0.2s ease !important;
    backdrop-filter: blur(6px) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

.stButton > button:hover {
    background: white !important;
    border-color: #8B5CF6 !important;
    color: #7C3AED !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.10) !important;
}

/* Chat input */
.stChatInputContainer {
    border-radius: 16px !important;
    border: 2px solid #E5E7EB !important;
    background: rgba(255,255,255,0.8) !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.06) !important;
    transition: border-color 0.2s ease !important;
}

.stChatInputContainer:focus-within {
    border-color: #7C3AED !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.12) !important;
}

/* User chat bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: transparent !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #7C3AED, #2563EB) !important;
    border-radius: 100px !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #7C3AED !important;
    border-right-color: #7C3AED !important;
}

/* Alert */
.stAlert {
    border-radius: 14px !important;
    border: none !important;
    font-size: 14px !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.6) !important;
    border-radius: 12px !important;
    border: 1px solid #EEF2FF !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    color: #374151 !important;
    backdrop-filter: blur(6px) !important;
}

.streamlit-expanderContent {
    background: transparent !important;
    border: none !important;
    padding-top: 8px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.5) !important;
    border: 2px dashed #C4B5FD !important;
    border-radius: 16px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #7C3AED !important;
    background: rgba(237,233,254,0.25) !important;
}

/* ── Quick action buttons ────────────────────── */
.qa-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }

.qa-btn {
    background: rgba(255,255,255,0.65);
    border: 1px solid rgba(229,231,235,0.8);
    border-radius: 14px;
    padding: 14px 16px;
    cursor: pointer;
    transition: all 0.2s cubic-bezier(.22,.68,0,1.2);
    text-align: left;
    backdrop-filter: blur(8px);
}

.qa-btn:hover {
    background: white;
    border-color: #A78BFA;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(124,58,237,0.10);
}

.qa-icon { font-size: 22px; margin-bottom: 6px; }
.qa-title { font-size: 13px; font-weight: 600; color: #1A1D2E; }
.qa-desc { font-size: 11px; color: #9CA3AF; margin-top: 2px; }

/* ── Empty state ─────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 32px 20px;
    color: #9CA3AF;
}
.empty-state .es-icon { font-size: 48px; margin-bottom: 10px; opacity: 0.5; }
.empty-state .es-text { font-size: 14px; line-height: 1.6; }

/* ── Mobile ──────────────────────────────────── */
@media (max-width: 640px) {
    .hero-title { font-size: 36px; letter-spacing: -1px; }
    .metric-num { font-size: 22px; }
    .qa-grid { grid-template-columns: 1fr 1fr; }
    .glass { padding: 16px; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

defaults = {
    "history": [],
    "db": None,
    "chunks": 0,
    "files": 0,
    "queries": 0,
    "file_names": [],
    "quick_question": None,
    "last_uploaded_files": [],
    # ✅ FIX 4: Cache retrieved source docs so we don't re-run MMR search
    "last_source_docs": [],
    "session_id": None,
    "hybrid_mode": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.session_id is None:
    st.session_state.session_id = chat_store.new_session_id()
    chat_store.create_session(st.session_state.session_id)

# ─────────────────────────────────────────────
# SIDEBAR — New Chat / History
# ─────────────────────────────────────────────

def _load_session(sid: str):
    """Restore a saved chat: metadata, messages, and its vector index."""
    meta = chat_store.get_session(sid)
    if not meta:
        return
    st.session_state.session_id = sid
    st.session_state.history = chat_store.get_messages(sid)
    st.session_state.file_names = json.loads(meta["file_names"])
    st.session_state.last_uploaded_files = st.session_state.file_names
    st.session_state.chunks = meta["chunks"]
    st.session_state.files = meta["files"]
    st.session_state.queries = meta["queries"]
    st.session_state.last_source_docs = []
    st.session_state.quick_question = None

    if st.session_state.file_names:
        embeddings = get_embeddings_model()
        st.session_state.db = chat_store.load_vectorstore(sid, embeddings)
    else:
        st.session_state.db = None


def _start_new_chat():
    new_sid = chat_store.new_session_id()
    chat_store.create_session(new_sid)
    st.session_state.session_id = new_sid
    st.session_state.history = []
    st.session_state.db = None
    st.session_state.chunks = 0
    st.session_state.files = 0
    st.session_state.queries = 0
    st.session_state.file_names = []
    st.session_state.last_uploaded_files = []
    st.session_state.last_source_docs = []
    st.session_state.quick_question = None


with st.sidebar:
    st.markdown("### 🧠 FlowMind AI")
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        _start_new_chat()
        st.rerun()

    st.markdown("#### 🕘 History")
    sessions = chat_store.list_sessions()
    if not sessions:
        st.caption("No previous chats yet.")
    for s in sessions:
        is_current = s["id"] == st.session_state.session_id
        label = ("🟣 " if is_current else "") + (s["title"] or "New Chat")
        col_a, col_b = st.columns([5, 1])
        with col_a:
            if st.button(label, key=f"sess_{s['id']}", use_container_width=True,
                         disabled=is_current):
                _load_session(s["id"])
                st.rerun()
        with col_b:
            if st.button("🗑️", key=f"del_{s['id']}"):
                was_current = is_current
                chat_store.delete_session(s["id"])
                if was_current:
                    _start_new_chat()
                st.rerun()
        if s["file_names"] and s["file_names"] != "[]":
            try:
                fnames = json.loads(s["file_names"])
                st.caption(", ".join(fnames[:2]) + (" +more" if len(fnames) > 2 else ""))
            except Exception:
                pass

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI-Powered Document Chat</div>
    <div class="hero-title">Flow<span>Mind</span></div>
    <p class="hero-sub">Upload your documents and ask anything — get instant, grounded answers powered by Gemini.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

st.markdown(f"""
<div class="metric-row">
    <div class="metric">
        <div class="metric-num">{st.session_state.files}</div>
        <div class="metric-lbl">📄 Documents</div>
    </div>
    <div class="metric">
        <div class="metric-num">{st.session_state.chunks}</div>
        <div class="metric-lbl">📦 Chunks</div>
    </div>
    <div class="metric">
        <div class="metric-num">{st.session_state.queries}</div>
        <div class="metric-lbl">💬 Questions</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UPLOAD CARD
# ─────────────────────────────────────────────

st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="card-label">📁 Upload Documents</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop PDF, DOCX, or TXT files here",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PROCESS FILES
# ─────────────────────────────────────────────

if uploaded_files:
    current_names = sorted([f.name for f in uploaded_files])
    last_names    = sorted(st.session_state.last_uploaded_files)

    if current_names != last_names:
        # Reset state
        for k in ("db", "chunks", "files", "file_names", "history",
                  "quick_question", "last_source_docs"):
            st.session_state[k] = [] if k in ("history", "file_names",
                                               "last_source_docs") else \
                                   None if k in ("db", "quick_question") else 0

        all_documents = []
        temp_paths    = []
        prog = st.progress(0, text="Reading files…")

        for i, uf in enumerate(uploaded_files):
            suffix = "." + uf.name.rsplit(".", 1)[-1]
            file_bytes = uf.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                temp_paths.append(tmp.name)

            # Keep the original PDF around (the RAG pipeline below only keeps
            # extracted text — without this, any images/diagrams in the PDF
            # would be lost once the temp file is deleted).
            if suffix.lower() == ".pdf":
                chat_store.save_source_file(st.session_state.session_id, uf.name, file_bytes)

            docs = load_documents(temp_paths[-1])

            # ✅ FIX: Replace ugly temp path with the real original filename
            for doc in docs:
                doc.metadata["source"] = uf.name

            all_documents.extend(docs)
            prog.progress(int((i + 1) / len(uploaded_files) * 65),
                          text=f"Loaded {uf.name}")

        prog.progress(75, text="Building vector index…")
        db, n_chunks = create_vectorstore(all_documents)
        prog.progress(100, text="Done!")
        time.sleep(0.3)
        prog.empty()

        # ✅ FIX 5: Clean up temp files after processing
        for p in temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

        st.session_state.db                 = db
        st.session_state.chunks             = n_chunks
        st.session_state.files              = len(uploaded_files)
        st.session_state.file_names         = current_names
        st.session_state.last_uploaded_files = current_names

        # ✅ Persist: save the FAISS index + session metadata so this chat
        # (documents + history) survives a refresh or app restart.
        chat_store.save_vectorstore(st.session_state.session_id, db)
        title = current_names[0] + (f" +{len(current_names)-1} more" if len(current_names) > 1 else "")
        chat_store.update_session_files(
            st.session_state.session_id,
            file_names=current_names,
            chunks=n_chunks,
            files=len(uploaded_files),
            title=title,
        )

        st.success(f"✅ {len(uploaded_files)} file(s) indexed — {n_chunks} chunks ready")
        st.rerun()

# ─────────────────────────────────────────────
# LOADED FILES CARD
# ─────────────────────────────────────────────

if st.session_state.file_names:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">📂 Loaded Files</div>', unsafe_allow_html=True)

    icons = {"pdf": "📄", "txt": "📝", "docx": "📘"}
    chips = ""
    for name in st.session_state.file_names:
        ext  = name.rsplit(".", 1)[-1].lower()
        icon = icons.get(ext, "📄")
        chips += f'<span class="chip">{icon} {name}</span>'
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

    st.write("")
    col_sp, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_source_docs = []
            chat_store.clear_messages(st.session_state.session_id)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# QUICK ACTIONS CARD
# ─────────────────────────────────────────────

if st.session_state.file_names:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">⚡ Quick Actions</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="small")
    qa = [
        (c1, "📄", "Summary",  "Summarize the key points of this document"),
        (c2, "🧠", "Concepts", "Explain the main concepts in simple terms"),
        (c3, "📚", "Notes",    "Generate concise study notes from this document"),
        (c4, "❓", "Quiz",     "Generate 5 quiz questions with answers"),
    ]
    for col, icon, label, prompt in qa:
        with col:
            if st.button(f"{icon} {label}", use_container_width=True):
                st.session_state.quick_question = prompt

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────

if st.session_state.file_names:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">💬 Conversation</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class="empty-state">
            <div class="es-icon">💬</div>
            <div class="es-text">Ask anything about your documents.<br>Try a Quick Action above to get started.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for item in st.session_state.history:
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.markdown(
                    f'<div class="answer-bubble">{item["answer"]}</div>',
                    unsafe_allow_html=True
                )

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────

if st.session_state.file_names:
    st.session_state.hybrid_mode = st.toggle(
        "🌐 Also use general knowledge (not just the uploaded documents)",
        value=st.session_state.hybrid_mode,
        help="ON: answers blend your documents with the model's own knowledge, "
             "clearly labeled. OFF: answers use ONLY your uploaded documents."
    )
    question = st.chat_input("Ask anything about your documents…")
else:
    question = st.chat_input("Upload documents first…", disabled=True)

# Quick-action overrides input
if st.session_state.quick_question:
    question = st.session_state.quick_question
    st.session_state.quick_question = None

# ─────────────────────────────────────────────
# GENERATE ANSWER
# ─────────────────────────────────────────────

if question:

    # 🧠 AI Productivity Assistant - Tool Router
    selected_tool = select_tool(question)

    if selected_tool != "rag":

        with st.spinner("Running AI tool..."):

            # First retrieve document context
            retrieved_docs = st.session_state.db.max_marginal_relevance_search(
                question, k=5, fetch_k=20
            )

            context = "\n\n".join(
                [d.page_content for d in retrieved_docs]
            )

            if selected_tool == "summary":
                from tools.summary_tool import summarize_document
                answer = summarize_document(context)

            elif selected_tool == "email":
                from tools.email_tool import generate_email
                answer = generate_email(context)

            elif selected_tool == "todo":
                from tools.todo_tool import generate_todo
                answer = generate_todo(context)

            elif selected_tool == "workflow":
                from tools.workflow import run_workflow
                answer = run_workflow(context)

        st.chat_message("assistant").markdown(answer)

    else:
        if st.session_state.db is None:
            st.warning("⚠️ Please upload documents first.")
        else:
            st.session_state.queries += 1

        with st.spinner("Searching your documents…"):
            # ✅ FIX 4: Store retrieved docs to avoid double MMR search
            retrieved_docs = st.session_state.db.max_marginal_relevance_search(
                question, k=5, fetch_k=20
            )
            st.session_state.last_source_docs = retrieved_docs

            context = "\n\n".join([d.page_content for d in retrieved_docs])

            if st.session_state.hybrid_mode:
                # ✅ Hybrid: blend document context with the model's own
                # general knowledge, but keep the two clearly labeled so
                # the answer stays trustworthy and verifiable.
                prompt = f"""You are FlowMind AI, an expert document assistant.

Answer the user's question using BOTH of these sources:
1. The provided document context (treat this as the primary, authoritative source).
2. Your own general knowledge (use this to add helpful background, definitions,
   or related information the documents don't cover).

Rules:
- Be clear, concise, and well-structured. Use bullet points where helpful.
- Clearly separate the two sources using these exact headings:
  **📄 From your documents:** — information found in the context below.
  **🌐 From general knowledge:** — relevant background not found in the context.
- If the context alone fully answers the question, you may omit the
  "General knowledge" section.
- Never present general knowledge as if it came from the uploaded documents.

Context:
{context}

Question: {question}

Answer:"""
            else:
                # ✅ Document-only: strictly grounded, no outside knowledge.
                prompt = f"""You are FlowMind AI, an expert document assistant.

Answer the user's question using ONLY the information in the provided context.
- Be clear, concise, and well-structured.
- Use bullet points or numbered lists when listing multiple items.
- If the context does not contain enough information, say so honestly.
- Do NOT make up information outside the context.

Context:
{context}

Question: {question}

Answer:"""

            llm      = get_llm()   # ✅ FIX 3: Uses cached LLM
            response = llm.invoke(prompt)
            answer   = response.content

        st.session_state.history.append({
            "question": question,
            "answer":   answer
        })
        chat_store.add_message(st.session_state.session_id, question, answer)
        st.rerun()

# ─────────────────────────────────────────────
# SOURCES (uses cached docs — no extra MMR call)
# ─────────────────────────────────────────────

if st.session_state.db and st.session_state.history and st.session_state.last_source_docs:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">📚 Source Passages</div>', unsafe_allow_html=True)

    for i, doc in enumerate(st.session_state.last_source_docs):
        # Build a clean label using real filename + page number
        meta     = doc.metadata if hasattr(doc, "metadata") else {}
        filename = meta.get("source", f"Document {i+1}")
        page_num = meta.get("page")
        page_str = f"  —  Page {int(page_num) + 1}" if page_num is not None else ""
        label    = f"📄 {filename}{page_str}"

        with st.expander(label):
            st.markdown(
                f'<div class="source-box">{doc.page_content[:900]}…</div>',
                unsafe_allow_html=True
            )

            # ✅ Show the actual PDF page as an image, so diagrams/pictures
            # on that page are visible (not just the extracted text).
            if filename.lower().endswith(".pdf") and page_num is not None:
                pdf_path = chat_store.get_source_file_path(
                    st.session_state.session_id, filename
                )
                if pdf_path:
                    try:
                        img_bytes = chat_store.render_pdf_page(pdf_path, int(page_num))
                        st.image(
                            img_bytes,
                            caption=f"Page {int(page_num) + 1} — original page",
                            use_container_width=True,
                        )
                    except Exception:
                        st.caption("⚠️ Page preview unavailable (is `pymupdf` installed?).")

    st.markdown('</div>', unsafe_allow_html=True)