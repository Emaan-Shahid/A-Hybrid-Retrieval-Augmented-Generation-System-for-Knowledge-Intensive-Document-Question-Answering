"""
chat_store.py
─────────────────────────────────────────────
Persistent storage for FlowMind AI.

Adds "New Chat" / "History" behaviour (like ChatGPT/Claude) to the
Streamlit RAG app:

- Every chat gets a session_id (uuid4).
- Session metadata + full Q&A history are stored in a local SQLite
  file (flowmind.db) so they survive app restarts / browser refresh.
- The FAISS vectorstore built from the uploaded documents is saved to
  disk per-session (vector_stores/<session_id>/) so re-opening an old
  chat does NOT require re-uploading or re-embedding the documents.

No external services needed — everything lives next to the app.
"""

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "flowmind.db")
VSTORE_DIR = os.path.join(os.path.dirname(__file__), "vector_stores")
SOURCE_DIR = os.path.join(os.path.dirname(__file__), "source_files")

os.makedirs(VSTORE_DIR, exist_ok=True)
os.makedirs(SOURCE_DIR, exist_ok=True)


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id            TEXT PRIMARY KEY,
                title         TEXT NOT NULL,
                file_names    TEXT NOT NULL DEFAULT '[]',
                chunks        INTEGER NOT NULL DEFAULT 0,
                files         INTEGER NOT NULL DEFAULT 0,
                queries       INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)


def new_session_id() -> str:
    return str(uuid.uuid4())


def create_session(session_id: str, title: str = "New Chat"):
    now = datetime.now().isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (id, title, file_names, chunks, files, "
            "queries, created_at, updated_at) VALUES (?, ?, '[]', 0, 0, 0, ?, ?)",
            (session_id, title, now, now),
        )


def session_exists(session_id: str) -> bool:
    with _conn() as c:
        row = c.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row is not None


def update_session_files(session_id: str, file_names: list, chunks: int, files: int, title: str = None):
    now = datetime.now().isoformat()
    with _conn() as c:
        if title:
            c.execute(
                "UPDATE sessions SET file_names=?, chunks=?, files=?, updated_at=?, title=? WHERE id=?",
                (json.dumps(file_names), chunks, files, now, title, session_id),
            )
        else:
            c.execute(
                "UPDATE sessions SET file_names=?, chunks=?, files=?, updated_at=? WHERE id=?",
                (json.dumps(file_names), chunks, files, now, session_id),
            )


def add_message(session_id: str, question: str, answer: str):
    now = datetime.now().isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (session_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
            (session_id, question, answer, now),
        )
        c.execute(
            "UPDATE sessions SET queries = queries + 1, updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        # Auto-title the chat from the first question if still "New Chat"
        row = c.execute("SELECT title FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row and row["title"] == "New Chat":
            short = question.strip()
            if len(short) > 48:
                short = short[:45].rstrip() + "…"
            c.execute("UPDATE sessions SET title=? WHERE id=?", (short, session_id))


def clear_messages(session_id: str):
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        c.execute("UPDATE sessions SET queries = 0 WHERE id = ?", (session_id,))


def get_messages(session_id: str) -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT question, answer FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [{"question": r["question"], "answer": r["answer"]} for r in rows]


def get_session(session_id: str):
    with _conn() as c:
        row = c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None


def list_sessions() -> list:
    """Most recently updated first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_session(session_id: str):
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    # Remove its saved vector index and original source files too
    import shutil
    for base in (VSTORE_DIR, SOURCE_DIR):
        path = os.path.join(base, session_id)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


# ─────────────────────────────────────────────
# Original file storage (so we can show images/pages later —
# the RAG pipeline only keeps extracted TEXT, so without this,
# any pictures/diagrams in the original PDF are gone for good)
# ─────────────────────────────────────────────

def save_source_file(session_id: str, filename: str, data: bytes) -> str:
    folder = os.path.join(SOURCE_DIR, session_id)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "wb") as f:
        f.write(data)
    return path


def get_source_file_path(session_id: str, filename: str):
    path = os.path.join(SOURCE_DIR, session_id, filename)
    return path if os.path.isfile(path) else None


def render_pdf_page(pdf_path: str, page_number: int, dpi: int = 150) -> bytes:
    """Render a single PDF page to PNG bytes, so any diagrams/screenshots
    on that page are visible even though the RAG pipeline only stored text."""
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


# ─────────────────────────────────────────────
# Vectorstore (FAISS) persistence per session
# ─────────────────────────────────────────────

def vectorstore_path(session_id: str) -> str:
    return os.path.join(VSTORE_DIR, session_id)


def save_vectorstore(session_id: str, db):
    path = vectorstore_path(session_id)
    os.makedirs(path, exist_ok=True)
    db.save_local(path)


def load_vectorstore(session_id: str, embeddings):
    from langchain_community.vectorstores import FAISS
    path = vectorstore_path(session_id)
    if not os.path.isdir(path):
        return None
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)