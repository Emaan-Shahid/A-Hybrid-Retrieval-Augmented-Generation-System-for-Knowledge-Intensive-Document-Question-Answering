import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ✅ FIX 1: Cache the embeddings model so it loads ONCE per session
# instead of re-loading it every time a new file is uploaded.
@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    docs = splitter.split_documents(documents)

    # ✅ FIX 2: Reuse the cached embeddings instance
    embeddings = get_embeddings_model()

    db = FAISS.from_documents(docs, embeddings)

    return db, len(docs)