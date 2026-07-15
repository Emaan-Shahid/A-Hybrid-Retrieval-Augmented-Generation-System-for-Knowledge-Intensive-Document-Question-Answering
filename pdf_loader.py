import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader


def load_documents(file_path: str) -> list:
    """
    Load documents from a file path.
    Supports PDF, DOCX, and TXT formats.

    ✅ FIX: Caller is responsible for deleting the temp file after this returns,
    using:  os.unlink(temp_path)
    """
    documents = []

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        documents = loader.load()

    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

    else:
        raise ValueError(
            f"Unsupported file format: {file_path}. "
            "Supported formats: .pdf, .docx, .txt"
        )

    return documents