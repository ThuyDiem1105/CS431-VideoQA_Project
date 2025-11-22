from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import streamlit as st
import os

VECTOR_STORE_DIR = "vector_store_db"


def create_and_save_vector_store(documents):
    """
    documents: List[Document]
        Document.page_content = text
        Document.metadata = {"start_time": ..., "end_time": ...}
    """
    if not documents:
        st.error("Không có Document nào để tạo vector store.")
        return False
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        st.info("Đang tạo vector store...")
        vector_store = FAISS.from_documents(documents, embedding=embeddings)

        if not os.path.exists(VECTOR_STORE_DIR):
            os.makedirs(VECTOR_STORE_DIR)

        vector_store.save_local(VECTOR_STORE_DIR)
        return True

    except Exception as e:
        st.error(f"Lỗi khi tạo hoặc lưu vector store: {e}")
        return False


def load_vector_store():
    """Load lại vector store đã lưu"""
    if not os.path.exists(VECTOR_STORE_DIR):
        st.error("Chưa tìm thấy vector store.")
        return None

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(
            VECTOR_STORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vector_store

    except Exception as e:
        st.error(f"Lỗi khi load vector store: {e}")
        return None
