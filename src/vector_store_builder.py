from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import os

VECTOR_STORE_DIR = "vector_store_db"

def create_and_save_vector_store(chunks):
    if not chunks:
        st.error("Không có đoạn văn bản để tạo vector store.")
        return False
    try:
        # Khởi tạo embeddings 
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Tạo vector store từ các đoạn văn bản
        st.info("Đang tạo vector store...")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        # Lưu vector store vào thư mục
        if not os.path.exists(VECTOR_STORE_DIR):
            os.makedirs(VECTOR_STORE_DIR)
        vector_store.save_local(VECTOR_STORE_DIR)
        return True
    except Exception as e:
        st.error(f"Lỗi khi tạo hoặc lưu vector store: {e}")
        return False
    
def load_vector_store():
    """
    Load lại vector store đã lưu trong VECTOR_STORE_DIR.
    Dùng hàm này để truy vấn.
    """
    if not os.path.exists(VECTOR_STORE_DIR):
        st.error("Chưa tìm thấy vector store. Hãy chạy app.py để tạo vector store trước.")
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