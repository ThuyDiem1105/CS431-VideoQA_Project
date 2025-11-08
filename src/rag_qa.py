from typing import Dict, Any, List

import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document  # cài từ langchain-core
from src.vector_store_builder import load_vector_store

load_dotenv()  # đọc .env


def build_rag_pipeline() -> Dict[str, Any] | None:
    """
    Tạo pipeline RAG đơn giản:
    - Load vector store (FAISS) từ phần thành viên 1
    - Tạo LLM (ChatOpenAI)
    Trả về dict: {"vector_store": ..., "llm": ...}
    """
    vector_store = load_vector_store()
    if vector_store is None:
        st.error("Không load được vector store. Hãy chạy app.py để tạo vector store trước.")
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Thiếu OPENAI_API_KEY trong file .env")
        return None

    base_url = os.getenv("OPENAI_BASE_URL", None)

    llm = ChatOpenAI(
        model="gpt-4o-mini",   # gà muốn so sánh model thì sửa tên ở đây
        temperature=0.2,
        api_key=api_key,
        base_url=base_url,
    )

    return {"vector_store": vector_store, "llm": llm}


def _build_prompt(question: str, docs: List[Document]) -> str:
    """
    Ghép context từ các đoạn transcript + câu hỏi thành prompt cho LLM.
    """
    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        context_blocks.append(f"[Đoạn {i}]\n{doc.page_content}")

    context_str = "\n\n".join(context_blocks)

    prompt = f"""
Bạn là trợ giảng môn CS431, nhiệm vụ là trả lời câu hỏi của sinh viên
dựa trên transcript bài giảng dưới đây. Chỉ dùng thông tin trong transcript.
Nếu không đủ thông tin, hãy nói rõ là "không tìm thấy trong video".

TRANSCRIPT:
{context_str}

CÂU HỎI:
{question}

HÃY TRẢ LỜI NGẮN GỌN, RÕ RÀNG, BẰNG TIẾNG VIỆT:
"""
    return prompt.strip()


def ask_question(pipeline: Dict[str, Any], question: str) -> Dict[str, Any]:
    """
    Thực hiện:
    1. similarity_search trên vector store để lấy các đoạn liên quan
    2. Build prompt từ context + câu hỏi
    3. Gọi LLM để sinh câu trả lời

    Trả về:
    - answer: str
    - sources: list[Document]
    """
    vector_store = pipeline["vector_store"]
    llm: ChatOpenAI = pipeline["llm"]

    # Lấy top-k đoạn liên quan (vector_store tự embed query giùm mình)
    docs: List[Document] = vector_store.similarity_search(question, k=5)

    prompt = _build_prompt(question, docs)
    response = llm.invoke(prompt)   # .invoke trả về ChatMessage
    answer = response.content

    return {
        "answer": answer,
        "sources": docs,
    }
