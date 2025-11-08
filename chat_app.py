import streamlit as st
from src.rag_qa import build_rag_pipeline, ask_question

st.set_page_config(
    layout="wide",
    page_title="Video QA Chat",
    page_icon="🎓",
)

st.title("🎓 Hệ thống Hỏi đáp Video CS431 (RAG)")

st.markdown(
    """
Hệ thống này sử dụng **vector store** được tạo từ video bài giảng.

Bây giờ bạn có thể:
- Gõ câu hỏi về nội dung video
- Hệ thống sẽ truy xuất transcript liên quan và dùng LLM để sinh câu trả lời.
"""
)

@st.cache_resource
def init_pipeline():
    return build_rag_pipeline()

pipeline = init_pipeline()
if pipeline is None:
    st.stop()

# Lưu lịch sử hội thoại
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị hội thoại trước đó
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ô nhập câu hỏi
user_input = st.chat_input("Nhập câu hỏi của bạn về video...")

if user_input:
    # User message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và sinh câu trả lời..."):
            result = ask_question(pipeline, user_input)
            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

            # Hiển thị context đã dùng
            with st.expander("Xem các đoạn transcript được sử dụng"):
                if not sources:
                    st.write("Không có đoạn transcript nào được trả về.")
                for i, doc in enumerate(sources, start=1):
                    st.markdown(f"**Đoạn {i}:**")
                    st.markdown(doc.page_content)
                    st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": answer})
