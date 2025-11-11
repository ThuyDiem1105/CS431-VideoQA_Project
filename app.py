# app.py

import os
import streamlit as st

from src.video_processor import save_uploaded_file, extract_audio
from src.text_processor import transcribe_audio, chunk_text
from src.vector_store_builder import create_and_save_vector_store
from src.rag_qa import build_rag_pipeline, ask_question

st.set_page_config(
    layout="wide",
    page_title="Hệ thống Hỏi đáp Video CS431",
    page_icon="🎓",
)

st.title("🎓 Hệ thống Hỏi đáp Video CS431")

st.markdown(
    """
Ứng dụng này gồm **2 bước trong 1**:

1. 📥 Tải lên video bài giảng và **bấm nút xử lý** → hệ thống tách âm thanh, phiên âm, chia nhỏ văn bản, tạo **vector store**.
2. 💬 Sau khi tạo xong, bạn có thể **chat hỏi đáp** ngay phía dưới, dựa trên nội dung video đó.
"""
)

# ========== KHỞI TẠO STATE ==========

if "video_name" not in st.session_state:
    st.session_state.video_name = None

if "processed" not in st.session_state:
    st.session_state.processed = False  # video đã xử lý xong chưa?

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None    # {"vector_store": ..., "llm": ...}

if "messages" not in st.session_state:
    st.session_state.messages = []      # lịch sử chat


def reset_state_for_new_video(filename: str):
    """Reset toàn bộ state khi đổi video khác."""
    st.session_state.video_name = filename
    st.session_state.processed = False
    st.session_state.pipeline = None
    st.session_state.messages = []


# ========== BƯỚC 1: UPLOAD & XỬ LÝ VIDEO ==========

st.header("1️⃣ Tải lên Video & Xây dựng Vector Store")

uploaded_file = st.file_uploader(
    "Chọn tệp video (.mp4) để xây dựng hệ thống hỏi đáp:",
    type=["mp4"],
)

if uploaded_file is not None:
    # Nếu user chọn video khác với lần trước -> reset lại toàn bộ
    if st.session_state.video_name != uploaded_file.name:
        reset_state_for_new_video(uploaded_file.name)

    st.markdown(f"**Video hiện tại:** `{uploaded_file.name}`")

    col_btn, col_status = st.columns([1, 3])

    with col_btn:
        process_btn = st.button(
            "🚀 Bắt đầu xử lý video",
            type="primary",
            disabled=st.session_state.processed,
        )

    with col_status:
        if st.session_state.processed:
            st.success("Video đã được xử lý. Bạn có thể kéo xuống dưới để bắt đầu hỏi đáp 👇")

    # CHỈ xử lý video khi user bấm nút
    if process_btn:
        with st.spinner("Đang xử lý video..."):
            # 1. Lưu video tạm
            st.subheader("📂 Đang lưu tệp video...")
            video_path = save_uploaded_file(uploaded_file)
            st.success("Tệp video đã được lưu.")

            # 2. Tách âm thanh
            st.subheader("🎧 Đang tách âm thanh...")
            audio_path = extract_audio(video_path)
            if audio_path:
                st.success("Âm thanh đã được tách thành công.")

                # 3. Phiên âm
                st.subheader("📝 Đang phiên âm âm thanh...")
                transcript = transcribe_audio(audio_path)
                if transcript:
                    st.success("Phiên âm hoàn tất.")
                    st.text_area("Văn bản phiên âm:", transcript, height=200)

                    # 4. Chia nhỏ văn bản
                    st.subheader("✂️ Đang chia nhỏ văn bản...")
                    chunks = chunk_text(transcript)
                    if chunks:
                        st.success(f"Chia nhỏ thành công thành {len(chunks)} đoạn.")

                        # 5. Tạo và lưu vector store
                        st.subheader("📦 Đang tạo và lưu vector store...")
                        if create_and_save_vector_store(chunks):
                            st.success("Vector store đã được tạo và lưu thành công.")

                            # 6. Khởi tạo RAG pipeline (vector store + LLM Groq)
                            st.subheader("🧠 Đang khởi tạo RAG pipeline...")
                            pipeline = build_rag_pipeline()
                            if pipeline is not None:
                                st.session_state.pipeline = pipeline
                                st.session_state.processed = True
                                st.success("RAG pipeline đã sẵn sàng. Kéo xuống dưới để bắt đầu hỏi đáp 👇")
                            else:
                                st.error("Không khởi tạo được RAG pipeline.")
                        else:
                            st.error("Không thể tạo hoặc lưu vector store.")
                    else:
                        st.error("Không thể chia nhỏ văn bản.")
                else:
                    st.error("Không thể phiên âm âm thanh.")
            else:
                st.error("Không thể tách âm thanh từ video.")

            # Xoá file tạm
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                st.warning(f"Không thể xoá tệp tạm thời: {e}")

else:
    st.info("Hãy tải lên một video (.mp4) để bắt đầu.")

st.markdown("---")

# ========== BƯỚC 2: CHAT HỎI ĐÁP ==========

st.header("2️⃣ Hỏi đáp dựa trên nội dung video")

if not st.session_state.processed or st.session_state.pipeline is None:
    st.info("Chưa có pipeline sẵn sàng. Hãy chắc chắn bạn đã bấm **“Bắt đầu xử lý video”** và quá trình đã hoàn tất.")
else:
    st.markdown(
        "✅ Vector store & mô hình LLM đã sẵn sàng. "
        "Bây giờ bạn có thể đặt câu hỏi về **nội dung video**."
    )

    # Hiển thị lịch sử hội thoại
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Ô nhập câu hỏi
    user_input = st.chat_input("Nhập câu hỏi của bạn về bài giảng...")

    if user_input:
        # Lưu & hiển thị tin nhắn user
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đang truy xuất và sinh câu trả lời..."):
                result = ask_question(st.session_state.pipeline, user_input)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                # Hiển thị các đoạn transcript liên quan
                with st.expander("Xem các đoạn transcript đã sử dụng"):
                    if not sources:
                        st.write("Không có đoạn transcript nào được trả về.")
                    for i, doc in enumerate(sources, start=1):
                        st.markdown(f"**Đoạn {i}:**")
                        st.markdown(doc.page_content)
                        st.markdown("---")

        # Lưu lịch sử trả lời
        st.session_state.messages.append({"role": "assistant", "content": answer})