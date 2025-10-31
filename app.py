import streamlit as st
import os
from src.video_processor import save_uploaded_file, extract_audio
from src.text_processor import transcribe_audio, chunk_text
from src.vector_store_builder import create_and_save_vector_store

st.set_page_config(layout="wide", page_title="Video QA Vector Store Builder")
st.title("Tạo Vector Store từ Video cho Hệ thống Hỏi đáp")

st.header("1. Tải lên Video")
uploaded_file = st.file_uploader("Chọn tệp video (định dạng .mp4)", type=["mp4"])

if uploaded_file is not None:
    with st.spinner("Đang xử lý video..."):
        # Xử lí video
        st.subheader("Đang lưu tệp video...")
        video_path = save_uploaded_file(uploaded_file)
        st.info("Tệp video đã được lưu.")

        audio_path = extract_audio(video_path)
        if audio_path:
            st.info("Âm thanh đã được tách thành công.")

            # Phiên âm âm thanh
            st.subheader("Đang phiên âm âm thanh...")
            transcript = transcribe_audio(audio_path)
            if transcript:
                st.success("Phiên âm hoàn tất.")
                st.text_area("Văn bản phiên âm:", transcript, height=250)

                # Chia nhỏ văn bản
                st.subheader("Đang chia nhỏ văn bản...")
                chunks = chunk_text(transcript)
                if chunks:
                    st.success(f"Chia nhỏ thành công thành {len(chunks)} đoạn.")

                    # Tạo và lưu vector store
                    st.subheader("Đang tạo và lưu vector store...")
                    if create_and_save_vector_store(chunks):
                        st.success("Vector store đã được tạo và lưu thành công.")
                    else:
                        st.error("Không thể tạo hoặc lưu vector store.")
                else:
                    st.error("Không thể chia nhỏ văn bản.")
            else:
                st.error("Không thể phiên âm âm thanh.")
        else:
            st.error("Không thể tách âm thanh từ video.")

        try:
            # Xoá tệp tạm thời
            os.remove(video_path)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            st.warning(f"Không thể xoá tệp tạm thời: {e}")