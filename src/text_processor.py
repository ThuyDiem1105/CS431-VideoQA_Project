import whisper
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

def transcribe_audio(audio_path):
    #phiên âm âm thanh bằng Whisper
    try:
        # Tải mô hình 'base' (nhanh, cân bằng)
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, fp16=False)
        return result['text']
    except Exception as e:
        st.error(f"Lỗi khi phiên âm âm thanh: {e}")
        return None

def chunk_text(transcript):
    # chia nhỏ văn bản thành các đoạn
    if not transcript:
        st.error("Không có văn bản để chia nhỏ.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(transcript)
    return chunks