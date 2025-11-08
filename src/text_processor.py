import re
import whisper
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

def clean_transcript(text: str) -> str:
    """
    Làm sạch nhẹ transcript sau khi Whisper phiên âm:
    - Chuẩn hoá khoảng trắng, dấu câu
    - Sửa một số lỗi nghe nhầm phổ biến trong bài giảng ML
    - Gà có thể tự thêm bớt rule trong dict REPLACEMENTS
    """
    if not text:
        return text

    # 1. Chuẩn hóa khoảng trắng
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)           # nhiều space -> 1 space
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # bỏ space trước dấu câu
    text = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", text)  # thêm space sau dấu nếu thiếu

    # 2. Sửa một số từ ML thường bị nghe sai (gà chỉnh thêm nếu muốn)
    REPLACEMENTS = {
        # tiếng Anh kỹ thuật
        r"\blogit stick\b": "logistic",
        r"\blogit sik\b": "logistic",
        r"\blogit\b": "logit",
        r"\blót\s+sí(ch|t)\b": "loss",
        r"\blót\b": "loss",
        r"\bcro(x|ss)\s+entơ(pi|py)\b": "cross entropy",
        r"\bgrê đi en\b": "gradient",
        r"\bgradiền\b": "gradient",
        r"\bđi sen\b": "descent",
        r"\bthê ta\b": "theta",
        r"\bthê tơ\b": "theta",

        # một số lỗi tiếng Việt hay gặp
        r"\bhạm\b": "hàm",
        r"\bhảm\b": "hàm",
        r"\bđạo hảm\b": "đạo hàm",
    }

    for pattern, repl in REPLACEMENTS.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # 3. Cắt bớt space dư ở đầu/cuối
    text = text.strip()

    return text

def transcribe_audio(audio_path):
    #phiên âm âm thanh bằng Whisper
    try:
        # Tải mô hình 'base' (nhanh, cân bằng)
        model = whisper.load_model("small")
        result = model.transcribe(audio_path, fp16=False, language='vi') # ép tiếng Việt, nhưng vẫn giữ các từ tiếng Anh
        raw_text = result["text"]
        cleaned_text = clean_transcript(raw_text)
        return cleaned_text
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