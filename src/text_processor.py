import re
import whisper
from moviepy import AudioFileClip  # dùng để lấy độ dài audio
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st


def clean_transcript(text: str) -> str:
    """
    Làm sạch nhẹ transcript sau khi Whisper phiên âm:
    - Chuẩn hoá khoảng trắng, dấu câu
    - Sửa một số lỗi nghe nhầm phổ biến trong bài giảng ML
    - Có thể tự thêm bớt rule trong dict REPLACEMENTS
    """
    if not text:
        return text

    # 1. Chuẩn hóa khoảng trắng
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)           # nhiều space -> 1 space
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # bỏ space trước dấu câu
    text = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", text)  # thêm space sau dấu nếu thiếu

    # 2. Sửa một số từ ML thường bị nghe sai
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


def transcribe_audio(audio_path: str):
    """
    Phiên âm audio bằng Whisper và trả về:
    {
        "segments": [
            {"text": "...", "start": float, "end": float},
            ...
        ],
        "full_text": "toàn bộ transcript đã clean"
    }
    """
    try:
        # Lấy độ dài audio để ước lượng
        clip = AudioFileClip(audio_path)
        duration = clip.duration  # giây
        clip.close()

        minutes = duration / 60
        # tuỳ máy, Whisper small trên CPU thường chậm hơn thời gian thực ~1–2 lần
        est_minutes = minutes * 1.5

        st.info(
            f"Âm thanh dài khoảng **{minutes:.1f} phút**. "
            f"Thời gian phiên âm ước tính khoảng **{est_minutes:.1f} phút** (tuỳ cấu hình máy)."
        )
        
        # ----- 2. Load model -----
        model = whisper.load_model("small")

        # ----- 3. Chạy phiên âm -----
        result = model.transcribe(
            audio_path,
            fp16=False,
            language="vi",   # ưu tiên tiếng Việt, vẫn giữ thuật ngữ tiếng Anh
        )

        raw_segments = result.get("segments", [])
        segments = []
        all_texts = []

        for seg in raw_segments:
            seg_text = clean_transcript(seg.get("text", ""))
            if not seg_text:
                continue

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": seg_text,
                }
            )
            all_texts.append(seg_text)

        full_text = " ".join(all_texts).strip()

        if not segments:
            st.error("Whisper không trả về segment nào.")
            return None

        return {
            "segments": segments,
            "full_text": full_text,
        }

    except Exception as e:
        st.error(f"Lỗi khi phiên âm âm thanh: {e}")
        return None

def chunk_text(segments, max_chars: int = 2000, overlap_sec: float = 2.0):
    """
    Nhận vào list `segments` (có start/end/text) và gộp thành các chunk lớn hơn,
    mỗi chunk kèm metadata start/end (để nhảy video).

    Trả về:
        chunks: List[str]
        metadatas: List[dict]  (mỗi dict có start/end/chunk_index)
    """
    if not segments:
        st.error("Không có segment để chia nhỏ.")
        return [], []

    chunks = []
    metadatas = []

    cur_text = ""
    cur_start = None
    cur_end = None
    chunk_index = 0

    for seg in segments:
        seg_text = seg["text"]
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])

        # nếu thêm đoạn này vào thì vượt max_chars -> đóng chunk hiện tại
        if cur_text and len(cur_text) + 1 + len(seg_text) > max_chars:
            # đóng chunk
            chunks.append(cur_text.strip())
            metadatas.append(
                {
                    "chunk_index": chunk_index,
                    "start": max(cur_start - overlap_sec, 0.0),
                    "end": cur_end,
                }
            )
            chunk_index += 1
            # mở chunk mới
            cur_text = seg_text
            cur_start = seg_start
            cur_end = seg_end
        else:
            if not cur_text:
                cur_start = seg_start
            else:
                cur_text += " "
            cur_text += seg_text
            cur_end = seg_end

    # chunk cuối
    if cur_text:
        chunks.append(cur_text.strip())
        metadatas.append(
            {
                "chunk_index": chunk_index,
                "start": max(cur_start - overlap_sec, 0.0),
                "end": cur_end,
            }
        )

    return chunks, metadatas
