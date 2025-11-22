import os, shutil
import streamlit as st
import base64
import streamlit.components.v1 as components

from langchain_core.documents import Document
from src.video_processor import save_uploaded_file, extract_audio
from src.text_processor import transcribe_audio, chunk_text
from src.vector_store_builder import create_and_save_vector_store
from src.rag_qa import build_rag_pipeline, ask_question, MODEL_OPTIONS

# ===== KHỞI TẠO TRẠNG THÁI PHIÊN =====
def init_session_state():
    if "qa_ready" not in st.session_state:
        st.session_state["qa_ready"] = False
    if "is_processing" not in st.session_state:
        st.session_state["is_processing"] = False
    if "video_name" not in st.session_state:
        st.session_state.video_name = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_model_id" not in st.session_state:
        st.session_state.current_model_id = None
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

# ===== CÁC HÀM TIỆN ÍCH =====
def reset_state_for_new_video(filename: str):
    st.session_state.video_name = filename
    st.session_state.processed = False
    st.session_state.pipeline = None
    st.session_state.messages = []
    st.session_state.qa_ready = False
    st.session_state.is_processing = False
    st.session_state.current_model_id = None
    st.session_state.last_sources = []

def apply_global_styles():
    st.markdown("""
    <style>
    /* Theme sáng pastel (mặc định cho light mode) */
    .main { background-color: #f0f8ff; color: #333; }
    .block-container { max-width: 1200px; padding: 2rem; }
    h1, h2, h3 { color: #ffb3ba; font-family: 'Segoe UI', sans-serif; }
    .stButton>button { background-color: #bae1ff; color: #333; border-radius: 8px; border: none; padding: 0.5rem 1rem; transition: 0.3s; }
    .stButton>button:hover { background-color: #a8d8ff; transform: scale(1.05); }
    .stTextInput, .stSelectbox, .stFileUploader { border-radius: 8px; border: 1px solid #ddd; background-color: #fefefe; color: #333; margin-bottom: 1rem; padding: 0.5rem; }
    .stChatMessage { border-radius: 12px; background-color: #fefefe; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .stExpander { background-color: #fefefe; border-radius: 8px; border: 1px solid #eee; }
    .step-card { background: linear-gradient(135deg, #ffb3ba 0%, #bae1ff 100%); border-radius: 15px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 8px rgba(0,0,0,0.1); color: #333; }
    .step-card.step-2 { background: linear-gradient(135deg, #ffdfba 0%, #d4a5ff 100%); }
    .badge { background: #ffdfba; color: #333; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }
    .badge.success { background: #baffc9; }
    
    /* ===== SIDEBAR HỒNG PASTEL ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffd4e5 0%, #ffe8f0 50%, #fff5f8 100%) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1.5rem;
    }
    
    /* Text trong sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #5a3a4a !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #d6537c !important;  /* Màu đậm hơn cho tiêu đề */
    }
    
    [data-testid="stSidebar"] strong {
        color: #d6537c !important;  /* Màu đậm hơn cho Bước 1, 2 */
    }
    
    /* Divider trong sidebar - RÕ HƠN */
    [data-testid="stSidebar"] hr {
        border: none !important;
        height: 2px !important;
        background: #ffb3c6 !important;
        opacity: 0.6 !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Dark mode (khi browser bật dark mode) */
    @media (prefers-color-scheme: dark) {
        .main { background-color: #2a2a3a; color: #f0f8ff; }  /* Nền tối pastel */
        h1, h2, h3 { color: #ffb3ba; }  /* Giữ hồng pastel cho tiêu đề */
        .stButton>button { background-color: #4a5d7a; color: #f0f8ff; }  /* Xanh xám tối pastel */
        .stButton>button:hover { background-color: #5a6d8a; }
        .stTextInput, .stSelectbox, .stFileUploader { background-color: #3a3a4a; color: #f0f8ff; border: 1px solid #555; margin-bottom: 1rem; padding: 0.5rem; }
        .stChatMessage { background-color: #3a3a4a; box-shadow: 0 2px 4px rgba(255,255,255,0.1); }
        .stExpander { background-color: #3a3a4a; border: 1px solid #555; }
        .step-card { background: linear-gradient(135deg, #6b5b95 0%, #4a5d7a 100%); color: #f0f8ff; box-shadow: 0 4px 8px rgba(255,255,255,0.1); }  /* Gradient tím-xanh tối pastel */
        .step-card.step-2 { background: linear-gradient(135deg, #8b7d6b 0%, #5a4a7a 100%); }  /* Gradient nâu-tím tối pastel */
        .badge { background: #8b7d6b; color: #f0f8ff; }  /* Nâu tối pastel */
        .badge.success { background: #5a7a5a; }  /* Xanh lá tối pastel */
        
        /* SIDEBAR DARK MODE */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #6b4f5e 0%, #5a3a4a 50%, #4a2e3e 100%) !important;
        }
        
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: #ffd4e5 !important;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #ff8ba7 !important;  /* Màu đậm hơn cho dark mode */
        }
        
        [data-testid="stSidebar"] strong {
            color: #ff8ba7 !important;
        }
        
        [data-testid="stSidebar"] hr {
            background: #8a5a6a !important;
            opacity: 0.7 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        # Header
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0 1.5rem 0;'>
            <div style='font-size: 3.5rem;'>🎓</div>
            <h2 style='margin: 0; font-size: 1.5rem; font-weight: 700; color: #d6537c !important;'>Hướng Dẫn Sử Dụng</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Hướng dẫn
        st.markdown("""
        <div style='padding: 0.5rem 0; line-height: 1.8;'>
            <p style='margin-bottom: 1rem;'>
                <strong style='font-size: 1rem; color: #d6537c;'>• Bước 1:</strong> 
                <span style='display: block; margin-left: 1.2rem; margin-top: 0.3rem;'>
                    Upload video (.mp4) và xử lý để tạo vector store.
                </span>
            </p>
            <p style='margin-bottom: 1rem;'>
                <strong style='font-size: 1rem; color: #d6537c;'>• Bước 2:</strong> 
                <span style='display: block; margin-left: 1.2rem; margin-top: 0.3rem;'>
                    Chọn mô hình LLM và hỏi đáp dựa trên nội dung video.
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Divider rõ hơn trước footer
        st.markdown("""
        <div style='margin: 2rem 0 1.5rem 0;'>
            <hr style='border: none; height: 2px; background: #ffb3c6; opacity: 0.6;'>
        </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div style='text-align: center; padding: 0 0 1rem 0;'>
            <p style='margin: 0; font-size: 0.9rem; opacity: 0.7;'>Đồ án CS431 - UIT</p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem; opacity: 0.6;'>Powered by RAG & LLM</p>
        </div>
        """, unsafe_allow_html=True)

def render_hero():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🎓 Hệ Thống Hỏi Đáp Video CS431</h1>
        <p style="color: #666; font-size: 1.1rem;">Ứng dụng RAG cho video bài giảng UIT – Demo đồ án CS431</p>
    </div>
    """, unsafe_allow_html=True)

def render_step1():
    st.markdown("""
    <div class="step-card">
        <h3>📤 Bước 1: Upload & Xử Lý Video</h3>
        <p>Chọn video bài giảng (.mp4), sau đó nhấn nút để tách âm thanh, phiên âm và xây dựng vector store.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Chọn tệp video (.mp4):", type=["mp4"])
    
    if uploaded_file:
        if st.session_state.video_name != uploaded_file.name:
            reset_state_for_new_video(uploaded_file.name)
        
        st.markdown(f"**Video hiện tại:** `{uploaded_file.name}` <span class='badge'>Đã tải lên</span>", unsafe_allow_html=True)
        
        col_btn, col_status = st.columns([1, 3])

        with col_btn:
            process_btn = st.button(
                "🚀 Bắt đầu xử lý video",
                type="primary",
                disabled=st.session_state.processed,
            )
        
        with col_status:
            if st.session_state.processed:
                st.markdown("<div class='badge success'>✅ Video đã xử lý xong</div>", unsafe_allow_html=True)
        
        # CHỈ xử lý video khi user bấm nút
        if process_btn:
            with st.spinner("Đang xử lý video..."):
                process_video(uploaded_file)
    else:
        st.info("Vui lòng tải lên một video để bắt đầu.")

def format_time(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

def render_step2():
    st.markdown("""
    <div class="step-card step-2">
        <h3>💬 Bước 2: Hỏi Đáp Với AI</h3>
        <p>Chọn mô hình LLM và đặt câu hỏi dựa trên nội dung video đã xử lý.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.qa_ready:
        st.warning("Vui lòng hoàn thành Bước 1 trước.")
        return
    
    # Chọn mô hình
    placeholder = "— Chọn mô hình LLM —"
    model_labels = [placeholder] + list(MODEL_OPTIONS.keys())
    selected_label = st.selectbox("Chọn mô hình LLM:", model_labels, index=0, key="model_select")
    model_chosen = selected_label != placeholder
    selected_model_id = MODEL_OPTIONS[selected_label] if model_chosen else None
    
    if model_chosen:
        st.caption(f"Model ID: `{selected_model_id}`")
        # Xây dựng pipeline nếu cần
        if st.session_state.pipeline is None or st.session_state.current_model_id != selected_model_id:
            with st.spinner("Khởi tạo pipeline..."):
                st.session_state.pipeline = build_rag_pipeline(selected_model_id)
            st.session_state.current_model_id = selected_model_id
        pipeline = st.session_state.pipeline
        
        # Hiển thị chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ====================== VIDEO + PLAYLIST TIMELINE ======================
        if st.session_state.last_sources:
            video_path = st.session_state.get("video_path")
            if not video_path or not os.path.exists(video_path):
                st.warning("Không tìm thấy file video để phát lại.")
            else:
                with open(video_path, "rb") as vf:
                    video_bytes = vf.read()
                video_b64 = base64.b64encode(video_bytes).decode("utf-8")

                sources = st.session_state.last_sources

                buttons_html = ""
                for i, doc in enumerate(sources, 1):
                    meta = getattr(doc, "metadata", {}) or {}
                    start = meta.get("start")
                    end = meta.get("end")

                    if start is not None:
                        label_time = format_time(start)
                        label = f"Đoạn {i} (≈ {label_time})"
                        buttons_html += f"""
                        <button onclick="seekTo({start:.1f})"
                            style="margin:4px; padding:4px 10px; border-radius:999px;
                                border:none; background:#ffb3ba; color:#333;
                                cursor:pointer; font-size:0.8rem;">
                            ▶ {label}
                        </button>
                        """
                    else:
                        buttons_html += f"""
                        <button disabled
                            style="margin:4px; padding:4px 10px; border-radius:999px;
                                border:none; background:#ddd; color:#777;
                                font-size:0.8rem;">
                            Đoạn {i} (không có timestamp)
                        </button>
                        """

                html = f"""
                <div style="margin-top:1rem; text-align:center;">
                  <video id="video-player" controls
                    style="max-height:440px; object-fit:contain">
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4" />
                    Trình duyệt không hỗ trợ phát video.
                  </video>
                  <div style="margin-top:0.5rem; display:flex; flex-wrap:wrap; justify-content:center;">
                    {buttons_html}
                  </div>
                </div>
                <script>
                function seekTo(t) {{
                  var v = document.getElementById("video-player");
                  if (v) {{
                    v.currentTime = t;
                    v.play();
                  }}
                }}
                </script>
                """

                with st.expander(
                    "🎬 Xem lại các đoạn trong video (theo câu trả lời gần nhất)",
                    expanded=True
                ):
                    components.html(html, height=500, scrolling=False)
        
        # ====================== CHAT INPUT ======================
        new_question = st.chat_input("Nhập câu hỏi của bạn...")
        if new_question:
            st.session_state.messages.append({"role": "user", "content": new_question})
            result = ask_question(pipeline, new_question)
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            st.session_state.last_sources = result["sources"]
            st.rerun()
    else:
        st.caption("⚠️ Vui lòng chọn mô hình trước khi hỏi đáp.")

def process_video(uploaded_file):
    st.session_state.is_processing = True
    st.session_state.qa_ready = False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Lưu video
    status_text.text("📂 Đang lưu video...")
    video_path = save_uploaded_file(uploaded_file)
    st.session_state["video_path"] = video_path
    progress_bar.progress(20)
    st.success("Video đã lưu thành công! ✅")
    
    # 2. Tách âm thanh
    status_text.text("🎧 Đang tách âm thanh...")
    audio_path = extract_audio(video_path)
    if not audio_path:
        st.error("Lỗi: Không thể tách âm thanh.")
        st.session_state.is_processing = False
        return
    progress_bar.progress(40)
    st.success("Âm thanh đã tách! 🎵")

    # 3. Phiên âm
    status_text.text("📝 Đang phiên âm...")
    trans_result = transcribe_audio(audio_path)
    if not trans_result:
        st.error("Lỗi: Không thể phiên âm.")
        st.session_state.is_processing = False
        return
    
    segments = trans_result["segments"]
    progress_bar.progress(60)
    st.success("Phiên âm hoàn tất! 📜")
    
    # 4. Chia nhỏ + tạo Document có metadata start/end
    status_text.text("✂️ Đang chia nhỏ văn bản...")
    chunks, metadatas = chunk_text(segments)
    if not chunks:
        st.error("Lỗi: Không thể chia nhỏ.")
        st.session_state.is_processing = False
        return
    progress_bar.progress(80)
    st.success(f"Chia nhỏ thành {len(chunks)} đoạn! ✂️")

    # 5. Tạo vector store (kèm metadata start/end)
    status_text.text("📦 Đang tạo vector store...")
    documents = []
    for i in range(len(chunks)):
        documents.append(
            Document(
                page_content=chunks[i],
                metadata={
                    "start": metadatas[i]["start"],   # 👈 Đổi thành start
                    "end": metadatas[i]["end"],       # 👈 Đổi thành end
                    "video_name": uploaded_file.name, # thêm cũng được
                },
            )
        )


    if create_and_save_vector_store(documents):
        progress_bar.progress(100)
        st.session_state.qa_ready = True
        st.session_state.is_processing = False
        st.session_state.processed = True
        st.success("Vector store sẵn sàng! Bây giờ bạn có thể chuyển sang Bước 2 để hỏi đáp. 🎉")        
        
        # Xoá file audio tạm, giữ video để phát lại
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            st.warning(f"Không thể xoá file audio tạm: {e}")

        st.rerun()
    else:
        st.error("Lỗi: Không thể tạo vector store.")
        st.session_state.is_processing = False

# ===== ỨNG DỤNG CHÍNH =====
def main():
    st.set_page_config(layout="wide", page_title="Hệ thống Hỏi đáp Video CS431", page_icon="🎓")
    init_session_state()
    apply_global_styles()
    render_sidebar()
    render_hero()

    tab1, tab2 = st.tabs(["📤 Bước 1: Upload & Xử Lý", "💬 Bước 2: Hỏi Đáp"])
    with tab1:
        render_step1()
    with tab2:
        render_step2()

if __name__ == "__main__":
    main()