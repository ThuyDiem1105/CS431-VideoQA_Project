import os
from moviepy import VideoFileClip
import streamlit as st

def save_uploaded_file(uploaded_file):
    #lưu tệp vào thư mục temp
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
    
def extract_audio(video_path):
    "tách âm thanh và lưu vào tệp .mp3"
    try:
        video_clip = VideoFileClip(video_path)
        audio_path = os.path.join("temp", os.path.basename(video_path).replace(".mp4", ".mp3"))
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        return audio_path
    except Exception as e:
        st.error(f"Lỗi khi tách âm thanh: {e}")
        return None