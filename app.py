# app.py
import streamlit as st
import whisper
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# Set page config
st.set_page_config(page_title="Whisper Voice2Text", layout="wide")

st.title("🎤 Whisper AI Voice to Text Converter")
st.markdown("""
Convert speech to text using OpenAI's Whisper model. Choose either:
1. 🗄️ Upload an audio file
2. 🎤 Record live audio
""")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def process_audio(file_path):
    return load_whisper_model().transcribe(file_path)

def main():
    model = load_whisper_model()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("File Upload")
        uploaded_file = st.file_uploader("Choose audio", 
                                       type=["mp3", "wav", "m4a", "mp4"],
                                       accept_multiple_files=False)
        if uploaded_file and st.button("Transcribe File"):
            with tempfile.NamedTemporaryFile(suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                result = process_audio(tmp_file.name)
                show_results(result)

    with col2:
        st.subheader("Live Recording")
        audio_bytes = audio_recorder(text="Click to record", 
                                  pause_threshold=5,
                                  neutral_color="#6aa36f",
                                  recording_color="#e34500")
        
        if audio_bytes and st.button("Transcribe Recording"):
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                result = process_audio(tmp_file.name)
                show_results(result)

def show_results(result):
    if result:
        st.subheader("Transcription")
        st.write(result["text"])
        with st.expander("Detailed Segments"):
            for seg in result["segments"]:
                st.write(f"{seg['start']:.1f}s - {seg['end']:.1f}s: {seg['text']}")

if __name__ == "__main__":
    main()