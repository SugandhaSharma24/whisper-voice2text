# app.py
import os
import streamlit as st
import whisper
import torch
import tempfile
import subprocess
from datetime import datetime

# ======================
# Environment Configuration
# ======================
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

# ======================
# System Verification
# ======================
def verify_system_dependencies():
    """Check for required system binaries"""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True,
                      stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL)
    except Exception as e:
        st.error("""
        FFmpeg not found! Required for audio processing.
        Add these to packages.txt:
        ffmpeg liblapack3 libopenblas-base libgfortran5
        """)
        st.stop()

# ======================
# Model Loading
# ======================
@st.cache_resource
def load_whisper_model():
    """Load Whisper model with CPU optimization"""
    try:
        model = whisper.load_model(
            "base",
            device="cpu",
            download_root="/tmp/whisper_models",
            _progress=False  # Disable progress bars
        )
        return model
    except Exception as e:
        st.error(f"""
        Model loading failed: {str(e)}
        Verify requirements.txt contains:
        torch==2.2.2+cpu
        openai-whisper==20231117
        """)
        st.stop()

# ======================
# File Handling
# ======================
def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp location"""
    try:
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"File handling error: {str(e)}")
        return None

# ======================
# Main Application
# ======================
def main():
    # System checks
    verify_system_dependencies()
    
    # Load model
    model = load_whisper_model()
    
    # Configure page
    st.set_page_config(page_title="Whisper Voice2Text", layout="wide")
    st.title("🎤 Whisper AI Voice to Text Converter")
    st.markdown("""
    Upload an audio file (mp3, wav, mp4, etc.) to convert speech to text
    using OpenAI's Whisper model.
    """)

    # Debug panel
    with st.expander("System Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"PyTorch: {torch.__version__}")
            st.write(f"Whisper: {whisper.__version__}")
            st.write(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        with col2:
            st.write(f"Python: {os.sys.version}")
            st.write(f"FFmpeg: {subprocess.check_output(['ffmpeg', '-version']).decode().splitlines()[0]}")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["mp3", "wav", "m4a", "mp4", "ogg", "flac"],
        accept_multiple_files=False
    )

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing... Please wait..."):
                try:
                    # Save temp file
                    temp_path = save_uploaded_file(uploaded_file)
                    if not temp_path:
                        return

                    # Transcribe with progress
                    progress_bar = st.progress(0)
                    result = model.transcribe(temp_path, fp16=False)
                    progress_bar.progress(100)

                    # Show results
                    st.subheader("Transcription Result")
                    st.code(result["text"], language="text")

                    # Detailed segments
                    with st.expander("Detailed Timestamps"):
                        for seg in result["segments"]:
                            st.write(f"{seg['start']:.1f}s - {seg['end']:.1f}s: {seg['text']}")

                    # Cleanup
                    os.unlink(temp_path)

                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")
                    if "temp_path" in locals():
                        try: os.unlink(temp_path)
                        except: pass

# ======================
# Entry Point
# ======================
if __name__ == "__main__":
    main()