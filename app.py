# app.py
import streamlit as st
import whisper
import torch
import os
import tempfile
import subprocess

# Environment configuration
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Set page title and layout
st.set_page_config(page_title="Whisper Voice2Text", layout="wide")

# Page header
st.title("🎤 Whisper AI Voice to Text Converter")
st.markdown("""
Upload an audio file (mp3, wav, mp4, etc.) and convert speech to text using OpenAI's Whisper model.
""")

# Function to load Whisper model
@st.cache_resource
def load_whisper_model():
    try:
        # Verify FFmpeg installation
        subprocess.run(['ffmpeg', '-version'], check=True, 
                      stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        
        # Load model with CPU optimization
        model = whisper.load_model(
            "base",
            device="cpu",
            download_root="/tmp/whisper_models"
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to save uploaded file temporarily
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling file: {e}")
        return None

# Main function
def main():
    model = load_whisper_model()
    
    # System info debug
    with st.expander("System Information"):
        st.write(f"PyTorch version: {torch.__version__}")
        st.write(f"Whisper commit: {whisper.__version__}")
        st.write(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["mp3", "wav", "m4a", "mp4", "ogg", "flac"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing... This might take a while..."):
                try:
                    temp_file_path = save_uploaded_file(uploaded_file)
                    
                    # Add progress bar
                    progress_bar = st.progress(0)
                    result = model.transcribe(temp_file_path, fp16=False)
                    progress_bar.progress(100)
                    
                    # Display results
                    st.subheader("Transcription Result")
                    st.code(result["text"], language="txt")
                    
                    # Detailed information
                    with st.expander("Show Detailed Information"):
                        for segment in result["segments"]:
                            st.write(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
                    
                    # Cleanup
                    os.unlink(temp_file_path)
                    
                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")
                    if "ffmpeg" in str(e).lower():
                        st.error("FFmpeg not properly configured. Check packages.txt")

# Run the app
if __name__ == "__main__":
    main()