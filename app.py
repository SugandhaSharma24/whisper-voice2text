# app.py
import streamlit as st
import whisper
from datetime import datetime
import os
import tempfile
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR" 

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
        model = whisper.load_model("base", device="cpu")  # Change to "small", "medium", or "large" for better accuracy
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to save uploaded file temporarily
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling file: {e}")
        return None

# Main function
def main():
    model = load_whisper_model()
    
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
                    # Save temp file
                    temp_file_path = save_uploaded_file(uploaded_file)
                    
                    # Transcribe audio
                    result = model.transcribe(temp_file_path)
                    
                    # Display results
                    st.subheader("Transcription Result")
                    st.code(result["text"], language="txt")
                    
                    # Show additional information
                    with st.expander("Show Detailed Information"):
                        st.write("Segments:")
                        for segment in result["segments"]:
                            st.write(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
                        
                        st.write("Full JSON Output:")
                        st.json(result)
                    
                    # Clean up temp file
                    os.unlink(temp_file_path)
                    
                except Exception as e:
                    st.error(f"Transcription failed: {e}")

# Run the app
if __name__ == "__main__":
    main()