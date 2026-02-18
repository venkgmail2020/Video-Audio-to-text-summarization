import streamlit as st
import requests
import tempfile
import os
import time
import nltk
import ssl
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# ===== FIXED NLTK DOWNLOAD FOR STREAMLIT CLOUD =====
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data path
nltk_data_dir = '/home/appuser/nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add to path
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Download required data
required_packages = ['punkt', 'punkt_tab']
for package in required_packages:
    try:
        nltk.data.find(f'tokenizers/{package}')
    except:
        nltk.download(package, download_dir=nltk_data_dir, quiet=True)
# ==================================================

st.set_page_config(page_title="Video Summarizer", page_icon="🎥")

st.title("🎥 Video/Audio/PDF Summarizer")
st.write("✅ Fixed PDF + AssemblyAI")

# ========== PDF EXTRACTION ==========

def extract_pdf_text(pdf_path):
    """Extract text from PDF using PyPDF2"""
    try:
        import PyPDF2
        text = ""
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            st.info(f"📄 PDF has {num_pages} pages")
            
            # Read first 5 pages only
            for page_num in range(min(5, num_pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    st.caption(f"✅ Page {page_num+1} extracted")
        
        if text:
            st.success(f"✅ Extracted {len(text)} characters")
            return text[:10000]
        else:
            st.warning("No text found in PDF")
            return None
    except Exception as e:
        st.error(f"❌ PDF extraction failed: {e}")
        return None

# ========== ASSEMBLYAI TRANSCRIPTION ==========

def transcribe_with_assemblyai(audio_path):
    """Transcribe using AssemblyAI"""
    try:
        headers = {'authorization': st.session_state.assemblyai_key}
        
        # Upload file
        with st.spinner("📤 Uploading to AssemblyAI..."):
            with open(audio_path, 'rb') as f:
                response = requests.post(
                    'https://api.assemblyai.com/v2/upload',
                    headers=headers,
                    data=f
                )
            
            if response.status_code != 200:
                st.error(f"Upload failed: {response.text}")
                return None
            
            upload_url = response.json().get('upload_url')
            if not upload_url:
                st.error("No upload_url in response")
                return None
        
        # Request transcription
        with st.spinner("⏳ Requesting transcription..."):
            transcript_request = {
                'audio_url': upload_url,
                'language_detection': True,
                'speech_models': ['universal-2']
            }
            
            response = requests.post(
                'https://api.assemblyai.com/v2/transcript',
                json=transcript_request,
                headers=headers
            )
            
            if response.status_code != 200:
                st.error(f"Transcription request failed: {response.text}")
                return None
            
            result = response.json()
            transcript_id = result.get('id')
            
            if not transcript_id:
                st.error(f"No ID in response: {result}")
                return None
        
        # Poll for result
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(30):
            status_text.text(f"⏳ Transcribing... ({i*2} seconds)")
            progress_bar.progress(min(i * 3, 90))
            
            response = requests.get(
                f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                status = result.get('status')
                
                if status == 'completed':
                    progress_bar.progress(100)
                    status_text.text("✅ Transcription complete!")
                    return result.get('text', '')
                elif status == 'error':
                    st.error(f"Transcription failed: {result.get('error', 'Unknown error')}")
                    return None
            
            time.sleep(2)
        
        st.error("Transcription timeout")
        return None
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

# ========== SUMMARIZATION ==========

def summarize_text(text):
    """Summarize text"""
    if not text or len(text) < 100:
        st.warning("Text too short to summarize")
        return
    
    with st.spinner("📝 Generating summary..."):
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LexRankSummarizer()
            summary = summarizer(parser.document, 3)
            summary_text = " ".join(str(s) for s in summary)
            
            st.session_state.transcript = text
            st.session_state.summary = summary_text
            
            # Display
            st.markdown("---")
            st.markdown("## 📊 Results")
            
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("📄 Original Text", expanded=False):
                    st.text(text[:500] + ("..." if len(text) > 500 else ""))
            
            with col2:
                st.markdown("### 📝 Summary")
                st.info(summary_text)
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download Transcript", text, "transcript.txt")
            with col2:
                st.download_button("📥 Download Summary", summary_text, "summary.txt")
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original", f"{len(text)} chars")
            with col2:
                st.metric("Summary", f"{len(summary_text)} chars")
            with col3:
                reduction = int((1 - len(summary_text)/len(text)) * 100) if len(text) > 0 else 0
                st.metric("Reduced", f"{reduction}%")
                
        except Exception as e:
            st.error(f"Summarization failed: {e}")

def process_file(uploaded_file):
    """Process uploaded file"""
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    with st.status("Processing...", expanded=True) as status:
        try:
            # Save temp file
            status.update(label="📁 Saving file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp:
                tmp.write(uploaded_file.getvalue())
                file_path = tmp.name
            
            # Handle different file types
            if file_ext == 'pdf':
                status.update(label="📄 Extracting PDF text...")
                text = extract_pdf_text(file_path)
                if text:
                    summarize_text(text)
            
            elif file_ext == 'txt':
                status.update(label="📝 Reading text file...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                summarize_text(text)
            
            else:  # Audio/Video
                status.update(label="🎤 Transcribing with AssemblyAI...")
                text = transcribe_with_assemblyai(file_path)
                if text:
                    summarize_text(text)
            
            # Cleanup
            os.unlink(file_path)
            status.update(label="✅ Complete!", state="complete")
            
        except Exception as e:
            status.update(label="❌ Error!", state="error")
            st.error(f"Error: {str(e)}")

# ========== UI ==========

# Initialize session state
if 'assemblyai_key' not in st.session_state:
    st.session_state.assemblyai_key = ''

# Sidebar
with st.sidebar:
    st.header("🔑 API Keys")
    
    # Try to get from secrets first
    default_key = st.secrets.get("ASSEMBLYAI_KEY", "") if hasattr(st, 'secrets') else ""
    
    assembly_key = st.text_input(
        "AssemblyAI Key",
        value=st.session_state.assemblyai_key or default_key,
        type="password"
    )
    
    if st.button("Save Keys", use_container_width=True):
        st.session_state.assemblyai_key = assembly_key
        st.success("✅ Keys saved!")
    
    st.markdown("---")
    st.markdown("### 📌 Note")
    st.markdown("✅ PDF: PyPDF2")
    st.markdown("✅ AssemblyAI: universal-2")

# Main content
uploaded_file = st.file_uploader(
    "Choose file",
    type=['mp4', 'mp3', 'wav', 'pdf', 'txt', 'avi', 'mov']
)

if uploaded_file:
    # File info
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
    st.info(f"📊 Size: {file_size:.2f} MB")
    
    # Preview
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext in ['mp4', 'avi', 'mov']:
        st.video(uploaded_file)
    elif file_ext in ['mp3', 'wav']:
        st.audio(uploaded_file)
    
    # Process button
    if st.button("🚀 Process File", type="primary", use_container_width=True):
        if not st.session_state.assemblyai_key and file_ext not in ['pdf', 'txt']:
            st.error("❌ Please enter AssemblyAI Key for audio/video files")
        else:
            process_file(uploaded_file)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Fixed for Streamlit Cloud")
