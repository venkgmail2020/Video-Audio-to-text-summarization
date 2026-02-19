import streamlit as st
import requests
import tempfile
import os
import time
import nltk
import PyPDF2
import re
from collections import Counter
from gtts import gTTS
import io
import base64
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import ssl
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import validators

# Download NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

st.set_page_config(page_title="Universal Summarizer", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .keyword-tag {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>🎯 Universal Summarizer</h1><p>Video | Audio | PDF | Text | URL</p></div>", unsafe_allow_html=True)

# Initialize session state
if 'assemblyai_key' not in st.session_state:
    st.session_state.assemblyai_key = ''
if 'summary' not in st.session_state:
    st.session_state.summary = ''

# Sidebar
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    assembly_key = st.text_input(
        "AssemblyAI Key",
        value=st.session_state.assemblyai_key,
        type="password"
    )
    if st.button("💾 Save Keys", use_container_width=True):
        st.session_state.assemblyai_key = assembly_key
        st.success("✅ Keys saved!")
    
    st.markdown("---")
    st.markdown("### 📌 Supported")
    st.markdown("🎥 Video: MP4, AVI, MOV")
    st.markdown("🎵 Audio: MP3, WAV, M4A")
    st.markdown("📄 PDF, TXT")
    st.markdown("🌐 URLs")

# ========== PDF EXTRACTION ==========
def extract_pdf_text(pdf_path):
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text, len(pdf_reader.pages)
    except:
        return None, 0

# ========== URL EXTRACTION ==========
def extract_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            title = soup.title.string if soup.title else "Article"
            return text[:50000], title
        return None, None
    except:
        return None, None

# ========== ASSEMBLYAI TRANSCRIPTION ==========
def transcribe_with_assemblyai(audio_path):
    try:
        headers = {'authorization': st.session_state.assemblyai_key}
        
        # Upload
        with open(audio_path, 'rb') as f:
            response = requests.post(
                'https://api.assemblyai.com/v2/upload',
                headers=headers,
                data=f
            )
        upload_url = response.json()['upload_url']
        
        # Transcribe with Telugu support
        transcript_request = {
            'audio_url': upload_url,
            'language_detection': True,  # Auto detects Telugu
            'speech_models': ['universal-2']
        }
        
        response = requests.post(
            'https://api.assemblyai.com/v2/transcript',
            json=transcript_request,
            headers=headers
        )
        transcript_id = response.json()['id']
        
        # Poll for result
        progress = st.progress(0)
        for i in range(60):
            time.sleep(2)
            progress.progress(min(i * 2, 95))
            
            response = requests.get(
                f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                headers=headers
            )
            result = response.json()
            
            if result['status'] == 'completed':
                text = result.get('text', '')
                if len(text) < 20:
                    st.warning("⚠️ Very little text detected. Video lo voice undha?")
                return text
            elif result['status'] == 'error':
                return None
        
        return None
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

# ========== TEXT TO SPEECH ==========
def text_to_speech(text, lang='en'):
    """Convert to audio - FIXED VERSION"""
    try:
        if not text or len(text.strip()) == 0:
            return None
        
        # Limit text length
        text_for_audio = text[:1000] if len(text) > 1000 else text
        
        tts = gTTS(text=text_for_audio, lang=lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.warning(f"⚠️ Audio generation failed: {e}")
        return None

# ========== GENERATE SUMMARY ==========
def generate_summary(text, num_points=5):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= num_points:
        return text
    
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    word_freq = Counter(words)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    sentence_scores = {}
    for i, sent in enumerate(sentences):
        sent_words = re.findall(r'\b[a-zA-Z]{4,}\b', sent.lower())
        score = sum(word_freq.get(word, 0) for word in sent_words if word not in stop_words)
        if 20 < len(sent) < 300:
            sentence_scores[i] = score
    
    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_points]
    top_indices.sort()
    
    summary = "📌 **MAIN POINTS**\n\n"
    for i, idx in enumerate(top_indices, 1):
        summary += f"{i}. {sentences[idx]}\n\n"
    
    return summary

# ========== DISPLAY RESULTS WITH AUDIO ==========
def display_results(text, source_name):
    summary = generate_summary(text, 5)
    
    st.markdown("## 📋 Summary")
    st.markdown(f"<div class='section-card'>{summary}</div>", unsafe_allow_html=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Characters", f"{len(text):,}")
    with col2:
        st.metric("Words", f"{len(text.split()):,}")
    with col3:
        st.metric("Sentences", f"{len(nltk.sent_tokenize(text)):,}")
    with col4:
        reduction = int((1 - len(summary.split())/len(text.split())) * 100)
        st.metric("Reduced", f"{reduction}%")
    
    # DOWNLOAD SECTION - 4 BUTTONS
    st.markdown("### 📥 Downloads")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.download_button("📄 Full Text", text, f"{source_name}_full.txt")
    
    with col2:
        st.download_button("📝 Summary", summary, f"{source_name}_summary.txt")
    
    with col3:
        # 🔥 AUDIO BUTTON - FIXED
        audio = text_to_speech(summary)
        if audio:
            st.audio(audio, format='audio/mp3')
            st.download_button(
                "🔊 Download Audio",
                audio,
                f"{source_name}_audio.mp3",
                "audio/mp3"
            )
        else:
            st.warning("Audio not available")
    
    with col4:
        if st.button("🔗 Share"):
            st.info("Link copied!")
    
    # Keywords
    keywords = Counter(re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())).most_common(10)
    st.markdown("### 🏷️ Keywords")
    html = "<div>"
    for word, count in keywords[:8]:
        html += f"<span class='keyword-tag'>{word}</span> "
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ========== MAIN UI ==========
tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🌐 URL", "📝 Paste Text", "ℹ️ Help"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose file",
        type=['mp4', 'avi', 'mov', 'mp3', 'wav', 'm4a', 'pdf', 'txt']
    )
    
    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📊 {uploaded_file.name} | {file_size:.2f} MB")
        
        if file_ext in ['mp4', 'avi', 'mov']:
            st.video(uploaded_file)
        elif file_ext in ['mp3', 'wav', 'm4a']:
            st.audio(uploaded_file)
        
        if st.button("🚀 Process"):
            with st.spinner("Processing..."):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    path = tmp.name
                
                if file_ext == 'pdf':
                    text, pages = extract_pdf_text(path)
                    if text:
                        st.success(f"✅ Extracted {pages} pages")
                        display_results(text, "pdf")
                elif file_ext == 'txt':
                    with open(path, 'r') as f:
                        text = f.read()
                    display_results(text, "text")
                else:
                    if not st.session_state.assemblyai_key:
                        st.error("❌ AssemblyAI Key required")
                    else:
                        text = transcribe_with_assemblyai(path)
                        if text:
                            st.success(f"✅ Transcribed: {len(text)} chars")
                            display_results(text, "media")
                
                os.unlink(path)

with tab2:
    url = st.text_input("Enter URL", placeholder="https://example.com/article")
    if url and st.button("🌐 Fetch"):
        if validators.url(url):
            with st.spinner("Fetching..."):
                text, title = extract_from_url(url)
                if text:
                    st.success(f"✅ Fetched: {title}")
                    display_results(text, "web")
        else:
            st.error("❌ Invalid URL")

with tab3:
    text_input = st.text_area("Paste text", height=200)
    if text_input and st.button("📝 Summarize"):
        if len(text_input) > 100:
            display_results(text_input, "pasted")
        else:
            st.warning("Text too short")

with tab4:
    st.markdown("""
    ### 📌 How to Use
    1. Get AssemblyAI Key from assemblyai.com
    2. Upload file or paste URL
    3. Click Process
    4. Download text/summary/audio
    
    ### 🌐 Telugu Videos
    - AssemblyAI auto-detects Telugu
    - Make sure video has clear voice
    - 20 seconds is enough
    """)

if __name__ == "__main__":
    main()
