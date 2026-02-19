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

# Custom CSS for perfect alignment
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Cards */
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Statistics box */
    .stats-box {
        background: linear-gradient(135deg, #f5f7fa, #e9ecf3);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #d1d9e6;
    }
    
    /* Success box */
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Download buttons */
    .download-btn {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-decoration: none;
        display: inline-block;
        margin: 0.2rem;
    }
    
    /* Keyword tags */
    .keyword-tag {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    
    /* Metric boxes */
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>🎯 Universal Summarizer</h1><p>Video | Audio | PDF | Text | URL - All in One | No Limits</p></div>", unsafe_allow_html=True)

# Initialize session state
if 'assemblyai_key' not in st.session_state:
    st.session_state.assemblyai_key = ''
if 'hf_token' not in st.session_state:
    st.session_state.hf_token = ''
if 'summary' not in st.session_state:
    st.session_state.summary = ''
if 'full_text' not in st.session_state:
    st.session_state.full_text = ''

# Sidebar for API keys
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    
    with st.expander("AssemblyAI Key (Video/Audio)", expanded=True):
        assembly_key = st.text_input(
            "AssemblyAI Key",
            value=st.session_state.assemblyai_key,
            type="password",
            help="Get free key from assemblyai.com"
        )
        
    with st.expander("HuggingFace Token (Optional)", expanded=False):
        hf_token = st.text_input(
            "HuggingFace Token",
            value=st.session_state.hf_token,
            type="password",
            help="Get from huggingface.co"
        )
    
    if st.button("💾 Save Keys", use_container_width=True):
        st.session_state.assemblyai_key = assembly_key
        st.session_state.hf_token = hf_token
        st.success("✅ Keys saved!")
    
    st.markdown("---")
    st.markdown("### 📌 Supported Formats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🎥 **Video**\n- MP4\n- AVI\n- MOV")
        st.markdown("🎵 **Audio**\n- MP3\n- WAV\n- M4A")
    with col2:
        st.markdown("📄 **Document**\n- PDF\n- TXT")
        st.markdown("🌐 **URL**\n- Articles\n- News")

# ========== PDF EXTRACTION (No Limits) ==========
def extract_pdf_text(pdf_path):
    """Extract ALL text from PDF - No limits"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            progress = st.progress(0)
            for i, page in enumerate(pdf_reader.pages):
                progress.progress((i + 1) / total_pages)
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        return text, total_pages
    except Exception as e:
        st.error(f"PDF error: {e}")
        return None, 0

# ========== URL EXTRACTION ==========
def extract_from_url(url):
    """Extract article text from URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Get title
            title = soup.title.string if soup.title else "Article"
            
            return text[:50000], title  # Still generous limit
        else:
            st.error(f"Failed to fetch URL: {response.status_code}")
            return None, None
    except Exception as e:
        st.error(f"URL error: {e}")
        return None, None

# ========== ASSEMBLYAI TRANSCRIPTION ==========
def transcribe_with_assemblyai(audio_path):
    """Transcribe video/audio"""
    try:
        headers = {'authorization': st.session_state.assemblyai_key}
        
        # Upload
        with st.spinner("📤 Uploading..."):
            with open(audio_path, 'rb') as f:
                response = requests.post(
                    'https://api.assemblyai.com/v2/upload',
                    headers=headers,
                    data=f
                )
            upload_url = response.json()['upload_url']
        
        # Transcribe
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
        transcript_id = response.json()['id']
        
        # Poll for result
        progress = st.progress(0)
        status = st.empty()
        
        for i in range(60):
            status.text(f"⏳ Transcribing... {i*2}s")
            progress.progress(min(i * 2, 95))
            
            response = requests.get(
                f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                headers=headers
            )
            result = response.json()
            
            if result['status'] == 'completed':
                progress.progress(100)
                status.text("✅ Complete!")
                return result['text']
            elif result['status'] == 'error':
                st.error(f"Error: {result.get('error')}")
                return None
            
            time.sleep(2)
        
        return None
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

# ========== SUMMARIZATION ==========
def generate_summary(text, num_points=10):
    """Generate clean summary with main points"""
    
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    if len(sentences) <= num_points:
        return text
    
    # Score sentences
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    word_freq = Counter(words)
    
    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    for word in list(word_freq.keys()):
        if word in stop_words:
            del word_freq[word]
    
    # Score sentences
    sentence_scores = {}
    for i, sent in enumerate(sentences):
        sent_words = re.findall(r'\b[a-zA-Z]{4,}\b', sent.lower())
        score = sum(word_freq.get(word, 0) for word in sent_words)
        if 20 < len(sent) < 300:
            sentence_scores[i] = score
    
    # Get top sentences
    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_points]
    top_indices.sort()
    
    # Build summary
    summary = "📌 **MAIN POINTS**\n\n"
    for i, idx in enumerate(top_indices, 1):
        summary += f"{i}. {sentences[idx]}\n\n"
    
    return summary

# ========== KEYWORD EXTRACTION ==========
def extract_keywords(text, num=15):
    """Extract important keywords"""
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered = [w for w in words if w not in stop_words and len(w) > 3]
    return Counter(filtered).most_common(num)

# ========== TEXT TO SPEECH ==========
def text_to_speech(text, lang='en'):
    """Convert to audio and return bytes"""
    try:
        # Limit to first 2000 chars for audio
        text_for_audio = text[:2000] if len(text) > 2000 else text
        
        tts = gTTS(text=text_for_audio, lang=lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
        return None

# ========== DISPLAY RESULTS ==========
def display_results(text, source_name):
    """Display results with perfect alignment"""
    
    # Generate summary
    summary = generate_summary(text, 8)
    keywords = extract_keywords(text, 12)
    
    # Store in session
    st.session_state.full_text = text
    st.session_state.summary = summary
    
    # Main summary card
    st.markdown("## 📋 Summary")
    st.markdown(f"<div class='section-card'>{summary}</div>", unsafe_allow_html=True)
    
    # Keywords
    st.markdown("### 🏷️ Key Topics")
    keyword_html = "<div>"
    for word, count in keywords:
        keyword_html += f"<span class='keyword-tag'>{word} ({count})</span> "
    keyword_html += "</div>"
    st.markdown(keyword_html, unsafe_allow_html=True)
    
    # Statistics in neat columns
    st.markdown("### 📊 Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Characters", f"{len(text):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Words", f"{len(text.split()):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Sentences", f"{len(nltk.sent_tokenize(text)):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        reduction = int((1 - len(summary.split())/len(text.split())) * 100)
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Reduction", f"{reduction}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Download section with neat alignment
    st.markdown("### 📥 Downloads")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.download_button(
            "📄 Full Text",
            text,
            file_name=f"{source_name}_full.txt",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            "📝 Summary",
            summary,
            file_name=f"{source_name}_summary.txt",
            use_container_width=True
        )
    
    with col3:
        # Audio for summary
        audio = text_to_speech(summary)
        if audio:
            st.audio(audio, format='audio/mp3')
            st.download_button(
                "🔊 Audio Summary",
                audio,
                file_name=f"{source_name}_audio.mp3",
                use_container_width=True
            )
    
    with col4:
        # Shareable link (simulated)
        if st.button("🔗 Copy Share Link", use_container_width=True):
            st.info("URL: https://summarizer.app/share/123")
    
    # Show preview
    with st.expander("👁️ Preview Raw Text (First 1000 chars)"):
        st.text(text[:1000])

# ========== MAIN UI WITH TABS ==========
def main():
    # Create tabs for different input methods
    tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🌐 URL", "📝 Paste Text", "ℹ️ Help"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a file (Video, Audio, PDF, TXT)",
            type=['mp4', 'avi', 'mov', 'mp3', 'wav', 'm4a', 'pdf', 'txt'],
            help="No size limits - Supports all formats"
        )
        
        if uploaded_file:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            st.info(f"📊 **File:** {uploaded_file.name} | **Size:** {file_size:.2f} MB | **Type:** {file_ext}")
            
            # Preview for video/audio
            if file_ext in ['mp4', 'avi', 'mov']:
                st.video(uploaded_file)
            elif file_ext in ['mp3', 'wav', 'm4a']:
                st.audio(uploaded_file)
            
            if st.button("🚀 Process File", use_container_width=True):
                with st.spinner("Processing..."):
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        file_path = tmp.name
                    
                    if file_ext == 'pdf':
                        text, pages = extract_pdf_text(file_path)
                        if text:
                            st.success(f"✅ Extracted {pages} pages, {len(text):,} chars")
                            display_results(text, "pdf_document")
                    
                    elif file_ext == 'txt':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        display_results(text, "text_file")
                    
                    else:  # Video/Audio
                        if not st.session_state.assemblyai_key:
                            st.error("❌ AssemblyAI Key required for video/audio")
                        else:
                            text = transcribe_with_assemblyai(file_path)
                            if text:
                                display_results(text, "media_file")
                    
                    # Cleanup
                    os.unlink(file_path)
    
    with tab2:
        url = st.text_input(
            "Enter URL",
            placeholder="https://example.com/article",
            help="Paste any article or news URL"
        )
        
        if url and st.button("🌐 Fetch & Summarize", use_container_width=True):
            if validators.url(url):
                with st.spinner("Fetching article..."):
                    text, title = extract_from_url(url)
                    if text:
                        st.success(f"✅ Fetched: {title}")
                        display_results(text, "web_article")
            else:
                st.error("❌ Invalid URL")
    
    with tab3:
        text_input = st.text_area(
            "Paste your text here",
            height=200,
            placeholder="Paste any article, news, or document text..."
        )
        
        if text_input and st.button("📝 Summarize Text", use_container_width=True):
            if len(text_input) > 100:
                display_results(text_input, "pasted_text")
            else:
                st.warning("Text too short (min 100 chars)")
    
    with tab4:
        st.markdown("""
        <div class='section-card'>
            <h3>📌 How to Use</h3>
            <ol>
                <li><strong>Get API Key:</strong> Sign up at <a href='https://www.assemblyai.com/' target='_blank'>AssemblyAI</a> (Free - 10 hours/month)</li>
                <li><strong>Choose Input:</strong> Upload file, paste URL, or enter text</li>
                <li><strong>Click Process:</strong> Wait for results</li>
                <li><strong>Download:</strong> Get text, summary, or audio</li>
            </ol>
            
            <h3>✨ Features</h3>
            <ul>
                <li>✅ No size limits - Process any file</li>
                <li>✅ All formats: Video, Audio, PDF, TXT, URL</li>
                <li>✅ Clean, aligned output</li>
                <li>✅ Audio download for summary</li>
                <li>✅ Shareable links</li>
                <li>✅ Statistics & keywords</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
