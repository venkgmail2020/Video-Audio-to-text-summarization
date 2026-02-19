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
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

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
    .slider-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>🎯 Universal Summarizer</h1><p>Video | Audio | PDF | Text | URL | YouTube</p></div>", unsafe_allow_html=True)

# Initialize session state
if 'assemblyai_key' not in st.session_state:
    st.session_state.assemblyai_key = ''
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}

# Sidebar
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    assembly_key = st.text_input(
        "AssemblyAI Key",
        value=st.session_state.assemblyai_key,
        type="password",
        help="Required for YouTube videos without captions"
    )
    if st.button("💾 Save Keys", use_container_width=True):
        st.session_state.assemblyai_key = assembly_key
        st.success("✅ Keys saved!")
    
    st.markdown("---")
    st.markdown("### 📌 Supported")
    st.markdown("🎥 Video: MP4, AVI, MOV")
    st.markdown("🎵 Audio: MP3, WAV, M4A")
    st.markdown("📄 PDF, TXT")
    st.markdown("🌐 Articles")
    st.markdown("▶️ YouTube (auto-transcribe)")

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
    except Exception as e:
        st.error(f"PDF error: {e}")
        return None, 0

# ========== ARTICLE URL EXTRACTION ==========
def extract_article_from_url(url):
    """Extract article content from URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Get title
            title = soup.title.string if soup.title else "Article"
            title = re.sub(r'[|\-].*$', '', title).strip()
            
            # Get all paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs if len(p.get_text()) > 40])
            
            # Clean text
            text = re.sub(r'\s+', ' ', text).strip()
            
            if text and len(text) > 200:
                return text, title
        return None, None
    except Exception as e:
        st.error(f"URL extraction error: {e}")
        return None, None

# ========== YOUTUBE HANDLER ==========
def extract_youtube_content(url):
    """Extract content from YouTube video"""
    try:
        # Extract video ID
        video_id = None
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('watch?v=')[-1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[-1].split('?')[0]
        
        if not video_id:
            return None, None, "Invalid YouTube URL"
        
        # METHOD 1: Try to get captions
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try languages in order
            for lang in ['te', 'en', 'hi']:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    transcript_data = transcript.fetch()
                    full_text = ' '.join([item['text'] for item in transcript_data])
                    
                    # Get video title
                    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                        info = ydl.extract_info(url, download=False)
                        title = info.get('title', 'YouTube Video')
                    
                    return full_text, f"YouTube: {title} (Captions)", None
                except:
                    continue
        except:
            pass
        
        # METHOD 2: Download and transcribe with AssemblyAI
        if not st.session_state.assemblyai_key:
            return None, None, "No captions found. AssemblyAI Key required for transcription"
        
        with st.spinner("📥 Downloading audio from YouTube..."):
            # Download audio
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': 'youtube_audio.%(ext)s',
                'quiet': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'YouTube Video')
            
            audio_path = 'youtube_audio.mp3'
            
            # Transcribe with AssemblyAI
            st.info("🎤 Transcribing audio with AssemblyAI...")
            text = transcribe_with_assemblyai(audio_path)
            
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            if text:
                return text, f"YouTube: {title} (Transcribed)", None
            else:
                return None, None, "Transcription failed. Audio might be silent or API error."
        
    except Exception as e:
        return None, None, f"YouTube error: {str(e)}"

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
                return result.get('text', '')
            elif result['status'] == 'error':
                return None
            
            time.sleep(2)
        
        return None
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

# ========== TEXT TO SPEECH ==========
def text_to_speech(text):
    try:
        if not text or len(text.strip()) == 0:
            return None
        
        text_for_audio = text[:1000] if len(text) > 1000 else text
        tts = gTTS(text=text_for_audio, lang='en', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.warning(f"Audio failed: {e}")
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
    
    summary = f"📌 **MAIN POINTS ({num_points} of {len(sentences)} sentences)**\n\n"
    for i, idx in enumerate(top_indices, 1):
        summary += f"{i}. {sentences[idx]}\n\n"
    
    return summary

# ========== DISPLAY RESULTS ==========
def display_results(text, source_name):
    total_sentences = len(nltk.sent_tokenize(text))
    
    # Slider
    st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        max_val = min(30, total_sentences)
        num_points = st.slider(
            "Summary sentences:",
            min_value=3,
            max_value=max_val,
            value=min(5, max_val),
            key=f"slider_{source_name}"
        )
    with col2:
        st.metric("Total", total_sentences)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Summary
    summary = generate_summary(text, num_points)
    st.markdown("## 📋 Summary")
    st.markdown(f"<div class='section-card'>{summary}</div>", unsafe_allow_html=True)
    
    # Statistics
    cols = st.columns(4)
    with cols[0]:
        st.metric("Characters", f"{len(text):,}")
    with cols[1]:
        st.metric("Words", f"{len(text.split()):,}")
    with cols[2]:
        st.metric("Sentences", f"{total_sentences:,}")
    with cols[3]:
        reduction = int((1 - num_points/total_sentences) * 100)
        st.metric("Reduced", f"{reduction}%")
    
    # Downloads
    st.markdown("### 📥 Downloads")
    cols = st.columns(3)
    
    with cols[0]:
        st.download_button("📄 Full Text", text, f"{source_name}_full.txt")
    
    with cols[1]:
        st.download_button("📝 Summary", summary, f"{source_name}_summary.txt")
    
    with cols[2]:
        audio = text_to_speech(summary)
        if audio:
            st.audio(audio, format='audio/mp3')
            st.download_button("🔊 Audio", audio, f"{source_name}_audio.mp3", "audio/mp3")
    
    # Keywords
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    keywords = Counter(words).most_common(10)
    if keywords:
        st.markdown("### 🏷️ Keywords")
        html = "<div>"
        for word, count in keywords[:8]:
            html += f"<span class='keyword-tag'>{word} ({count})</span> "
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

# ========== MAIN UI ==========
def main():
    tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🌐 URL/YouTube", "📝 Paste Text", "ℹ️ Help"])
    
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
            
            if st.button("🚀 Process", key="proc_file"):
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
                        with open(path, 'r', encoding='utf-8') as f:
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
        url = st.text_input("Enter URL", placeholder="https://example.com/article or YouTube link")
        
        if url and st.button("🌐 Fetch Content", key="fetch_url"):
            # Check if YouTube
            if 'youtube.com' in url or 'youtu.be' in url:
                with st.spinner("Processing YouTube video..."):
                    text, title, error = extract_youtube_content(url)
                    
                    if text:
                        st.success(f"✅ {title}")
                        st.info(f"📊 Extracted {len(text)} characters")
                        display_results(text, "youtube")
                    else:
                        st.markdown(f"<div class='warning-box'>⚠️ {error}</div>", unsafe_allow_html=True)
                        if "AssemblyAI Key" in error:
                            st.info("🔑 Add your AssemblyAI key in sidebar to transcribe this video")
            
            # Regular article URL
            elif validators.url(url):
                with st.spinner("Fetching article..."):
                    text, title = extract_article_from_url(url)
                    if text:
                        st.success(f"✅ {title}")
                        st.info(f"📊 Extracted {len(text)} characters")
                        display_results(text, "web")
                    else:
                        st.warning("⚠️ No readable content found")
            else:
                st.error("❌ Invalid URL")
    
    with tab3:
        text_input = st.text_area("Paste text", height=200)
        if text_input and st.button("📝 Summarize", key="summ_text"):
            if len(text_input) > 100:
                display_results(text_input, "pasted")
            else:
                st.warning("Text too short")
    
    with tab4:
        st.markdown("""
        <div class='section-card'>
            <h3>📌 How to Use</h3>
            <ol>
                <li><strong>Get AssemblyAI Key:</strong> Free at assemblyai.com</li>
                <li><strong>Upload:</strong> File, URL, or paste text</li>
                <li><strong>YouTube:</strong> Auto-detects and transcribes</li>
                <li><strong>Adjust:</strong> Use slider for summary length</li>
                <li><strong>Download:</strong> Text, summary, or audio</li>
            </ol>
            
            <h3>🎯 YouTube Support</h3>
            <ul>
                <li>✅ With captions: Instant text</li>
                <li>✅ Without captions: Auto-download + AssemblyAI</li>
                <li>✅ Telugu videos supported</li>
                <li>✅ No manual downloading needed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
