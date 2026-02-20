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
from sumy.parsers.plaintext import PlaintextParser  # ✅ Fixed: lowercase 't'
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

st.set_page_config(page_title="Text Summarizer Using NLP", page_icon="📝", layout="wide")

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
    .update-button {
        background: #28a745;
        color: white;
    }
    .fixed-info {
        background: #e7f3ff;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #0066cc;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>📝 Text Summarizer Using NLP</h1><p>Video | Audio | PDF | Text | URL | YouTube</p></div>", unsafe_allow_html=True)

# Initialize session state
if 'assemblyai_key' not in st.session_state:
    st.session_state.assemblyai_key = ''
if 'current_text' not in st.session_state:
    st.session_state.current_text = ''
if 'current_summary' not in st.session_state:
    st.session_state.current_summary = ''
if 'slider_value' not in st.session_state:
    st.session_state.slider_value = 5
if 'original_words' not in st.session_state:
    st.session_state.original_words = 0
if 'reduction_percent' not in st.session_state:
    st.session_state.reduction_percent = 0

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
    st.markdown("▶️ YouTube")
    
    st.markdown("---")
    st.markdown("### ✅ Fixed Issues")
    st.markdown("• URL shows correct % now")
    st.markdown("• No refresh on slider")
    st.markdown("• Word count based reduction")
    st.markdown("• sumy import fixed")

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
        return text
    except:
        return None

# ========== URL EXTRACTION (FIXED) ==========
def extract_from_url(url):
    """Extract text from URL - FIXED version"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Get all paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs if len(p.get_text()) > 30])
            
            # Clean text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Get title
            title = soup.title.string if soup.title else "Article"
            
            if text and len(text) > 200:
                return text, title
        return None, None
    except Exception as e:
        st.error(f"URL error: {e}")
        return None, None

# ========== YOUTUBE EXTRACTION ==========
def extract_youtube_content(url):
    try:
        video_id = None
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('watch?v=')[-1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[-1].split('?')[0]
        
        if not video_id:
            return None, None
        
        # Try captions
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for lang in ['te', 'en', 'hi']:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    transcript_data = transcript.fetch()
                    full_text = ' '.join([item['text'] for item in transcript_data])
                    
                    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                        info = ydl.extract_info(url, download=False)
                        title = info.get('title', 'YouTube Video')
                    
                    return full_text, f"YouTube: {title}"
                except:
                    continue
        except:
            pass
        
        # Fallback to description
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'YouTube Video')
            description = info.get('description', '')
            
            if description:
                return description, f"YouTube: {title}"
        
        return None, None
    except:
        return None, None

# ========== ASSEMBLYAI TRANSCRIPTION ==========
def transcribe_with_assemblyai(audio_path, original_filename):
    try:
        headers = {'authorization': st.session_state.assemblyai_key}
        
        # Clean filename
        import re
        name = original_filename.rsplit('.', 1)[0]
        ext = original_filename.rsplit('.', 1)[1] if '.' in original_filename else 'mp4'
        clean_name = re.sub(r'[^a-zA-Z\s]', '', name)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        if not clean_name:
            clean_name = "video_file"
        clean_filename = f"{clean_name}.{ext}"
        
        # Upload file
        with open(audio_path, 'rb') as f:
            response = requests.post(
                'https://api.assemblyai.com/v2/upload',
                headers=headers,
                files={'file': (clean_filename, f, 'audio/mp4')}
            )
        
        if response.status_code != 200:
            st.error(f"Upload failed: {response.status_code}")
            return None
            
        upload_url = response.json()['upload_url']
        
        # Request transcription
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
        for i in range(60):
            time.sleep(2)
            progress.progress(min(i * 2, 95))
            
            response = requests.get(
                f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                headers=headers
            )
            result = response.json()
            
            if result['status'] == 'completed':
                return result.get('text', '')
            elif result['status'] == 'error':
                return None
        
        return None
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

# ========== TEXT TO SPEECH ==========
def text_to_speech(text):
    try:
        if not text:
            return None
        text_for_audio = text[:1000] if len(text) > 1000 else text
        tts = gTTS(text=text_for_audio, lang='en', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except:
        return None

# ========== GENERATE SUMMARY ==========
def generate_summary(text, num_points):
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
    
    if not sentence_scores:
        return ' '.join(sentences[:num_points])
    
    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_points]
    top_indices.sort()
    
    summary = f"📌 **MAIN POINTS ({num_points} of {len(sentences)} sentences)**\n\n"
    for i, idx in enumerate(top_indices, 1):
        summary += f"{i}. {sentences[idx]}\n\n"
    
    return summary

# ========== DISPLAY RESULTS ==========
def display_results(text, source_name):
    if not text or len(text.strip()) == 0:
        st.error("No text to display")
        return
    
    # Store in session state
    st.session_state.current_text = text
    
    # Calculate statistics
    total_sentences = len(nltk.sent_tokenize(text))
    original_words = len(text.split())
    original_chars = len(text)
    
    # Store original words for reduction calculation
    st.session_state.original_words = original_words
    
    # Show info about source
    st.markdown(f"<div class='fixed-info'>📊 Processing: {source_name} | Total Words: {original_words}</div>", unsafe_allow_html=True)
    
    # Slider with NO AUTO REFRESH (using session state)
    st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if total_sentences < 3:
            st.warning(f"⚠️ Only {total_sentences} sentence(s)")
            num_points = total_sentences
        else:
            max_val = min(30, total_sentences)
            num_points = st.slider(
                "Number of summary sentences:",
                min_value=3,
                max_value=max_val,
                value=st.session_state.slider_value,
                key="main_slider"
            )
    
    with col2:
        st.metric("Total", total_sentences)
    
    with col3:
        # Update button - only when clicked
        if st.button("🔄 Update Summary", key="update_btn"):
            st.session_state.slider_value = num_points
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Use slider value from session state
    current_points = st.session_state.slider_value
    
    # Generate summary
    if total_sentences < 3:
        summary = text
        summary_words = original_words
    else:
        summary = generate_summary(text, current_points)
        summary_words = len(summary.split())
    
    # Store summary
    st.session_state.current_summary = summary
    
    # Calculate reduction (USING WORDS - CORRECT for URL also)
    if original_words > 0:
        reduction = int((1 - summary_words/original_words) * 100)
        reduction = max(0, min(100, reduction))
    else:
        reduction = 0
    
    # Store reduction percent
    st.session_state.reduction_percent = reduction
    
    # Display summary
    st.markdown("## 📋 Summary")
    st.markdown(f"<div class='section-card'>{summary}</div>", unsafe_allow_html=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Characters", f"{original_chars:,}")
    with col2:
        st.metric("Words", f"{original_words:,}")
    with col3:
        st.metric("Sentences", f"{total_sentences:,}")
    with col4:
        st.metric("Reduced", f"{reduction}%")
    
    # Downloads
    st.markdown("### 📥 Downloads")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button("📄 Full Text", text, f"{source_name}_full.txt", key=f"full_{source_name}")
    
    with col2:
        st.download_button("📝 Summary", summary, f"{source_name}_summary.txt", key=f"summ_{source_name}")
    
    with col3:
        audio = text_to_speech(summary)
        if audio:
            st.audio(audio, format='audio/mp3')
            st.download_button("🔊 Audio", audio, f"{source_name}_audio.mp3", "audio/mp3", key=f"audio_{source_name}")
    
    # Keywords
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    if words:
        keywords = Counter(words).most_common(10)
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
                        text = extract_pdf_text(path)
                        if text:
                            st.success("✅ Extracted PDF")
                            display_results(text, "pdf")
                    elif file_ext == 'txt':
                        with open(path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        display_results(text, "text")
                    else:
                        if not st.session_state.assemblyai_key:
                            st.error("❌ AssemblyAI Key required")
                        else:
                            text = transcribe_with_assemblyai(path, uploaded_file.name)
                            if text:
                                st.success(f"✅ Transcribed: {len(text)} chars")
                                display_results(text, "media")
                    
                    os.unlink(path)
    
    with tab2:
        url = st.text_input("Enter URL", placeholder="https://example.com/article or YouTube link")
        
        if url and st.button("🌐 Fetch", key="fetch_url"):
            if 'youtube.com' in url or 'youtu.be' in url:
                with st.spinner("Fetching YouTube..."):
                    text, title = extract_youtube_content(url)
                    if text:
                        st.success(f"✅ {title}")
                        display_results(text, "youtube")
                    else:
                        st.warning("No content found")
            elif validators.url(url):
                with st.spinner("Fetching article..."):
                    text, title = extract_from_url(url)
                    if text:
                        st.success(f"✅ {title}")
                        display_results(text, "web")
                    else:
                        st.warning("No content found")
            else:
                st.error("Invalid URL")
    
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
                <li>Get AssemblyAI Key from assemblyai.com</li>
                <li>Upload file or paste URL</li>
                <li>Adjust slider for summary length</li>
                <li>Click <strong>"Update Summary"</strong> to apply changes</li>
                <li>Download text/summary/audio</li>
            </ol>
            
            <h3>✅ FIXED</h3>
            <ul>
                <li>✅ <strong>URL shows correct %</strong> - Word count based reduction</li>
                <li>✅ <strong>No refresh on slider</strong> - Click Update button to apply</li>
                <li>✅ <strong>sumy import fixed</strong> - PlaintextParser (lowercase t)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
