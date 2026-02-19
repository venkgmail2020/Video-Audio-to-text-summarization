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
import edge_tts
import asyncio

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
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>🎯 Universal Summarizer</h1><p>Video | Audio | PDF | Text | URL | YouTube</p></div>", unsafe_allow_html=True)

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
    st.markdown("▶️ YouTube")

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

# ========== CLEAN URL EXTRACTION ==========
def extract_clean_from_url(url):
    """Extract ONLY main content, no ads/footer/copyright"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 
                               'form', 'button', 'iframe', 'meta', 'link']):
                element.decompose()
            
            # Try to find main content
            main_content = None
            for selector in ['main', 'article', '[role="main"]', '.content', '.post-content', 
                           '.article-content', '#content', '.main-content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if main_content:
                text = main_content.get_text()
            else:
                # Get all paragraphs (usually main content)
                paragraphs = soup.find_all('p')
                text = ' '.join([p.get_text() for p in paragraphs if len(p.get_text()) > 30])
            
            # Clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk and len(chunk) > 20)
            
            # Remove common footer/copyright patterns
            footer_patterns = [
                r'©.*?\d{4}.*?rights reserved',
                r'Privacy.*?Policy',
                r'Terms.*?of.*?Service',
                r'Contact.*?Us',
                r'About.*?Press',
                r'Copyright.*?\d{4}',
                r'All.*?Rights.*?Reserved',
                r'Subscribe.*?Newsletter',
                r'Follow.*?us on',
                r'Share this article'
            ]
            
            for pattern in footer_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Get title
            title = soup.title.string if soup.title else "Article"
            title = re.sub(r'[|\-].*$', '', title).strip()
            
            return text, title
        return None, None
    except Exception as e:
        st.error(f"URL extraction error: {e}")
        return None, None

# ========== YOUTUBE EXTRACTION ==========
def extract_youtube_content(url):
    """Extract actual content from YouTube video"""
    try:
        # Extract video ID
        video_id = None
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('watch?v=')[-1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[-1].split('?')[0]
        
        if not video_id:
            return None, None
        
        # METHOD 1: Try to get transcript (if captions available)
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try Telugu first, then English, then Hindi
            for lang in ['te', 'en', 'hi']:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    transcript_data = transcript.fetch()
                    
                    # Convert to text
                    full_text = ' '.join([item['text'] for item in transcript_data])
                    
                    # Get video title
                    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                        info = ydl.extract_info(url, download=False)
                        title = info.get('title', 'YouTube Video')
                    
                    return full_text, f"YouTube: {title} (Captions)"
                except:
                    continue
        except:
            pass
        
        # METHOD 2: If no transcript, try to get description as fallback
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'YouTube Video')
            description = info.get('description', '')
            
            if description and len(description) > 100:
                return description, f"YouTube: {title} (Description)"
        
        return None, "No captions or description available"
        
    except Exception as e:
        st.error(f"YouTube extraction error: {e}")
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

# ========== NORMAL VOICE TEXT TO SPEECH (ADDED) ==========
def text_to_speech_normal(text, voice_type="clear"):
    """Convert text to normal, clear voice using Edge TTS"""
    try:
        # Clear, natural voice options
        voices = {
            "clear": "en-US-JennyNeural",      # Clear American female
            "indian": "en-IN-NeerjaNeural",    # Indian female
            "telugu": "te-IN-ShrutiNeural",    # Telugu female
            "british": "en-GB-SoniaNeural",    # British female
            "male": "en-US-GuyNeural"          # American male
        }
        
        selected_voice = voices.get(voice_type, "en-US-JennyNeural")
        text_to_speak = text[:1000] if len(text) > 1000 else text
        
        async def generate():
            communicate = edge_tts.Communicate(text_to_speak, selected_voice)
            await communicate.save("normal_voice.mp3")
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate())
        loop.close()
        
        # Read audio file
        with open("normal_voice.mp3", "rb") as f:
            audio_bytes = f.read()
        
        # Clean up
        if os.path.exists("normal_voice.mp3"):
            os.remove("normal_voice.mp3")
        
        return audio_bytes
        
    except Exception as e:
        st.warning(f"⚠️ Audio generation failed: {e}")
        return None

# ========== GENERATE SUMMARY ==========
def generate_summary(text, num_points=5):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= num_points:
        return text, len(sentences)
    
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
        clean_sent = ' '.join(sentences[idx].split())
        summary += f"{i}. {clean_sent}\n\n"
    
    return summary, len(sentences)

# ========== DISPLAY RESULTS WITH NORMAL VOICE ==========
def display_results(text, source_name):
    # Sentence count
    total_sentences = len(nltk.sent_tokenize(text))
    
    # Slider for summary length - FIXED VERSION
    st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        # Handle very short texts (like YouTube descriptions)
        if total_sentences < 3:
            st.warning(f"⚠️ Text has only {total_sentences} sentence(s). Showing full text.")
            num_summary_sentences = total_sentences
            # Disable slider by showing info instead
            st.info(f"📝 Using all {total_sentences} sentences")
        else:
            # Calculate safe values
            max_slider = min(30, total_sentences)
            min_slider = 3
            default_val = min(5, max_slider)
            
            num_summary_sentences = st.slider(
                "📊 Number of summary sentences:",
                min_value=min_slider,
                max_value=max_slider,
                value=default_val,
                help="Adjust how many sentences you want in summary"
            )
    with col2:
        st.metric("Total Sentences", total_sentences)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Generate summary with selected length
    if total_sentences < 3:
        # For very short texts, just show the text itself
        summary = f"📌 **FULL TEXT ({total_sentences} sentences)**\n\n{text}"
        used_sentences = total_sentences
    else:
        summary, used_sentences = generate_summary(text, num_summary_sentences)
    
    st.markdown("## 📋 Summary")
    st.markdown(f"<div class='section-card'>{summary}</div>", unsafe_allow_html=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Characters", f"{len(text):,}")
    with col2:
        st.metric("Words", f"{len(text.split()):,}")
    with col3:
        st.metric("Sentences", f"{total_sentences:,}")
    with col4:
        if total_sentences > 0:
            reduction = int((1 - used_sentences/total_sentences) * 100)
        else:
            reduction = 0
        st.metric("Reduced", f"{reduction}%")
    
    # Download section with NORMAL VOICE
    st.markdown("### 📥 Downloads")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.download_button("📄 Full Text", text, f"{source_name}_full.txt")
    
    with col2:
        st.download_button("📝 Summary", summary, f"{source_name}_summary.txt")
    
    with col3:
        # Voice selector for normal voice
        voice_option = st.selectbox(
            "Voice Style",
            ["Clear English", "Indian English", "Telugu", "British", "American Male"],
            key=f"voice_{source_name}"
        )
        
        voice_map = {
            "Clear English": "clear",
            "Indian English": "indian",
            "Telugu": "telugu",
            "British": "british",
            "American Male": "male"
        }
        
        # Generate normal voice audio
        audio = text_to_speech_normal(summary, voice_map[voice_option])
        if audio:
            st.audio(audio, format='audio/mp3')
            st.download_button(
                "🔊 Download Audio",
                audio,
                file_name="summary_audio.mp3",
                mime="audio/mp3",
                use_container_width=True
            )
        else:
            st.warning("Audio not available")
    
    with col4:
        if st.button("🔗 Share"):
            st.success("✅ Link copied!")
    
    # Keywords
    keywords = Counter(re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())).most_common(10)
    st.markdown("### 🏷️ Keywords")
    html = "<div>"
    for word, count in keywords[:8]:
        html += f"<span class='keyword-tag'>{word} ({count})</span> "
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    
    # Show preview
    with st.expander("👁️ Preview Raw Text (First 500 chars)"):
        st.text(text[:500] + "..." if len(text) > 500 else text)

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
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                        display_results(text, "text")
                    else:
                        if not st.session_state.assemblyai_key:
                            st.error("❌ AssemblyAI Key required for video/audio")
                        else:
                            text = transcribe_with_assemblyai(path)
                            if text:
                                st.success(f"✅ Transcribed: {len(text)} chars")
                                display_results(text, "media")
                    
                    os.unlink(path)
    
    with tab2:
        st.markdown("### Enter URL (Article or YouTube)")
        url = st.text_input("", placeholder="https://example.com/article or https://youtube.com/watch?v=...")
        
        if url and st.button("🌐 Fetch Content"):
            # Check if it's a YouTube URL
            if 'youtube.com' in url or 'youtu.be' in url:
                with st.spinner("📹 Fetching YouTube content..."):
                    text, title = extract_youtube_content(url)
                    if text:
                        st.success(f"✅ {title}")
                        st.info(f"📊 Extracted {len(text)} characters, {len(text.split())} words")
                        display_results(text, "youtube")
                    else:
                        st.warning("⚠️ No transcript available. Try downloading the video and uploading directly.")
            elif validators.url(url):
                with st.spinner("Fetching clean content..."):
                    text, title = extract_clean_from_url(url)
                    if text and len(text) > 100:
                        st.success(f"✅ Fetched: {title}")
                        display_results(text, "web")
                    else:
                        st.warning("⚠️ No readable content found at this URL")
            else:
                st.error("❌ Invalid URL")
    
    with tab3:
        text_input = st.text_area("Paste text", height=200, placeholder="Paste any article, news, or text here...")
        
        if text_input and st.button("📝 Summarize"):
            if len(text_input) > 100:
                display_results(text_input, "pasted")
            else:
                st.warning("Text too short (minimum 100 characters)")
    
    with tab4:
        st.markdown("""
        <div class='section-card'>
            <h3>📌 How to Use</h3>
            <ol>
                <li><strong>Get API Key:</strong> Sign up at <a href='https://www.assemblyai.com/' target='_blank'>AssemblyAI</a> (Free - 10 hours/month)</li>
                <li><strong>Choose Input:</strong> Upload file, paste URL, or enter text</li>
                <li><strong>Adjust Sentences:</strong> Use slider to control summary length</li>
                <li><strong>Download:</strong> Get text, summary, or audio with normal voice</li>
            </ol>
            
            <h3>✨ Features</h3>
            <ul>
                <li>✅ <strong>Normal Voice</strong> - Clear, natural audio</li>
                <li>✅ <strong>YouTube Support</strong> - Extracts captions/description</li>
                <li>✅ <strong>Clean URLs</strong> - No footer/copyright text</li>
                <li>✅ <strong>Sentence slider</strong> - Control summary length</li>
                <li>✅ <strong>Fixed slider error</strong> - Handles short texts</li>
                <li>✅ <strong>All formats</strong> - Video, Audio, PDF, TXT, URL</li>
            </ul>
            
            <h3>🎤 Voice Options</h3>
            <ul>
                <li><strong>Clear English</strong> - American female, crystal clear</li>
                <li><strong>Indian English</strong> - Indian female, natural accent</li>
                <li><strong>Telugu</strong> - Telugu language female voice</li>
                <li><strong>British</strong> - British female, elegant</li>
                <li><strong>American Male</strong> - Deep male voice</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ========== CALL MAIN FUNCTION ==========
if __name__ == "__main__":
    main()
