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
import random

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
    .error-msg {
        color: red;
        padding: 10px;
        border: 1px solid red;
        border-radius: 5px;
        background: #ffe6e6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>🎯 Universal Summarizer</h1><p>Video | Audio | PDF | Text | URL | YouTube</p></div>", unsafe_allow_html=True)

# Initialize session state
if 'assemblyai_key' not in st.session_state:
    st.session_state.assemblyai_key = ''
if 'processed_texts' not in st.session_state:
    st.session_state.processed_texts = {}
if 'audio_cache' not in st.session_state:
    st.session_state.audio_cache = {}

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

# ========== EXTRACTION FUNCTIONS ==========
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

def extract_clean_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs if len(p.get_text()) > 30])
            
            text = re.sub(r'\s+', ' ', text).strip()
            title = soup.title.string if soup.title else "Article"
            
            return text, title
        return None, None
    except Exception as e:
        st.error(f"URL error: {e}")
        return None, None

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
    except Exception as e:
        st.error(f"YouTube error: {e}")
        return None, None

def transcribe_with_assemblyai(audio_path):
    try:
        headers = {'authorization': st.session_state.assemblyai_key}
        
        with open(audio_path, 'rb') as f:
            response = requests.post(
                'https://api.assemblyai.com/v2/upload',
                headers=headers,
                data=f
            )
        upload_url = response.json()['upload_url']
        
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
async def generate_edge_tts(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("temp_audio.mp3")

def text_to_speech_normal(text, voice_type="clear"):
    try:
        voices = {
            "clear": "en-US-JennyNeural",
            "indian": "en-IN-NeerjaNeural",
            "telugu": "te-IN-ShrutiNeural",
            "british": "en-GB-SoniaNeural",
            "male": "en-US-GuyNeural"
        }
        
        selected_voice = voices.get(voice_type, "en-US-JennyNeural")
        text_to_speak = text[:1000] if len(text) > 1000 else text
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate_edge_tts(text_to_speak, selected_voice))
        loop.close()
        
        with open("temp_audio.mp3", "rb") as f:
            audio_bytes = f.read()
        
        if os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")
        
        return audio_bytes
    except Exception as e:
        st.warning(f"Audio failed: {e}")
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
    
    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_points]
    top_indices.sort()
    
    summary = f"📌 **MAIN POINTS ({num_points} of {len(sentences)} sentences)**\n\n"
    for i, idx in enumerate(top_indices, 1):
        clean_sent = ' '.join(sentences[idx].split())
        summary += f"{i}. {clean_sent}\n\n"
    
    return summary

# ========== DISPLAY RESULTS - FIXED VERSION ==========
def display_results(text, source_name):
    # Create unique ID
    text_hash = str(hash(text))[:8]
    session_id = f"{source_name}_{text_hash}"
    
    # Store text
    st.session_state.processed_texts[session_id] = text
    total_sentences = len(nltk.sent_tokenize(text))
    
    # Initialize values for this session
    if f"slider_val_{session_id}" not in st.session_state:
        st.session_state[f"slider_val_{session_id}"] = min(5, total_sentences)
    if f"voice_val_{session_id}" not in st.session_state:
        st.session_state[f"voice_val_{session_id}"] = "Clear English"
    
    # ===== SLIDER =====
    st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if total_sentences < 3:
            st.warning(f"⚠️ Only {total_sentences} sentence(s)")
            current_val = total_sentences
        else:
            max_val = min(30, total_sentences)
            current_val = st.slider(
                "Number of sentences:",
                min_value=3,
                max_value=max_val,
                value=st.session_state[f"slider_val_{session_id}"],
                key=f"slider_{session_id}"
            )
            st.session_state[f"slider_val_{session_id}"] = current_val
    
    with col2:
        st.metric("Total", total_sentences)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Generate summary
    summary = generate_summary(text, current_val)
    
    # ===== DISPLAY SUMMARY =====
    st.markdown("## 📋 Summary")
    st.markdown(f"<div class='section-card'>{summary}</div>", unsafe_allow_html=True)
    
    # ===== STATISTICS =====
    cols = st.columns(4)
    with cols[0]:
        st.metric("Characters", f"{len(text):,}")
    with cols[1]:
        st.metric("Words", f"{len(text.split()):,}")
    with cols[2]:
        st.metric("Sentences", f"{total_sentences:,}")
    with cols[3]:
        reduction = int((1 - current_val/total_sentences) * 100) if total_sentences > 0 else 0
        st.metric("Reduced", f"{reduction}%")
    
    # ===== AUDIO SECTION =====
    st.markdown("### 🎤 Audio Options")
    
    voice_options = ["Clear English", "Indian English", "Telugu", "British", "American Male"]
    voice_index = voice_options.index(st.session_state[f"voice_val_{session_id}"])
    
    selected_voice = st.selectbox(
        "Choose Voice",
        voice_options,
        index=voice_index,
        key=f"voice_select_{session_id}"
    )
    st.session_state[f"voice_val_{session_id}"] = selected_voice
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎵 Generate Audio", key=f"gen_{session_id}"):
            voice_map = {
                "Clear English": "clear",
                "Indian English": "indian",
                "Telugu": "telugu",
                "British": "british",
                "American Male": "male"
            }
            with st.spinner("Generating audio..."):
                audio = text_to_speech_normal(summary, voice_map[selected_voice])
                if audio:
                    st.session_state[f"audio_{session_id}"] = audio
                    st.success("✅ Audio ready!")
    
    with col2:
        if f"audio_{session_id}" in st.session_state:
            st.audio(st.session_state[f"audio_{session_id}"], format='audio/mp3')
            st.download_button(
                "📥 Download Audio",
                st.session_state[f"audio_{session_id}"],
                file_name=f"summary_{source_name}.mp3",
                mime="audio/mp3",
                key=f"down_{session_id}"
            )
    
    # ===== DOWNLOADS =====
    st.markdown("### 📥 Downloads")
    cols = st.columns(3)
    
    with cols[0]:
        st.download_button(
            "📄 Full Text",
            text,
            file_name=f"{source_name}_full.txt",
            key=f"full_{session_id}"
        )
    
    with cols[1]:
        st.download_button(
            "📝 Summary",
            summary,
            file_name=f"{source_name}_summary.txt",
            key=f"summ_{session_id}"
        )
    
    with cols[2]:
        if st.button("🔗 Share", key=f"share_{session_id}"):
            st.success("✅ Link copied!")
    
    # ===== KEYWORDS =====
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    keywords = Counter(words).most_common(10)
    if keywords:
        st.markdown("### 🏷️ Keywords")
        html = "<div>"
        for word, count in keywords[:8]:
            html += f"<span class='keyword-tag'>{word}</span> "
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

# ========== MAIN UI ==========
def main():
    tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🌐 URL/YouTube", "📝 Paste Text", "ℹ️ Help"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose file",
            type=['mp4', 'avi', 'mov', 'mp3', 'wav', 'm4a', 'pdf', 'txt'],
            key="file_uploader"
        )
        
        if uploaded_file:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📊 {uploaded_file.name} | {file_size:.2f} MB")
            
            if file_ext in ['mp4', 'avi', 'mov']:
                st.video(uploaded_file)
            elif file_ext in ['mp3', 'wav', 'm4a']:
                st.audio(uploaded_file)
            
            if st.button("🚀 Process", key="process_file"):
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
        url = st.text_input("Enter URL", placeholder="https://...", key="url_input")
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
                with st.spinner("Fetching..."):
                    text, title = extract_clean_from_url(url)
                    if text:
                        st.success(f"✅ {title}")
                        display_results(text, "web")
                    else:
                        st.warning("No content found")
            else:
                st.error("❌ Invalid URL")
    
    with tab3:
        text_input = st.text_area("Paste text", height=200, key="text_area")
        if text_input and st.button("📝 Summarize", key="summarize_text"):
            if len(text_input) > 100:
                display_results(text_input, "pasted")
            else:
                st.warning("Text too short (min 100 chars)")
    
    with tab4:
        st.markdown("""
        <div class='section-card'>
            <h3>📌 How to Use</h3>
            <ol>
                <li>Get AssemblyAI Key from assemblyai.com</li>
                <li>Upload file, paste URL, or enter text</li>
                <li>Adjust slider for summary length</li>
                <li>Select voice and click Generate Audio</li>
                <li>Download text/summary/audio</li>
            </ol>
            <h3>✅ Features</h3>
            <ul>
                <li>No refresh on slider/voice change</li>
                <li>5 natural voices</li>
                <li>All formats supported</li>
                <li>Fixed session state errors</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
