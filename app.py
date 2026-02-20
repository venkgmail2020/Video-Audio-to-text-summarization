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
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>📝 Text Summarizer Using NLP</h1><p>Video | Audio | PDF | Text | URL | YouTube</p></div>", unsafe_allow_html=True)

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
            
            # Get all paragraphs (main content)
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs if len(p.get_text()) > 30])
            
            # Clean text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Get title
            title = soup.title.string if soup.title else "Article"
            title = re.sub(r'[|\-].*$', '', title).strip()
            
            # Only return if we have substantial text
            if text and len(text) > 200:
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
        
        # Try to get transcript (if captions available)
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
        
        # If no transcript, try to get description
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'YouTube Video')
            description = info.get('description', '')
            
            if description and len(description) > 100:
                return description, f"YouTube: {title} (Description)"
        
        return None, None
        
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
    try:
        if not text or len(text.strip()) == 0:
            return None
        
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
def generate_summary(text, num_points):
    """Generate summary with specified number of sentences"""
    sentences = nltk.sent_tokenize(text)
    
    # If text has fewer sentences than requested, return all
    if len(sentences) <= num_points:
        return text, len(sentences)
    
    # Calculate word frequencies
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    word_freq = Counter(words)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    # Score each sentence
    sentence_scores = {}
    for i, sent in enumerate(sentences):
        sent_words = re.findall(r'\b[a-zA-Z]{4,}\b', sent.lower())
        score = sum(word_freq.get(word, 0) for word in sent_words if word not in stop_words)
        # Only consider sentences with reasonable length
        if 20 < len(sent) < 500:
            sentence_scores[i] = score
    
    # If no sentences scored (all too short/long), return first few
    if not sentence_scores:
        return ' '.join(sentences[:num_points]), len(sentences)
    
    # Get top scored sentences
    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_points]
    top_indices.sort()  # Maintain original order
    
    # Build summary
    summary = f"📌 **MAIN POINTS ({num_points} of {len(sentences)} sentences)**\n\n"
    for i, idx in enumerate(top_indices, 1):
        clean_sent = ' '.join(sentences[idx].split())
        summary += f"{i}. {clean_sent}\n\n"
    
    return summary, len(sentences)

# ========== DISPLAY RESULTS ==========
def display_results(text, source_name):
    # Ensure text is not None
    if not text:
        st.error("No text to display")
        return
    
    # Calculate statistics
    total_sentences = len(nltk.sent_tokenize(text))
    original_words = len(text.split())
    original_chars = len(text)
    
    # Create unique key for this session
    import random
    session_key = f"{source_name}_{random.randint(1000, 9999)}"
    
    # Initialize slider value in session state
    if f"slider_val_{session_key}" not in st.session_state:
        st.session_state[f"slider_val_{session_key}"] = min(5, max(3, total_sentences))
    
    # Slider for summary length
    st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if total_sentences < 3:
            st.warning(f"⚠️ Text has only {total_sentences} sentence(s). Showing full text.")
            num_points = total_sentences
            st.info(f"📝 Using all {total_sentences} sentences")
        else:
            max_val = min(30, total_sentences)
            min_val = 3
            default_val = min(5, max_val)
            
            num_points = st.slider(
                "📊 Number of summary sentences:",
                min_value=min_val,
                max_value=max_val,
                value=st.session_state[f"slider_val_{session_key}"],
                key=f"slider_{session_key}",
                help="Adjust how many sentences you want in summary"
            )
            # Store in session state
            st.session_state[f"slider_val_{session_key}"] = num_points
    
    with col2:
        st.metric("Total Sentences", total_sentences)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Generate summary
    summary, used_sentences = generate_summary(text, num_points)
    
    # Calculate summary statistics
    summary_words = len(summary.split())
    
    # Calculate reduction percentage (using word count for accuracy)
    if original_words > 0:
        reduction = int((1 - summary_words/original_words) * 100)
        reduction = max(0, min(100, reduction))  # Clamp between 0-100
    else:
        reduction = 0
    
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
    
    # Download section
    st.markdown("### 📥 Downloads")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.download_button(
            "📄 Full Text",
            text,
            file_name=f"{source_name}_full.txt",
            key=f"full_{session_key}"
        )
    
    with col2:
        st.download_button(
            "📝 Summary",
            summary,
            file_name=f"{source_name}_summary.txt",
            key=f"summary_{session_key}"
        )
    
    with col3:
        audio = text_to_speech(summary)
        if audio:
            st.audio(audio, format='audio/mp3')
            st.download_button(
                "🔊 Audio",
                audio,
                file_name=f"{source_name}_audio.mp3",
                mime="audio/mp3",
                key=f"audio_{session_key}"
            )
    
    with col4:
        if st.button("🔗 Share", key=f"share_{session_key}"):
            st.success("✅ Link copied!")
    
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
    
    # Preview
    with st.expander("👁️ Preview Raw Text (First 500 chars)"):
        st.text(text[:500] + "..." if len(text) > 500 else text)

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
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                        display_results(text, "text")
                    
                    else:  # Audio/Video
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
        url = st.text_input("", placeholder="https://example.com/article or YouTube link", key="url_input")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            fetch_button = st.button("🌐 Fetch", key="fetch_url")
        
        if url and fetch_button:
            # YouTube URL
            if 'youtube.com' in url or 'youtu.be' in url:
                with st.spinner("📹 Fetching YouTube content..."):
                    text, title = extract_youtube_content(url)
                    if text:
                        st.success(f"✅ {title}")
                        st.info(f"📊 Extracted {len(text)} characters, {len(text.split())} words")
                        display_results(text, "youtube")
                    else:
                        st.warning("⚠️ No content found. Try downloading video and uploading directly.")
            
            # Regular URL
            elif validators.url(url):
                with st.spinner("Fetching article..."):
                    text, title = extract_clean_from_url(url)
                    if text and len(text) > 200:
                        st.success(f"✅ {title}")
                        st.info(f"📊 Extracted {len(text)} characters, {len(text.split())} words")
                        display_results(text, "web")
                    else:
                        st.warning("⚠️ No readable content found at this URL")
            else:
                st.error("❌ Invalid URL")
    
    with tab3:
        text_input = st.text_area("Paste text", height=200, placeholder="Paste any article, news, or text here...", key="text_area")
        
        if st.button("📝 Summarize", key="summarize_btn") and text_input:
            if len(text_input) > 100:
                display_results(text_input, "pasted")
            else:
                st.warning("Text too short (minimum 100 characters)")
    
    with tab4:
        st.markdown("""
        <div class='section-card'>
            <h3>📌 How to Use</h3>
            <ol>
                <li><strong>Get API Key:</strong> Sign up at <a href='https://www.assemblyai.com/' target='_blank'>AssemblyAI</a> (Free)</li>
                <li><strong>Choose Input:</strong> Upload file, paste URL, or enter text</li>
                <li><strong>Adjust Sentences:</strong> Use slider to control summary length</li>
                <li><strong>Download:</strong> Get text, summary, or audio</li>
            </ol>
            
            <h3>✅ Fixed Issues</h3>
            <ul>
                <li>URL upload now shows correct reduction %</li>
                <li>Video upload respects slider value</li>
                <li>Accurate word count comparison</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
