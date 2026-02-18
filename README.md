# 🎥 Video/Audio/PDF Summarizer

A powerful tool that converts videos, audio files, and PDFs into text and generates concise summaries.

## ✨ Features

- 🎬 **Video to Text** - Upload MP4, AVI, MOV files
- 🎵 **Audio Transcription** - Support for MP3, WAV files  
- 📄 **PDF Extraction** - Extract text from PDF documents
- 📝 **Smart Summarization** - Get 3-sentence summaries
- 🔑 **API Ready** - Works with AssemblyAI
- 📥 **Download Options** - Save transcripts and summaries

## 🚀 Live Demo

[Click here to try the app](https://video-audio-to-text-summarization.streamlit.app)

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Transcription**: AssemblyAI API
- **Summarization**: NLTK + Sumy (LexRank)
- **PDF Processing**: PyPDF2

## 📋 How to Use

1. Get a free API key from [AssemblyAI](https://www.assemblyai.com/)
2. Upload your file (video, audio, PDF, or text)
3. Click "Process" and wait for results
4. Download transcript or summary

## ⚙️ Installation (Local)

```bash
# Clone repository
git clone https://github.com/venkgmail2020/Video-Audio-to-text-summarization.git

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
