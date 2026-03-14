# AudioMind AI Service

Python FastAPI service for audio processing, speech recognition, speaker diarization, and AI analysis.

## Features

- 🎤 Speech-to-Text (Whisper)
- 👥 Speaker Diarization (pyannote.audio)
- 🔊 Voice Activity Detection (Silero VAD)
- 🤖 AI Meeting Analysis (GPT-4)
- 📊 Structured Meeting Notes

## Architecture

```
AI Service (Port 8000)
│
├── Audio Processing
│   ├── VAD
│   └── Audio Segmentation
│
├── Speech Recognition
│   └── Whisper
│
├── Speaker Diarization
│   └── pyannote.audio
│
├── AI Analysis
│   └── OpenAI GPT-4
│
└── Database
    └── PostgreSQL
```

## Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run database migrations:
```bash
alembic upgrade head
```

5. Start server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /api/process` - Process audio file
- `GET /api/meeting/{meeting_id}/transcript` - Get transcript
- `GET /api/meeting/{meeting_id}/analysis` - Get AI analysis
- `GET /health` - Health check

## Requirements

- Python 3.9+
- PostgreSQL 14+
- CUDA (optional, for GPU acceleration)
- FFmpeg

## Configuration

See `.env.example` for all configuration options.

## Models

- **Whisper**: large-v3
- **Speaker Diarization**: pyannote/speaker-diarization-3.1
- **VAD**: silero-vad
- **LLM**: GPT-4/GPT-4o
