from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/audiomind"
    
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # LLM Provider
    ai_provider: str = "openai"  # openai | ollama

    # Ollama (local LLM)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b-instruct"
    ollama_timeout_seconds: int = 120
    
    # Hugging Face
    huggingface_token: str = ""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Storage
    audio_storage_path: str = "./storage/audio"
    temp_storage_path: str = "./storage/temp"
    
    # Model Settings
    whisper_model: str = "base"
    device: str = "cpu"  # or cuda
    enable_speaker_diarization: bool = False
    lazy_load_models: bool = True
    
    # Processing
    max_chunk_duration: int = 30
    vad_threshold: float = 0.5
    
    class Config:
        env_file = str(ENV_FILE)
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
