import whisper
import torch
from typing import List, Dict
from loguru import logger
import numpy as np

from app.ffmpeg_utils import ensure_ffmpeg_on_path


class SpeechRecognizer:
    """
    Speech-to-text using OpenAI Whisper
    """
    
    def __init__(self, model_name: str = "large-v3", device: str = "cpu"):
        """
        Initialize Whisper model
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large, large-v3)
            device: Device to run model on (cpu, cuda)
        """
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading Whisper model: {model_name} on {device}")
        self.model = whisper.load_model(model_name, device=device)
        logger.info("Whisper model loaded successfully")
    
    def transcribe(
        self, 
        audio_path: str,
        language: str = None,
        initial_prompt: str = None,
        temperature: float = 0.0,
        beam_size: int = 5,
        best_of: int = 5,
        condition_on_previous_text: bool = True,
    ) -> Dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'vi'). None for auto-detect
            
        Returns:
            Dictionary with transcription results
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            ensure_ffmpeg_on_path()
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                word_timestamps=True,
                initial_prompt=initial_prompt,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
                condition_on_previous_text=condition_on_previous_text,
                verbose=False
            )
            
            logger.info(f"Transcription completed. Detected language: {result.get('language', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    def transcribe_segment(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        language: str = None
    ) -> Dict:
        """
        Transcribe audio segment
        
        Args:
            audio: Audio numpy array
            sr: Sample rate
            language: Language code
            
        Returns:
            Transcription result
        """
        try:
            # Whisper expects float32 audio normalized to [-1, 1]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize if needed
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            # Pad or trim to 30 seconds for Whisper
            target_length = 30 * sr
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            # Transcribe
            result = self.model.transcribe(
                audio,
                language=language,
                task="transcribe",
                word_timestamps=True,
                verbose=False
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Segment transcription error: {e}")
            raise
    
    def format_transcript(self, result: Dict) -> List[Dict]:
        """
        Format Whisper output to structured segments
        
        Args:
            result: Whisper transcription result
            
        Returns:
            List of formatted segments
        """
        segments = []
        
        for segment in result.get("segments", []):
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "words": segment.get("words", [])
            })
        
        return segments
    
    def get_full_text(self, result: Dict) -> str:
        """
        Extract full text from transcription result
        
        Args:
            result: Whisper result
            
        Returns:
            Full transcribed text
        """
        return result.get("text", "").strip()
