from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import re
from loguru import logger
from sqlalchemy.orm import Session

from app.services.audio_processor import AudioProcessor
from app.services.speech_recognizer import SpeechRecognizer
from app.services.ai_analyzer import AIAnalyzer
from app.models import Transcript, Analysis
from app.config import get_settings

settings = get_settings()


class ProcessingPipeline:
    """
    Main processing pipeline orchestrating all services
    """
    
    def __init__(self):
        """Initialize all processing components"""
        
        logger.info("Initializing Processing Pipeline")
        
        # Lightweight components are initialized immediately.
        self.audio_processor = AudioProcessor(target_sr=16000)

        self.speech_recognizer = None
        self.speaker_diarizer = None
        self.ai_analyzer = None

        if not settings.lazy_load_models:
            self._ensure_models_loaded()
        
        logger.info("Processing Pipeline initialized successfully")

    def _ensure_models_loaded(self):
        """Load heavy models only when needed."""
        if self.speech_recognizer is None:
            self.speech_recognizer = SpeechRecognizer(
                model_name=settings.whisper_model,
                device=settings.device
            )

        if self.ai_analyzer is None:
            ai_provider = settings.ai_provider.lower()
            ai_model = settings.ollama_model if ai_provider == "ollama" else settings.openai_model
            self.ai_analyzer = AIAnalyzer(
                api_key=settings.openai_api_key,
                model=ai_model,
                provider=ai_provider,
                ollama_base_url=settings.ollama_base_url,
                timeout_seconds=settings.ollama_timeout_seconds,
            )

        if settings.enable_speaker_diarization and self.speaker_diarizer is None:
            from app.services.speaker_diarizer import SpeakerDiarizer
            self.speaker_diarizer = SpeakerDiarizer(
                hf_token=settings.huggingface_token,
                device=settings.device
            )

    def _resolve_audio_path(self, audio_path: str) -> str:
        """Resolve incoming path from other services to an existing local file path."""
        raw_path = Path(audio_path)

        project_root = Path(__file__).resolve().parent.parent
        workspace_root = project_root.parent

        candidates: list[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend([
                project_root / raw_path,
                workspace_root / raw_path,
                Path.cwd() / raw_path,
            ])

        # Fallback lookup: keep only filename and search common upload locations.
        audio_name = raw_path.name
        candidates.extend([
            Path.cwd() / "uploads" / audio_name,
            project_root / "uploads" / audio_name,
            workspace_root / "uploads" / audio_name,
            workspace_root / "meeting-service" / "uploads" / audio_name,
        ])

        checked = []
        seen = set()
        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str in seen:
                continue
            seen.add(candidate_str)
            checked.append(candidate_str)
            if candidate.exists() and candidate.is_file():
                resolved = str(candidate.resolve())
                logger.info(f"Resolved audio path: {audio_path} -> {resolved}")
                return resolved

        raise FileNotFoundError(
            f"Audio file not found for input path '{audio_path}'. Checked: {checked}"
        )

    def _build_initial_prompt(
        self,
        topic: Optional[str] = None,
        glossary_terms: Optional[List[str]] = None,
    ) -> str:
        """Build a concise prompt to bias Whisper toward domain terms."""
        topic_key = (topic or "").strip().lower()
        topic_defaults: Dict[str, List[str]] = {
            "engineering": [
                "model",
                "module",
                "API",
                "Docker",
                "Kubernetes",
                "SQL",
                "AWS",
                "IAM",
                "S3",
                "EC2",
                "RDS",
                "VPC",
                "EKS",
                "CloudWatch",
                "Lambda",
                "DevOps",
            ],
            "finance": ["invoice", "balance sheet", "VAT", "reconciliation", "cash flow"],
            "hr": ["onboarding", "payroll", "KPI", "OKR", "headcount"],
        }

        merged_terms = []
        seen = set()

        for term in topic_defaults.get(topic_key, []):
            key = term.lower()
            if key not in seen:
                seen.add(key)
                merged_terms.append(term)

        for term in (glossary_terms or []):
            clean_term = str(term).strip()
            if not clean_term:
                continue
            key = clean_term.lower()
            if key not in seen:
                seen.add(key)
                merged_terms.append(clean_term)

        base = (
            "Meeting transcript may contain Vietnamese mixed with English technical terms. "
            "Do not transliterate English terms into Vietnamese phonetics; keep original English spelling."
        )
        if not merged_terms:
            return base

        return f"{base} Keep original spelling for these terms: {', '.join(merged_terms)}."

    def _build_normalization_map(
        self,
        topic: Optional[str] = None,
        glossary_terms: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Common Vietnamese phonetic variants mapped to canonical technical terms."""
        topic_key = (topic or "").strip().lower()

        normalization_map: Dict[str, str] = {
            r"\bmo\s*đun\b": "module",
            r"\bmô\s*đun\b": "module",
            r"\bmô\s*đen\b": "model",
            r"\bê\s*pi\s*ai\b": "API",
            r"\bđốc\s*cơ\b": "Docker",
            r"\bxì\s*kiu\s*eo\b": "SQL",
            r"\bxì\s*kiu\s*él\b": "SQL",
            r"\bi\s*w\s*s\b": "AWS",
            r"\biw\.?s\b": "AWS",
            r"\bay\s*đắp\b": "AWS",
            r"\bờ\s*quai\s*ét\s*chờ\b": "CloudWatch",
            r"\bđép\s*ốp\b": "DevOps",
        }

        if topic_key == "finance":
            normalization_map.update(
                {
                    r"\bin\s*voi\s*xe\b": "invoice",
                    r"\bva\s*ti\b": "VAT",
                }
            )

        for term in (glossary_terms or []):
            clean = str(term).strip()
            if not clean:
                continue
            # Keep explicit canonical terms safe against accidental spacing in transcript.
            letters_spaced_pattern = r"\\b" + r"\\s*".join(re.escape(ch) for ch in clean) + r"\\b"
            normalization_map.setdefault(letters_spaced_pattern, clean)

        return normalization_map

    def _normalize_transcript_segments(
        self,
        segments: List[Dict],
        topic: Optional[str] = None,
        glossary_terms: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Normalize common misrecognized terms while preserving timestamps/speakers."""
        replacements = self._build_normalization_map(topic=topic, glossary_terms=glossary_terms)
        normalized = []

        for seg in segments:
            text = str(seg.get("text", ""))
            for pattern, target in replacements.items():
                text = re.sub(pattern, target, text, flags=re.IGNORECASE)

            normalized.append(
                {
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": text,
                    "words": seg.get("words", []),
                }
            )

        return normalized
    
    def process_meeting(
        self,
        audio_path: str,
        meeting_id: int,
        db: Session,
        topic: Optional[str] = None,
        glossary_terms: Optional[List[str]] = None,
        language: Optional[str] = "vi",
    ) -> Dict:
        """
        Complete processing pipeline for a meeting
        
        Pipeline:
        1. Load and preprocess audio
        2. Speech-to-text transcription
        3. Speaker diarization
        4. Align transcript with speakers
        5. AI analysis
        6. Save to database
        
        Args:
            audio_path: Path to audio file
            meeting_id: Meeting ID from meeting-service
            db: Database session
            
        Returns:
            Processing result dictionary
        """
        try:
            logger.info(f"Starting processing pipeline for meeting {meeting_id}")
            self._ensure_models_loaded()
            resolved_audio_path = self._resolve_audio_path(audio_path)
            
            # Step 1: Load audio
            logger.info("Step 1: Loading audio")
            try:
                self.audio_processor.load_audio(resolved_audio_path)
            except Exception as e:
                logger.warning(
                    f"Step 1 failed but pipeline will continue with Whisper direct input: {repr(e)}"
                )
            
            # Step 2: Speech-to-text
            logger.info("Step 2: Speech recognition")
            initial_prompt = self._build_initial_prompt(topic=topic, glossary_terms=glossary_terms)
            logger.info(f"Using Whisper initial prompt: {initial_prompt}")

            transcript_result = self.speech_recognizer.transcribe(
                resolved_audio_path,
                language=language,
                initial_prompt=initial_prompt,
            )
            transcript_segments = self.speech_recognizer.format_transcript(transcript_result)
            transcript_segments = self._normalize_transcript_segments(
                transcript_segments,
                topic=topic,
                glossary_terms=glossary_terms,
            )
            
            logger.info(f"Transcription complete: {len(transcript_segments)} segments")
            
            if settings.enable_speaker_diarization:
                # Step 3: Speaker diarization
                logger.info("Step 3: Speaker diarization")
                diarization = self.speaker_diarizer.diarize(resolved_audio_path)
                speaker_segments = self.speaker_diarizer.format_diarization(diarization)

                speaker_count = self.speaker_diarizer.get_speaker_count(diarization)
                logger.info(f"Diarization complete: {speaker_count} speakers detected")

                # Step 4: Align transcript with speakers
                logger.info("Step 4: Aligning transcript with speakers")
                aligned_segments = self.speaker_diarizer.align_transcript_with_speakers(
                    transcript_segments,
                    speaker_segments
                )
            else:
                logger.info("Step 3/4: Speaker diarization disabled (low-memory mode)")
                speaker_count = 1
                aligned_segments = [
                    {
                        "speaker": "SPEAKER_1",
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"],
                    }
                    for seg in transcript_segments
                ]
            
            # Step 5: AI Analysis
            logger.info("Step 5: AI analysis")
            formatted_transcript = self.ai_analyzer.format_transcript_for_analysis(aligned_segments)
            analysis_result = self.ai_analyzer.analyze_meeting(formatted_transcript)
            
            # Step 6: Save to database
            logger.info("Step 6: Saving to database")
            self._save_results(meeting_id, aligned_segments, analysis_result, db)
            
            logger.info(f"Processing complete for meeting {meeting_id}")
            
            return {
                "meeting_id": meeting_id,
                "status": "completed",
                "transcript_segments": len(aligned_segments),
                "speaker_count": speaker_count,
                "analysis": analysis_result
            }
            
        except Exception as e:
            logger.exception(f"Processing pipeline error for meeting {meeting_id}: {repr(e)}")
            raise
    
    def _save_results(
        self, 
        meeting_id: int, 
        aligned_segments: List[Dict],
        analysis_result: Dict,
        db: Session
    ):
        """
        Save processing results to database

        Args:
            meeting_id: Meeting ID
            aligned_segments: Aligned transcript segments
            analysis_result: AI analysis results
            db: Database session
        """
        try:
            def _to_builtin(value):
                if value is None or isinstance(value, (str, int, float, bool, datetime)):
                    return value

                # numpy scalar types support .item()
                if hasattr(value, "item"):
                    try:
                        return _to_builtin(value.item())
                    except Exception:
                        pass

                if isinstance(value, dict):
                    return {str(k): _to_builtin(v) for k, v in value.items()}

                if isinstance(value, (list, tuple)):
                    return [_to_builtin(v) for v in value]

                return str(value)

            # Save transcripts
            for segment in aligned_segments:
                transcript = Transcript(
                    meeting_id=meeting_id,
                    speaker=str(segment.get("speaker", "UNKNOWN")),
                    start_time=float(_to_builtin(segment.get("start", 0.0))),
                    end_time=float(_to_builtin(segment.get("end", 0.0))),
                    text=str(segment.get("text", ""))
                )
                db.add(transcript)

            # Save analysis
            clean_analysis = _to_builtin(analysis_result or {})
            analysis = Analysis(
                meeting_id=meeting_id,
                summary=str(clean_analysis.get("summary", "")),
                keywords=clean_analysis.get("keywords", []),
                technical_terms=clean_analysis.get("technical_terms", []),
                action_items=clean_analysis.get("action_items", [])
            )
            db.add(analysis)

            # Commit
            db.commit()

            logger.info(f"Saved {len(aligned_segments)} transcript segments and analysis")

        except Exception as e:
            db.rollback()
            logger.error(f"Database save error: {e}")
            raise
    
    def get_transcript(self, meeting_id: int, db: Session) -> List[Transcript]:
        """
        Retrieve transcript for a meeting
        
        Args:
            meeting_id: Meeting ID
            db: Database session
            
        Returns:
            List of transcript segments
        """
        transcripts = db.query(Transcript).filter(
            Transcript.meeting_id == meeting_id
        ).order_by(Transcript.start_time).all()
        
        return transcripts
    
    def get_analysis(self, meeting_id: int, db: Session) -> Analysis:
        """
        Retrieve analysis for a meeting
        
        Args:
            meeting_id: Meeting ID
            db: Database session
            
        Returns:
            Analysis object
        """
        analysis = db.query(Analysis).filter(
            Analysis.meeting_id == meeting_id
        ).first()
        
        return analysis
