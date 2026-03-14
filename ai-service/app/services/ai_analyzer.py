from openai import OpenAI
from typing import List, Dict
from loguru import logger
import json
import re
import httpx


class AIAnalyzer:
    """
    AI-powered meeting analysis using OpenAI GPT
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        ollama_base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 120,
    ):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4, gpt-4o, gpt-4-turbo, etc.)
        """
        self.provider = (provider or "openai").lower()
        self.api_key = (api_key or "").strip()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = model
        self.ollama_base_url = (ollama_base_url or "http://127.0.0.1:11434").rstrip("/")
        self.timeout_seconds = timeout_seconds
        logger.info(f"Initialized AI Analyzer provider={self.provider}, model={model}")

    def _is_usable_api_key(self) -> bool:
        if not self.api_key:
            return False

        lowered = self.api_key.lower()
        placeholder_markers = ["replace", "your_api_key", "changeme", "dummy", "test"]
        return not any(marker in lowered for marker in placeholder_markers)

    def _fallback_analysis(self, transcript: str, reason: str) -> Dict:
        """Return a minimal local analysis when OpenAI is unavailable."""
        text = (transcript or "").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        preview = " ".join(lines[:5]) if lines else "No transcript content available."

        words = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text)
        freq: Dict[str, int] = {}
        for w in words:
            k = w.lower()
            if k.startswith("speaker"):
                continue
            freq[k] = freq.get(k, 0) + 1

        keywords = [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:10]]

        logger.warning(f"Using fallback analysis: {reason}")
        return {
            "summary": preview,
            "keywords": keywords,
            "technical_terms": [],
            "action_items": [],
        }
    
    def analyze_meeting(self, transcript: str) -> Dict:
        """
        Perform complete meeting analysis
        
        Args:
            transcript: Full meeting transcript with speaker labels
            
        Returns:
            Dictionary with summary, keywords, technical terms, and action items
        """
        try:
            logger.info("Starting AI meeting analysis")

            prompt = f"""
You are an AI assistant specialized in analyzing meeting transcripts.

Analyze the following meeting transcript and provide:
1. A concise meeting summary (2-3 paragraphs)
2. Important keywords (5-10 keywords)
3. Technical or domain-specific terms mentioned
4. Action items with tasks, owners, and deadlines (if mentioned)

MEETING TRANSCRIPT:
{transcript}

Respond in the following JSON format:
{{
    "summary": "meeting summary here",
    "keywords": ["keyword1", "keyword2", ...],
    "technical_terms": ["term1", "term2", ...],
    "action_items": [
        {{
            "task": "task description",
            "owner": "person name or null",
            "deadline": "deadline or null"
        }}
    ]
}}

Important:
- Extract only information explicitly mentioned in the transcript
- If no owner or deadline is mentioned, use null
- Technical terms should be actual technical/domain keywords, not common words
"""

            if self.provider == "ollama":
                result = self._analyze_with_ollama(prompt)
            else:
                if not self.client or not self._is_usable_api_key():
                    return self._fallback_analysis(transcript, "missing OpenAI API key")
                result = self._analyze_with_openai(prompt)
            
            logger.info("AI analysis completed successfully")
            logger.info(f"Found {len(result.get('keywords', []))} keywords")
            logger.info(f"Found {len(result.get('technical_terms', []))} technical terms")
            logger.info(f"Found {len(result.get('action_items', []))} action items")
            
            return result
            
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return self._fallback_analysis(transcript, repr(e))

    def _analyze_with_openai(self, prompt: str) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful meeting analysis assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def _analyze_with_ollama(self, prompt: str) -> Dict:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": "You are a helpful meeting analysis assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.ollama_base_url}/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()

        content = body.get("message", {}).get("content", "")
        if not content:
            raise ValueError(f"Empty response from Ollama: {body}")

        return json.loads(content)
    
    def generate_summary(self, transcript: str) -> str:
        """
        Generate meeting summary only
        
        Args:
            transcript: Meeting transcript
            
        Returns:
            Meeting summary text
        """
        result = self.analyze_meeting(transcript)
        return result.get("summary", "")
    
    def extract_keywords(self, transcript: str) -> List[str]:
        """
        Extract keywords from transcript
        
        Args:
            transcript: Meeting transcript
            
        Returns:
            List of keywords
        """
        result = self.analyze_meeting(transcript)
        keywords = result.get("keywords", [])
        logger.info(f"Extracted {len(keywords)} keywords")
        return keywords
    
    def extract_technical_terms(self, transcript: str) -> List[str]:
        """
        Extract technical or domain-specific terms
        
        Args:
            transcript: Meeting transcript
            
        Returns:
            List of technical terms
        """
        result = self.analyze_meeting(transcript)
        terms = result.get("technical_terms", [])
        logger.info(f"Extracted {len(terms)} technical terms")
        return terms
    
    def extract_action_items(self, transcript: str) -> List[Dict]:
        """
        Extract action items with owners and deadlines
        
        Args:
            transcript: Meeting transcript
            
        Returns:
            List of action items
        """
        result = self.analyze_meeting(transcript)
        action_items = result.get("action_items", [])
        logger.info(f"Extracted {len(action_items)} action items")
        return action_items
    
    def format_transcript_for_analysis(self, aligned_segments: List[Dict]) -> str:
        """
        Format aligned transcript segments for AI analysis
        
        Args:
            aligned_segments: List of segments with speaker, time, and text
            
        Returns:
            Formatted transcript string
        """
        lines = []
        
        for segment in aligned_segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")
            start = segment.get("start", 0)
            
            # Format: [00:12] Speaker 1: Text
            time_str = f"[{int(start//60):02d}:{int(start%60):02d}]"
            lines.append(f"{time_str} {speaker}: {text}")
        
        return "\n".join(lines)
