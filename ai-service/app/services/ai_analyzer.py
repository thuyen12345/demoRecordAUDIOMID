from openai import OpenAI
from typing import List, Dict
from loguru import logger
import json
import re
import httpx


class AIAnalyzer:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        ollama_base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 300,
    ):
        self.provider = (provider or "openai").lower()
        self.api_key = (api_key or "").strip()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = model
        self.ollama_base_url = (ollama_base_url or "http://127.0.0.1:11434").rstrip("/")
        self.timeout_seconds = timeout_seconds
        logger.info(f"Initialized AI Analyzer provider={self.provider}, model={model}")

    def _chunk_transcript(self, transcript: str, max_chars: int = 2000):
        chunks = []
        current = ""

        for line in transcript.split("\n"):
            if len(current) + len(line) < max_chars:
                current += line + "\n"
            else:
                if current.strip():
                    chunks.append(current)
                current = line + "\n"

        if current.strip():
            chunks.append(current)

        return chunks

    def _extract_json_object(self, text: str) -> str:
        text = (text or "").strip()

        if text.startswith("```json"):
            text = text[7:].strip()
        elif text.startswith("```"):
            text = text[3:].strip()

        if text.endswith("```"):
            text = text[:-3].strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        return text.strip()

    def _loads_json_safe(self, text: str) -> Dict:
        cleaned = self._extract_json_object(text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed at pos={e.pos}: {e}")
            logger.error(f"Raw response: {text}")
            logger.error(f"Cleaned response: {cleaned}")
            raise

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        data.setdefault("summary", "")
        data.setdefault("keywords", [])
        data.setdefault("technical_terms", [])
        data.setdefault("action_items", [])
        return data

    def _summarize_chunk(self, chunk: str) -> str:
        prompt = f"""
Hãy tóm tắt đoạn nội dung cuộc họp sau bằng tiếng Việt trong 2-3 câu.
Chỉ trả về phần tóm tắt.
Không thêm giải thích.
Giữ nguyên tên riêng, tên công nghệ, API, framework, thư viện, tên hàm, biến code hoặc thuật ngữ kỹ thuật nếu cần.

NỘI DUNG:
{chunk}
"""

        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 150
            },
            "messages": [
                {
                "role": "system",
                "content": "Bạn là trợ lý tóm tắt cuộc họp. Luôn trả lời bằng tiếng Việt, trừ tên riêng và thuật ngữ kỹ thuật cần giữ nguyên."
            },
                {"role": "user", "content": prompt}
            ],
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.ollama_base_url}/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()

        return body.get("message", {}).get("content", "").strip()

    def _is_usable_api_key(self) -> bool:
        if not self.api_key:
            return False

        lowered = self.api_key.lower()
        placeholder_markers = ["replace", "your_api_key", "changeme", "dummy", "test"]
        return not any(marker in lowered for marker in placeholder_markers)

    def _fallback_analysis(self, transcript: str, reason: str) -> Dict:
        text = (transcript or "").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        preview = " ".join(lines[:5]) if lines else "Không có nội dung transcript."

        words = re.findall(r"[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_\-]{2,}", text)
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
        try:
            logger.info("Starting AI meeting analysis (chunked)")

            chunks = self._chunk_transcript(transcript)
            logger.info(f"Split into {len(chunks)} chunks")

            summaries = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                s = self._summarize_chunk(chunk)
                summaries.append(s)

            combined_summary = "\n".join(summaries)

            final_prompt = f"""
Hãy phân tích phần tóm tắt cuộc họp sau và trả về đúng MỘT object JSON hợp lệ.

YÊU CẦU:
- Tất cả nội dung trong các value phải bằng tiếng Việt.
- Không dùng markdown.
- Không thêm giải thích ngoài JSON.
- Nếu không biết owner hoặc deadline thì để null.
- Giữ nguyên tên riêng, tên công nghệ, API, framework, thư viện, tên hàm, biến code hoặc thuật ngữ kỹ thuật nếu cần.
- "keywords" là các từ khóa chính của cuộc họp.
- "technical_terms" là các thuật ngữ kỹ thuật xuất hiện trong nội dung.
- "action_items" là các đầu việc cần thực hiện.

Schema:
{{
  "summary": "string",
  "keywords": ["string"],
  "technical_terms": ["string"],
  "action_items": [
    {{
      "task": "string",
      "owner": null,
      "deadline": null
    }}
  ]
}}

TEXT:
{combined_summary}
"""

            if self.provider == "ollama":
                result = self._analyze_with_ollama(final_prompt)
            else:
                if not self.client or not self._is_usable_api_key():
                    return self._fallback_analysis(transcript, "missing OpenAI API key")
                result = self._analyze_with_openai(final_prompt)

            logger.info("AI analysis completed (chunked)")
            return result

        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return self._fallback_analysis(transcript, repr(e))

    def _analyze_with_openai(self, prompt: str) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý phân tích biên bản họp. Hãy trả về đúng một object JSON hợp lệ và không thêm gì khác. Tất cả nội dung trong các value phải bằng tiếng Việt, trừ tên riêng và thuật ngữ kỹ thuật cần giữ nguyên."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content or "{}"
        return self._loads_json_safe(content)

    def _analyze_with_ollama(self, prompt: str) -> Dict:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_predict": 600
            },
            "messages": [
                {"role": "system", "content": "Bạn là trợ lý phân tích biên bản họp. Hãy trả về đúng một object JSON hợp lệ và không thêm gì khác. Tất cả nội dung trong các value phải bằng tiếng Việt, trừ tên riêng và thuật ngữ kỹ thuật cần giữ nguyên."},
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

        return self._loads_json_safe(content)

    def generate_summary(self, transcript: str) -> str:
        result = self.analyze_meeting(transcript)
        return result.get("summary", "")

    def extract_keywords(self, transcript: str) -> List[str]:
        result = self.analyze_meeting(transcript)
        return result.get("keywords", [])

    def extract_technical_terms(self, transcript: str) -> List[str]:
        result = self.analyze_meeting(transcript)
        return result.get("technical_terms", [])

    def extract_action_items(self, transcript: str) -> List[Dict]:
        result = self.analyze_meeting(transcript)
        return result.get("action_items", [])

    def format_transcript_for_analysis(self, aligned_segments: List[Dict]) -> str:
        lines = []

        for segment in aligned_segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")
            start = segment.get("start", 0)

            time_str = f"[{int(start//60):02d}:{int(start%60):02d}]"
            lines.append(f"{time_str} {speaker}: {text}")

        return "\n".join(lines)