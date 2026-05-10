import requests
import json
import re
from typing import List, Dict, Any, Optional

# ============================================================
# Shared helpers
# ============================================================
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with", "you", "your", "can", "may", "should",
}


def call_ollama(
    prompt: str,
    model: str = "mistral",
    temperature: float = 0.0,
    ollama_url: str = "http://localhost:11434/api/generate",
    timeout: int = 120,
    ) -> str:
    response = requests.post(
        ollama_url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            },
        },
        timeout=timeout,
    )
    
    response.raise_for_status()
    data = response.json()

    if "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {data}")

    return data["response"].strip()


def safe_json_parse(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def docs_to_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{doc.get('section_code', 'UNKNOWN')}] {doc.get('title', '')}\n{doc.get('chunk_text', '')}"
        for doc in retrieved_docs
    )


def split_claims(answer: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    claims = []

    for part in parts:
        cleaned = re.sub(r"^\s*[-*]\s*", "", part).strip()
        if len(cleaned.split()) >= 4:
            claims.append(cleaned)

    return claims


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*|\d+(?:\.\d+)?%?", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]