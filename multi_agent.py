"""Small scientific RAG core for mixed document uploads.

Design goal: keep the code modest. Files become section-ish chunks, chunks are
ranked with TF-IDF plus scientific/numeric/table boosts, and answers are
grounded in uploaded evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import hashlib
import base64
import textwrap
import zipfile
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from html import escape
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_: Any, **__: Any) -> bool:
        return False


EXTS = {".pdf", ".txt", ".md", ".csv", ".tsv", ".xlsx", ".xls", ".json", ".png", ".jpg", ".jpeg", ".webp", ".srt", ".vtt"}
PROVIDERS = {"local", "ollama", "openai", "claude", "grok", "gemini", "huggingface", "openrouter", "custom"}
DATABASE_URL = "DATABASE_URL"
USER_AGENT = "ScientificRAG-CompliantFetcher/1.0"


class TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = False
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip = False

    def handle_data(self, data: str) -> None:
        if not self.skip and data.strip():
            self.parts.append(re.sub(r"\s+", " ", data.strip()))

    @property
    def text(self) -> str:
        return "\n".join(self.parts)

INDIAN_PII_PATTERNS = {
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "phone": r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "upi": r"\b[\w.-]+@(?:upi|ybl|okicici|oksbi|okaxis|paytm|ibl|axl)\b",
    "account_like": r"\b\d{9,18}\b",
}


FREE_LLM_MODELS = [
    {
        "label": "Local evidence-only (no key)",
        "provider": "local",
        "model": "evidence-only",
        "base_url": "",
        "key_env": "",
    },
    {
        "label": "Ollama local - llama3.1",
        "provider": "ollama",
        "model": "llama3.1",
        "base_url": "http://localhost:11434/v1",
        "key_env": "",
    },
    {
        "label": "Ollama local - qwen2.5",
        "provider": "ollama",
        "model": "qwen2.5",
        "base_url": "http://localhost:11434/v1",
        "key_env": "",
    },
    {
        "label": "Ollama local - mistral",
        "provider": "ollama",
        "model": "mistral",
        "base_url": "http://localhost:11434/v1",
        "key_env": "",
    },
    {
        "label": "OpenRouter Free Router",
        "provider": "openrouter",
        "model": "openrouter/free",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    {
        "label": "OpenRouter Auto Free Model",
        "provider": "openrouter",
        "model": "auto",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    {
        "label": "OpenRouter Llama Free",
        "provider": "openrouter",
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    {
        "label": "OpenRouter Qwen Free",
        "provider": "openrouter",
        "model": "qwen/qwen-2.5-7b-instruct:free",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    {
        "label": "OpenRouter DeepSeek Free",
        "provider": "openrouter",
        "model": "deepseek/deepseek-chat:free",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    {
        "label": "Gemini Flash Free Tier",
        "provider": "gemini",
        "model": "gemini-1.5-flash",
        "base_url": "",
        "key_env": "GOOGLE_API_KEY",
    },
    {
        "label": "Hugging Face Router Free/Open Model",
        "provider": "huggingface",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "base_url": "https://router.huggingface.co/v1",
        "key_env": "HF_TOKEN",
    },
]


OCR_MODELS = [
    {
        "label": "Tesseract OCR v5 - free/local",
        "engine": "tesseract",
        "pricing": "free",
        "key_required": "no",
        "languages": "100+ including Hindi and English",
        "best_for": "printed documents, archives, lightweight digitization",
    },
    {
        "label": "PaddleOCR - free/local",
        "engine": "paddleocr",
        "pricing": "free",
        "key_required": "no",
        "languages": "80+ including Hindi and English",
        "best_for": "general OCR, mobile/scalable deployments, detection + recognition",
    },
    {
        "label": "IndicPhotoOCR - free/local or Bhashini ecosystem",
        "engine": "indicphotoocr",
        "pricing": "free/open ecosystem",
        "key_required": "no/self-host dependent",
        "languages": "11 Indian languages + English",
        "best_for": "Hindi/Indian scripts, signage, scene text, community documents",
    },
    {
        "label": "Google Document AI - paid/cloud",
        "engine": "google_document_ai",
        "pricing": "paid",
        "key_required": "GOOGLE_APPLICATION_CREDENTIALS",
        "languages": "global including Hindi and English",
        "best_for": "forms, invoices, contracts, structured enterprise extraction",
    },
    {
        "label": "Azure Document Intelligence - paid/cloud",
        "engine": "azure_document_intelligence",
        "pricing": "paid",
        "key_required": "AZURE_DOCUMENT_INTELLIGENCE_KEY",
        "languages": "global including Hindi and English",
        "best_for": "regulated enterprise OCR, forms, key-value extraction",
    },
    {
        "label": "AWS Textract - paid/cloud",
        "engine": "aws_textract",
        "pricing": "paid",
        "key_required": "AWS credentials",
        "languages": "English plus broad document extraction support",
        "best_for": "forms, tables, compliance workflows",
    },
    {
        "label": "VLM OCR via selected LLM - paid/free depending model",
        "engine": "vlm_ocr",
        "pricing": "depends on selected vision model",
        "key_required": "selected LLM provider key",
        "languages": "multilingual",
        "best_for": "complex layouts, tables, screenshots, HTML-like reconstruction",
    },
]

OCR_LANGUAGE_OPTIONS = [
    {"label": "Auto common India: English + Hindi + Urdu", "code": "eng+hin+urd", "script": "Latin/Devanagari/Arabic"},
    {"label": "English", "code": "eng", "script": "Latin"},
    {"label": "Hindi", "code": "hin", "script": "Devanagari"},
    {"label": "Urdu", "code": "urd", "script": "Arabic/Nastaliq"},
    {"label": "Arabic", "code": "ara", "script": "Arabic"},
    {"label": "Sanskrit", "code": "san", "script": "Devanagari"},
    {"label": "Bengali", "code": "ben", "script": "Bengali"},
    {"label": "Tamil", "code": "tam", "script": "Tamil"},
    {"label": "Telugu", "code": "tel", "script": "Telugu"},
    {"label": "Marathi", "code": "mar", "script": "Devanagari"},
    {"label": "Gujarati", "code": "guj", "script": "Gujarati"},
    {"label": "Kannada", "code": "kan", "script": "Kannada"},
    {"label": "Malayalam", "code": "mal", "script": "Malayalam"},
    {"label": "Punjabi", "code": "pan", "script": "Gurmukhi"},
    {"label": "Odia", "code": "ori", "script": "Odia"},
    {"label": "Nepali", "code": "nep", "script": "Devanagari"},
    {"label": "Sinhala", "code": "sin", "script": "Sinhala"},
    {"label": "Chinese Simplified", "code": "chi_sim", "script": "Han"},
    {"label": "Chinese Traditional", "code": "chi_tra", "script": "Han"},
    {"label": "Japanese", "code": "jpn", "script": "Kana/Kanji"},
    {"label": "Korean", "code": "kor", "script": "Hangul"},
    {"label": "French", "code": "fra", "script": "Latin"},
    {"label": "German", "code": "deu", "script": "Latin"},
    {"label": "Spanish", "code": "spa", "script": "Latin"},
    {"label": "Russian", "code": "rus", "script": "Cyrillic"},
    {"label": "Custom Tesseract language code", "code": "custom", "script": "Any installed traineddata"},
]


TRANSLITERATION_MODELS = [
    {"label": "Automatic LLM transliteration - selected provider", "engine": "auto_llm", "pricing": "depends on selected LLM", "key_required": "selected LLM key or local/Ollama"},
    {"label": "No transliteration", "engine": "none", "pricing": "free", "key_required": "no"},
    {"label": "Indic NLP Library - free/local", "engine": "indic_nlp", "pricing": "free", "key_required": "no"},
    {"label": "Aksharamukha - free/local/API", "engine": "aksharamukha", "pricing": "free/API dependent", "key_required": "no/API dependent"},
    {"label": "Indic transliteration rules - free/local", "engine": "indic_rules", "pricing": "free", "key_required": "no"},
    {"label": "iNLTK transliteration target - free/local optional", "engine": "inltk", "pricing": "free/optional", "key_required": "no"},
    {"label": "Google Input Tools - browser/manual aid", "engine": "google_input_tools", "pricing": "free/browser", "key_required": "no"},
    {"label": "Bhashini/Indic transliteration - free or platform dependent", "engine": "bhashini", "pricing": "free/platform dependent", "key_required": "BHASHINI_API_KEY if cloud"},
    {"label": "LLM-assisted transliteration - paid/free depending provider", "engine": "llm", "pricing": "depends on selected LLM", "key_required": "selected LLM key"},
]


SPEECH_TO_TEXT_MODELS = [
    {
        "label": "Paste transcript manually - free",
        "engine": "manual",
        "pricing": "free",
        "key_required": "no",
        "languages": "any typed transcript",
        "best_for": "fastest typing helper when external STT is unavailable",
    },
    {
        "label": "OpenAI Whisper API - paid/cloud",
        "engine": "openai_whisper",
        "pricing": "paid",
        "key_required": "OPENAI_API_KEY",
        "languages": "multilingual including Hindi and English",
        "best_for": "accurate speech-to-text from uploaded audio",
    },
    {
        "label": "Whisper local/faster-whisper - free/local",
        "engine": "whisper_local",
        "pricing": "free",
        "key_required": "no",
        "languages": "multilingual including Hindi and English",
        "best_for": "private local transcription when installed",
    },
    {
        "label": "Google Speech-to-Text - paid/cloud",
        "engine": "google_stt",
        "pricing": "paid",
        "key_required": "GOOGLE_APPLICATION_CREDENTIALS",
        "languages": "global including Hindi and English",
        "best_for": "enterprise multilingual transcription",
    },
    {
        "label": "Azure Speech - paid/cloud",
        "engine": "azure_speech",
        "pricing": "paid",
        "key_required": "AZURE_SPEECH_KEY",
        "languages": "global including Hindi and English",
        "best_for": "enterprise speech workflows",
    },
    {
        "label": "Bhashini ASR - platform dependent",
        "engine": "bhashini_asr",
        "pricing": "free/platform dependent",
        "key_required": "BHASHINI_API_KEY if cloud",
        "languages": "Indian languages",
        "best_for": "India-focused speech input and translation workflows",
    },
]


TEXT_TO_SPEECH_MODELS = [
    {
        "label": "Browser assistant voice - free/local",
        "engine": "browser_speech",
        "pricing": "free/browser",
        "key_required": "no",
        "languages": "browser/OS dependent",
        "best_for": "instant assistant-style spoken answers in the app",
        "url": "",
    },
    {
        "label": "Manual external TTS download - free",
        "engine": "manual_external",
        "pricing": "free",
        "key_required": "no",
        "languages": "depends on selected website",
        "best_for": "paste safe outreach text into a free TTS website and download MP3",
        "url": "",
    },
    {
        "label": "Galaxy.ai TTS - free/external",
        "engine": "galaxy_ai",
        "pricing": "free/external limits",
        "key_required": "no",
        "languages": "multilingual",
        "best_for": "fun outreach, character-style voices, WhatsApp-shareable MP3s",
        "url": "https://galaxy.ai/",
    },
    {
        "label": "QuillBot Voice - free/external",
        "engine": "quillbot_voice",
        "pricing": "free/external limits",
        "key_required": "no",
        "languages": "English plus major languages",
        "best_for": "professional narration, presentations, podcasts",
        "url": "https://quillbot.com/",
    },
    {
        "label": "Airvoz TTS - free/external",
        "engine": "airvoz",
        "pricing": "free/external limits",
        "key_required": "no",
        "languages": "100+ languages including Hindi",
        "best_for": "Hindi/community outreach, e-learning, accessibility narration",
        "url": "https://airvoz.com/",
    },
    {
        "label": "OpenAI TTS - paid/cloud",
        "engine": "openai_tts",
        "pricing": "paid",
        "key_required": "OPENAI_API_KEY",
        "languages": "multilingual",
        "best_for": "app-integrated MP3 generation with API key",
        "url": "",
    },
    {
        "label": "Edge TTS - free/local package",
        "engine": "edge_tts",
        "pricing": "free",
        "key_required": "no",
        "languages": "multilingual",
        "best_for": "local script-based voice generation when installed",
        "url": "",
    },
    {
        "label": "Coqui/Piper local TTS - free/local",
        "engine": "local_tts",
        "pricing": "free",
        "key_required": "no",
        "languages": "model dependent",
        "best_for": "privacy-preserving self-hosted voice generation",
        "url": "",
    },
    {
        "label": "eSpeak NG - classic free/offline",
        "engine": "espeak_ng",
        "pricing": "free/open-source",
        "key_required": "no",
        "languages": "100+ including Hindi/English support depending voice data",
        "best_for": "accessibility, embedded systems, low-resource offline narration",
        "url": "https://github.com/espeak-ng/espeak-ng",
    },
    {
        "label": "Festival Speech Synthesis - classic free/offline",
        "engine": "festival",
        "pricing": "free/open-source",
        "key_required": "no",
        "languages": "multiple languages depending voice packages",
        "best_for": "academic, DIY, customizable offline speech",
        "url": "https://www.cstr.ed.ac.uk/projects/festival/",
    },
    {
        "label": "MaryTTS - underrated free/research",
        "engine": "marytts",
        "pricing": "free/open-source",
        "key_required": "no",
        "languages": "multilingual depending voices",
        "best_for": "research, prosody control, community projects",
        "url": "https://github.com/marytts/marytts",
    },
    {
        "label": "Tacotron 2 - free/experimental neural TTS",
        "engine": "tacotron2",
        "pricing": "free/open-source implementations",
        "key_required": "no",
        "languages": "model/data dependent",
        "best_for": "learning neural TTS and student experiments",
        "url": "https://github.com/NVIDIA/tacotron2",
    },
    {
        "label": "Mozilla TTS / Coqui TTS legacy - free/community",
        "engine": "mozilla_tts",
        "pricing": "free/open-source",
        "key_required": "no",
        "languages": "multilingual depending models",
        "best_for": "community neural narration and offline experiments",
        "url": "https://github.com/coqui-ai/TTS",
    },
    {
        "label": "OpenAI Jukebox - free/research music generation",
        "engine": "jukebox",
        "pricing": "free/research code",
        "key_required": "no",
        "languages": "music/singing, experimental",
        "best_for": "creative music/singing experiments, not routine outreach narration",
        "url": "https://github.com/openai/jukebox",
    },
]


SWARM_AGENTS = [
    {"name": "planner", "role": "planner", "weight": 1.0, "level": 1, "status": "active"},
    {"name": "retriever", "role": "executor", "weight": 1.0, "level": 1, "status": "active"},
    {"name": "verifier", "role": "verifier", "weight": 1.2, "level": 2, "status": "active"},
    {"name": "school_clerk", "role": "office_automation", "weight": 1.1, "level": 1, "status": "active"},
    {"name": "compliance_guard", "role": "guard", "weight": 1.4, "level": 2, "status": "active"},
    {"name": "orchestrator", "role": "orchestrator", "weight": 1.6, "level": 3, "status": "active"},
]

SWARM_TOPOLOGIES = [
    "Hybrid",
    "Hierarchy",
    "Mesh",
    "Star",
    "Pipeline",
    "Ring",
    "Tree",
    "Blackboard",
    "Committee",
]


def ocr_model_options() -> List[Dict[str, str]]:
    return [dict(x) for x in OCR_MODELS]


def ocr_language_options() -> List[Dict[str, str]]:
    return [dict(x) for x in OCR_LANGUAGE_OPTIONS]


def transliteration_options() -> List[Dict[str, str]]:
    return [dict(x) for x in TRANSLITERATION_MODELS]


def speech_to_text_options() -> List[Dict[str, str]]:
    return [dict(x) for x in SPEECH_TO_TEXT_MODELS]


def text_to_speech_options() -> List[Dict[str, str]]:
    return [dict(x) for x in TEXT_TO_SPEECH_MODELS]


def _has_pkg(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def toolbox_catalog() -> List[Dict[str, str]]:
    rows = [
        ("RAG chatbot", "free/local", "scikit-learn, pandas, pypdf", "", "TF-IDF + grounded citations"),
        ("OpenAI embeddings", "paid/key", "openai", "OPENAI_API_KEY", "text-embedding-3-large"),
        ("LLM routing", "free/paid/key", "openai/google-generativeai", "OPENAI_API_KEY/GROK_API_KEY/GOOGLE_API_KEY/HF_TOKEN/OPENROUTER_API_KEY", "provider dropdown + custom endpoint"),
        ("PostgreSQL memory", "free/paid", "psycopg", "DATABASE_URL", "chunks, queries, integration registry"),
        ("OCR", "free/paid", "pytesseract/Pillow", "OCR_LANG/OCR_ENGINE", "Tesseract built in; others selectable integrations"),
        ("OCR languages", "free/local", "pytesseract", "OCR_LANG", "Tesseract traineddata codes including eng, hin, urd, ara, ben, tam, tel, custom"),
        ("Semantic chunking", "free/local", "transformers/torch", "CHUNKING_ENGINE/MBERT_MODEL", "section-aware chunks with optional mBERT semantic breakpoints"),
        ("NLP transliteration", "free/local/optional", "indic_nlp_library/aksharamukha/indic_transliteration", "TRANSLITERATION_ENGINE", "Indic NLP, Aksharamukha, iNLTK target, Google Input Tools guidance, LLM fallback"),
        ("Speech to text", "free/paid", "openai", "OPENAI_API_KEY", "manual transcript + Whisper API + integration targets"),
        ("Text to speech", "free/paid", "none required", "", "safe script + external/free/local model registry"),
        ("Website builder", "free/local", "streamlit", "", "HTML preview and download"),
        ("Templates", "free/local", "streamlit", "", "HTML/Markdown/JSON/CSV template generation"),
        ("School clerk automation", "free/local", "pandas", "", "result sheets, attendance, notices, certificates, roll lists"),
        ("Marketing", "free/local", "streamlit", "", "evidence-grounded campaign planning"),
        ("Media management", "free/local", "pandas/Pillow", "", "image/table/figure inventory"),
        ("Compliant web ingestion", "free/local", "stdlib", "", "robots.txt, URL confirmation, redaction, size limits"),
        ("International compliance", "free/local", "stdlib", "", "India DPDP + GDPR/UK/CCPA safe controls"),
        ("Swarm topology", "free/local", "streamlit", "", "hybrid/hierarchy/mesh/star/pipeline/ring/tree/blackboard/committee"),
        ("Human review", "free/local", "streamlit", "", "approval gates, metadata, audit trail"),
        ("Codex-style workflow", "free/local", "streamlit", "", "workspace-first actions, verification, review, package handoff"),
        ("Metrics and evals", "free/local + optional paid", "monitoring/langsmith/evidently/wandb", "LANGSMITH_API_KEY/WANDB_API_KEY", "RAG quality metrics, feedback loop, API readiness, MCP planning"),
    ]
    out = []
    for feature, cost, packages, envs, note in rows:
        pkg_names = [p.strip().split("[")[0].replace("-", "_") for p in re.split(r"[,/]", packages) if p.strip() and p.strip() not in {"stdlib", "none required"}]
        env_names = [e.strip() for e in re.split(r"[/,]", envs) if e.strip()]
        pkg_ready = all(_has_pkg(p) for p in pkg_names) if pkg_names else True
        env_ready = any(os.getenv(e) for e in env_names) if env_names else True
        out.append(
            {
                "feature": feature,
                "cost": cost,
                "packages": packages,
                "keys_or_env": envs or "none",
                "package_ready": "yes" if pkg_ready else "optional/missing",
                "key_ready": "yes" if env_ready else "not set",
                "note": note,
            }
        )
    return out


def tts_guidance(text: str, engine: str, language: str = "Hindi/English") -> Dict[str, str]:
    selected = next((x for x in TEXT_TO_SPEECH_MODELS if x["engine"] == engine), TEXT_TO_SPEECH_MODELS[0])
    pii = detect_personal_data(text)
    safe_text = redact_personal_data(text) if pii else text
    warning = (
        "Personal data was detected and redacted for safer third-party TTS use."
        if pii else
        "No common Indian personal identifiers were detected."
    )
    return {
        "engine": selected["engine"],
        "label": selected["label"],
        "pricing": selected["pricing"],
        "key_required": selected["key_required"],
        "language": language,
        "url": selected.get("url", ""),
        "safe_text": safe_text,
        "warning": warning,
        "note": "For external free TTS websites, paste only non-sensitive text and review voice rights, platform terms, and local law before publishing.",
    }


def synthesize_speech(text: str, engine: str = "browser_speech", voice: str = "alloy", language: str = "") -> Dict[str, Any]:
    """Generate spoken audio when a configured TTS backend is available."""

    if not text.strip():
        return {"ok": False, "audio": b"", "mime": "", "ext": "", "note": "No text was provided."}
    safe_text = redact_personal_data(text)[:8000]
    if engine == "browser_speech":
        return {"ok": False, "audio": b"", "mime": "", "ext": "", "note": "Use the browser Speak button. No server audio file is required."}
    if engine == "openai_tts":
        if not os.getenv("OPENAI_API_KEY"):
            return {"ok": False, "audio": b"", "mime": "", "ext": "", "note": "OPENAI_API_KEY is required for OpenAI TTS."}
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.audio.speech.create(
                model=os.getenv("OPENAI_TTS_MODEL", "tts-1"),
                voice=os.getenv("OPENAI_TTS_VOICE", voice or "alloy"),
                input=safe_text[:4000],
                response_format="mp3",
            )
            audio = getattr(response, "content", None)
            if audio is None and hasattr(response, "read"):
                audio = response.read()
            return {"ok": True, "audio": bytes(audio or b""), "mime": "audio/mpeg", "ext": "mp3", "note": "Generated with OpenAI TTS."}
        except Exception as exc:
            return {"ok": False, "audio": b"", "mime": "", "ext": "", "note": f"OpenAI TTS failed: {exc}"}
    if engine == "edge_tts":
        try:
            import asyncio
            import edge_tts

            async def _run() -> bytes:
                out = BytesIO()
                communicate = edge_tts.Communicate(safe_text[:6000], os.getenv("EDGE_TTS_VOICE", "en-IN-NeerjaNeural"))
                async for chunk in communicate.stream():
                    if chunk.get("type") == "audio":
                        out.write(chunk.get("data", b""))
                return out.getvalue()

            audio = asyncio.run(_run())
            return {"ok": bool(audio), "audio": audio, "mime": "audio/mpeg", "ext": "mp3", "note": "Generated with Edge TTS." if audio else "Edge TTS returned no audio."}
        except Exception as exc:
            return {"ok": False, "audio": b"", "mime": "", "ext": "", "note": f"Edge TTS is not configured: {exc}"}
    return {"ok": False, "audio": b"", "mime": "", "ext": "", "note": f"TTS engine `{engine}` is selectable but does not generate in-app audio yet. Use its linked external/local tool."}


def whatsapp_toolkit(message: str, service_url: str = "", audience: str = "opted-in users") -> Dict[str, Any]:
    pii = detect_personal_data(message)
    safe_message = redact_personal_data(message) if pii else message
    return {
        "channel": "WhatsApp Business Platform / Cloud API",
        "audience": audience,
        "safe_message": safe_message,
        "service_url": service_url,
        "policy_guardrails": [
            "Use only WhatsApp Business Platform or authorized providers.",
            "Send business-initiated messages only with approved templates where required.",
            "Respect the 24-hour customer service window for free-form replies.",
            "Use opt-in contacts only; keep consent and unsubscribe/stop handling.",
            "Do not use WhatsApp for unsolicited bulk spam.",
            "Do not expose sensitive personal data in messages or media links.",
            "Review Meta Commerce, Business, and Messaging policies before launch.",
            "Keep a human review and escalation path for sensitive or government/institutional outreach.",
        ],
        "template_draft": {
            "name": "service_update_outreach",
            "category": "UTILITY",
            "language": "en",
            "body": safe_message[:900] + ("\n\nLink: " + service_url if service_url else ""),
            "buttons": [{"type": "URL", "text": "Open", "url": service_url}] if service_url else [],
        },
        "cloud_api_text_payload": {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": "{{recipient_phone_e164}}",
            "type": "text",
            "text": {"preview_url": bool(service_url), "body": safe_message + (f"\n{service_url}" if service_url else "")},
        },
        "cloud_api_template_payload": {
            "messaging_product": "whatsapp",
            "to": "{{recipient_phone_e164}}",
            "type": "template",
            "template": {
                "name": "{{approved_template_name}}",
                "language": {"code": "en"},
                "components": [{"type": "body", "parameters": [{"type": "text", "text": safe_message[:900]}]}],
            },
        },
        "required_env": ["WHATSAPP_TOKEN", "WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_BUSINESS_ACCOUNT_ID"],
        "note": "Drafts only unless official WhatsApp Cloud API credentials are configured and the recipient has opted in.",
    }


def whatsapp_send_text(to: str, body: str) -> Dict[str, Any]:
    token = os.getenv("WHATSAPP_TOKEN")
    phone_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    if not token or not phone_id:
        return {"ok": False, "note": "WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID are required."}
    try:
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": True, "body": redact_personal_data(body)},
        }
        req = Request(
            f"https://graph.facebook.com/v20.0/{phone_id}/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "User-Agent": USER_AGENT},
            method="POST",
        )
        with urlopen(req, timeout=20) as resp:
            return {"ok": True, "response": json.loads(resp.read(1_000_000).decode("utf-8"))}
    except Exception as exc:
        return {"ok": False, "note": f"WhatsApp send failed: {exc}"}


def swarm_initial_state() -> Dict[str, Any]:
    return {
        "human": {"role": "final_authority", "level": 99, "immutable": True},
        "agents": [dict(a) | {"positive": 0, "negative": 0, "attention": a["weight"]} for a in SWARM_AGENTS],
        "topology": "Hybrid",
        "available_topologies": SWARM_TOPOLOGIES,
        "rules": {
            "promotion_threshold": 3,
            "demotion_threshold": 2,
            "max_agent_level": 3,
            "human_always_above_orchestrator": True,
        },
    }


def update_swarm_feedback(state: Dict[str, Any], agent_name: str, feedback: str) -> Dict[str, Any]:
    out = json.loads(json.dumps(state or swarm_initial_state()))
    for agent in out["agents"]:
        if agent["name"] != agent_name:
            continue
        if feedback == "positive":
            agent["positive"] = int(agent.get("positive", 0)) + 1
            agent["attention"] = round(float(agent.get("attention", 1.0)) + 0.15, 3)
            if agent["positive"] >= out["rules"]["promotion_threshold"] and agent["level"] < out["rules"]["max_agent_level"]:
                agent["level"] += 1
                agent["positive"] = 0
                agent["status"] = "promoted" if agent["level"] < 3 else "orchestrator_candidate"
        elif feedback == "negative":
            agent["negative"] = int(agent.get("negative", 0)) + 1
            agent["attention"] = round(max(0.1, float(agent.get("attention", 1.0)) - 0.2), 3)
            if agent["negative"] >= out["rules"]["demotion_threshold"] and agent["level"] > 1:
                agent["level"] -= 1
                agent["negative"] = 0
                agent["status"] = "demoted"
    out["agents"] = sorted(out["agents"], key=lambda x: (x["level"], x["attention"]), reverse=True)
    return out


def swarm_mermaid(state: Dict[str, Any], topology: str = "Hybrid") -> str:
    agents = state.get("agents", []) if state else swarm_initial_state()["agents"]
    names = [a["name"] for a in agents if a["name"] != "orchestrator"]
    labels = {a["name"]: f'{a["name"]}["{a["name"]}\\nlevel {a["level"]}\\nattention {a.get("attention", a.get("weight", 1))}"]' for a in agents}
    lines = ["flowchart TD", '  H["Human Reviewer\\nFinal Authority"] --> O["Orchestrator\\nAgent ceiling"]']
    for node in labels.values():
        lines.append("  " + node)
    if topology in {"Hierarchy", "Hybrid"}:
        for name in names:
            lines.append(f"  O --> {name}")
    if topology in {"Star", "Hybrid"}:
        lines += ["  O <--> planner", "  O <--> retriever", "  O <--> verifier", "  O <--> compliance_guard"]
    if topology in {"Pipeline", "Hybrid"}:
        lines += ["  planner --> retriever", "  retriever --> verifier", "  verifier --> compliance_guard", "  compliance_guard --> O"]
    if topology in {"Ring", "Hybrid"}:
        lines += ["  planner -.-> retriever", "  retriever -.-> verifier", "  verifier -.-> compliance_guard", "  compliance_guard -.-> planner"]
    if topology in {"Mesh", "Hybrid"}:
        lines += ["  planner <--> verifier", "  retriever <--> compliance_guard", "  planner <--> compliance_guard", "  retriever <--> verifier"]
    if topology in {"Tree", "Hybrid"}:
        lines += ["  O --> planner", "  planner --> retriever", "  planner --> verifier", "  verifier --> compliance_guard"]
    if topology in {"Blackboard", "Hybrid"}:
        lines += ['  B["Shared Blackboard\\nEvidence + Metadata"]', "  planner <--> B", "  retriever <--> B", "  verifier <--> B", "  compliance_guard <--> B", "  B --> O"]
    if topology in {"Committee", "Hybrid"}:
        lines += ['  C["Committee Vote\\nPlanner + Verifier + Guard"]', "  planner --> C", "  verifier --> C", "  compliance_guard --> C", "  C --> H"]
    lines += ["  O --> H", "  compliance_guard --> H", "  verifier --> H"]
    return "\n".join(lines)


def transcribe_audio(raw: bytes, filename: str, engine: str = "manual", language: str = "") -> str:
    if not raw:
        return ""
    if engine == "openai_whisper" and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            audio = BytesIO(raw)
            audio.name = filename
            result = client.audio.transcriptions.create(
                model=os.getenv("OPENAI_STT_MODEL", "whisper-1"),
                file=audio,
                language=language or None,
            )
            return getattr(result, "text", "") or ""
        except Exception as exc:
            return f"STT failed: {exc}"
    return f"STT engine `{engine}` is selected but not configured in this deployment. Paste the transcript manually."


def detect_personal_data(text: str) -> Dict[str, int]:
    found: Dict[str, int] = {}
    for name, pattern in INDIAN_PII_PATTERNS.items():
        count = len(re.findall(pattern, text or "", flags=re.I))
        if count:
            found[name] = count
    return found


def redact_personal_data(text: str) -> str:
    redacted = text or ""
    for name, pattern in INDIAN_PII_PATTERNS.items():
        redacted = re.sub(pattern, f"[REDACTED_{name.upper()}]", redacted, flags=re.I)
    return redacted


def compliance_report(corpus: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, int] = {}
    sources = set()
    for c in corpus:
        hits = detect_personal_data(c.get("text", ""))
        if hits:
            sources.add(c.get("source", ""))
        for key, value in hits.items():
            totals[key] = totals.get(key, 0) + value
    return {
        "jurisdiction": os.getenv("COMPLIANCE_JURISDICTION", "Global/Unknown"),
        "framework": "International privacy/copyright/safe-fetch readiness controls",
        "jurisdiction_policy": jurisdiction_policy(os.getenv("COMPLIANCE_JURISDICTION", "Global/Unknown")),
        "personal_data_detected": totals,
        "affected_sources": sorted(s for s in sources if s),
        "cloud_consent": os.getenv("DPDP_CLOUD_CONSENT", "false").lower() == "true",
        "redaction_enabled": os.getenv("DPDP_REDACT", "true").lower() == "true",
        "international_guidelines": [
            "Respect robots.txt and site terms before web ingestion.",
            "Do not bypass paywalls, logins, access controls, CAPTCHAs, or anti-bot systems.",
            "Minimise personal data and redact before cloud processing when possible.",
            "Use a lawful basis/consent where required.",
            "Keep evidence citations and do not fabricate claims.",
            "Apply retention, deletion, access control, breach, and data-subject/data-principal rights workflows.",
            "Review copyright/database rights before reusing fetched content.",
        ],
        "note": "Technical safeguard only; confirm local legal basis, notices, retention, breach handling, transfer rules, and rights workflow with qualified counsel.",
    }


def free_llm_models(extra: str = "") -> List[Dict[str, str]]:
    """Return built-in free options plus registry rows marked free/open."""

    items = [dict(x) for x in FREE_LLM_MODELS]
    for row in load_integrations_pg():
        text = " ".join(str(row.get(k, "")) for k in ("pricing", "category", "use")).lower()
        if any(word in text for word in ("free", "open", "oss")) and row.get("model"):
            provider = str(row.get("category", "custom")).lower()
            if provider not in PROVIDERS:
                provider = "custom"
            items.append(
                {
                    "label": row["name"],
                    "provider": provider,
                    "model": row.get("model", ""),
                    "base_url": row.get("base_url", ""),
                    "key_env": row.get("api_key_env", ""),
                }
            )
    for line in extra.splitlines():
        parts = [p.strip() for p in re.split(r"[,|]", line) if p.strip()]
        if len(parts) >= 2:
            items.append(
                {
                    "label": parts[0],
                    "provider": parts[1] if len(parts) > 1 else "custom",
                    "model": parts[2] if len(parts) > 2 else "",
                    "base_url": parts[3] if len(parts) > 3 else "",
                    "key_env": parts[4] if len(parts) > 4 else "",
                }
            )
    return items


def openrouter_catalog(limit: int = 300) -> List[Dict[str, str]]:
    """Best-effort live OpenRouter model catalog; falls back silently offline."""

    try:
        req = Request("https://openrouter.ai/api/v1/models?output_modalities=text", headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read(2_000_000).decode("utf-8"))
        rows = []
        for model in data.get("data", [])[:limit]:
            pricing = model.get("pricing") or {}
            prompt = float(pricing.get("prompt") or 0)
            completion = float(pricing.get("completion") or 0)
            free = prompt == 0 and completion == 0
            rows.append(
                {
                    "label": f"{'FREE' if free else 'PAID'} | OpenRouter | {model.get('name') or model.get('id')}",
                    "provider": "openrouter",
                    "model": model.get("id", ""),
                    "base_url": "https://openrouter.ai/api/v1",
                    "key_env": "OPENROUTER_API_KEY",
                    "pricing": "free" if free else "paid/key",
                    "requires_key": "yes",
                    "context": str(model.get("context_length", "")),
                }
            )
        return rows
    except Exception:
        return []


def llm_model_catalog(extra: str = "") -> List[Dict[str, str]]:
    """Free + paid LLM dropdown rows, with key env metadata."""

    rows = [
        {
            "label": "FREE | Local | Evidence-only grounded mode",
            "provider": "local",
            "model": "evidence-only",
            "base_url": "",
            "key_env": "",
            "pricing": "free",
            "requires_key": "no",
        }
    ]
    rows.extend(
        {
            "label": f"FREE | Ollama | {item['model']}",
            "provider": "ollama",
            "model": item["model"],
            "base_url": item["base_url"],
            "key_env": "",
            "pricing": "free/local",
            "requires_key": "no",
        }
        for item in FREE_LLM_MODELS
        if item.get("provider") == "ollama"
    )
    rows.extend(openrouter_catalog())
    if not any(r["model"] == "openrouter/free" for r in rows):
        for item in free_llm_models(extra):
            item = dict(item)
            item["label"] = f"FREE | {item.get('provider', 'model')} | {item['label']}"
            item["pricing"] = "free"
            item["requires_key"] = "no" if item.get("provider") == "local" else "yes"
            rows.append(item)
    rows.extend(
        [
            {"label": "PAID | OpenAI | GPT-4o mini", "provider": "openai", "model": "gpt-4o-mini", "base_url": "", "key_env": "OPENAI_API_KEY", "pricing": "paid/key", "requires_key": "yes"},
            {"label": "PAID | OpenAI | GPT-4o", "provider": "openai", "model": "gpt-4o", "base_url": "", "key_env": "OPENAI_API_KEY", "pricing": "paid/key", "requires_key": "yes"},
            {"label": "PAID | Claude | Claude 3.5 Sonnet", "provider": "claude", "model": "claude-3-5-sonnet-latest", "base_url": "https://api.anthropic.com/v1/messages", "key_env": "ANTHROPIC_API_KEY", "pricing": "paid/key", "requires_key": "yes"},
            {"label": "PAID | Claude | Claude 3.5 Haiku", "provider": "claude", "model": "claude-3-5-haiku-latest", "base_url": "https://api.anthropic.com/v1/messages", "key_env": "ANTHROPIC_API_KEY", "pricing": "paid/key", "requires_key": "yes"},
            {"label": "PAID | Grok/xAI | grok-2-latest", "provider": "grok", "model": "grok-2-latest", "base_url": "https://api.x.ai/v1", "key_env": "GROK_API_KEY", "pricing": "paid/key", "requires_key": "yes"},
            {"label": "FREE/PAID | Gemini | gemini-1.5-flash", "provider": "gemini", "model": "gemini-1.5-flash", "base_url": "", "key_env": "GOOGLE_API_KEY", "pricing": "free-tier/key", "requires_key": "yes"},
            {"label": "FREE/PAID | Hugging Face | Llama 3.1 8B Instruct", "provider": "huggingface", "model": "meta-llama/Llama-3.1-8B-Instruct", "base_url": "https://router.huggingface.co/v1", "key_env": "HF_TOKEN", "pricing": "free-tier/key", "requires_key": "yes"},
            {"label": "CUSTOM | Any OpenAI-compatible API", "provider": "custom", "model": os.getenv("CUSTOM_LLM_MODEL", ""), "base_url": os.getenv("CUSTOM_LLM_BASE_URL", ""), "key_env": os.getenv("CUSTOM_LLM_API_KEY_ENV", "CUSTOM_LLM_API_KEY"), "pricing": "free/paid depends", "requires_key": "depends"},
        ]
    )
    seen, deduped = set(), []
    for row in rows:
        key = (row.get("provider"), row.get("model"), row.get("label"))
        if key not in seen:
            deduped.append(row)
            seen.add(key)
    return deduped


@dataclass
class Chunk:
    source: str
    text: str
    page: int = 1
    section: str = "Document"
    kind: str = "text"
    score: float = 0.0

    @property
    def numbers(self) -> List[str]:
        return re.findall(r"[-+]?\d+(?:\.\d+)?\s?(?:%|Å|A|nm|µM|uM|mM|kDa|Da|°C|K)?", self.text)


def _bytes(path: Path, member: str | None = None) -> bytes:
    if member:
        with zipfile.ZipFile(path) as zf:
            return zf.read(member)
    return path.read_bytes()


def _members(path: Path) -> List[Tuple[str, bytes]]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            return [(n, zf.read(n)) for n in zf.namelist() if Path(n).suffix.lower() in EXTS]
    if path.suffix.lower() in EXTS:
        return [(path.name, path.read_bytes())]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="ignore")


def _pdf(raw: bytes, max_pages: int) -> List[Tuple[int, str]]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(raw))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages[:max_pages])]
    except Exception as exc:
        return [(1, f"PDF extraction failed: {exc}")]


def _table(raw: bytes, name: str) -> str:
    try:
        import pandas as pd

        ext = Path(name).suffix.lower()
        if ext in {".csv", ".tsv"}:
            return pd.read_csv(BytesIO(raw), sep="\t" if ext == ".tsv" else ",").to_markdown(index=False)
        sheets = pd.read_excel(BytesIO(raw), sheet_name=None)
        return "\n\n".join(f"Sheet: {k}\n{v.to_markdown(index=False)}" for k, v in sheets.items())
    except Exception as exc:
        return f"Table extraction failed for {name}: {exc}"


def _image(raw: bytes, name: str) -> str:
    try:
        from PIL import Image

        img = Image.open(BytesIO(raw))
        meta = f"Image {name}: {img.format}, {img.size[0]}x{img.size[1]}, mode {img.mode}."
        engine = os.getenv("OCR_ENGINE", "tesseract")
        try:
            lang = os.getenv("OCR_LANG", "eng")
            if engine == "tesseract":
                import pytesseract

                text = pytesseract.image_to_string(img, lang=lang).strip()
            else:
                text = ""
            return meta + ("\nOCR:\n" + text if text else "\nNo OCR text detected.")
        except Exception:
            return meta + f"\nOCR engine `{engine}` is selected but not configured in this deployment."
    except Exception as exc:
        return f"Image extraction failed for {name}: {exc}"


def _subtitle(raw: bytes, name: str) -> str:
    text = _decode(raw).replace("\ufeff", "")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    rows = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0].upper().startswith(("WEBVTT", "NOTE")):
            continue
        if lines[0].isdigit():
            lines = lines[1:]
        stamp = ""
        if lines and "-->" in lines[0]:
            stamp = lines[0].split("-->", 1)[0].strip()
            lines = lines[1:]
        clean = " ".join(re.sub(r"<[^>]+>", "", line) for line in lines).strip()
        if clean:
            rows.append(f"[{stamp or '00:00:00'}] {clean}")
    return f"Subtitle transcript from {name}\n" + "\n".join(rows) if rows else text


def _text(raw: bytes, name: str, max_pages: int) -> List[Tuple[int, str, str]]:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return [(p, t, "text") for p, t in _pdf(raw, max_pages)]
    if ext in {".csv", ".tsv", ".xlsx", ".xls"}:
        return [(1, _table(raw, name), "table")]
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return [(1, _image(raw, name), "image")]
    if ext in {".srt", ".vtt"}:
        return [(1, _subtitle(raw, name), "transcript")]
    if ext == ".json":
        try:
            return [(1, json.dumps(json.loads(_decode(raw)), indent=2), "text")]
        except Exception:
            pass
    return [(1, _decode(raw), "text")]


def _section(line: str) -> bool:
    s = line.strip()
    known = {"abstract", "summary", "introduction", "background", "methods", "materials and methods", "results", "discussion", "conclusion", "references", "supplementary", "experimental procedures"}
    return bool(s) and len(s) < 100 and not s.endswith(".") and (s.istitle() or s.lower() in known or bool(re.match(r"^\d+(?:\.\d+)*\s+\w+", s)))


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?।؟])\s+|\n+", text or "")
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]


def _cosine(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def _mbert_vectors(sentences: List[str]) -> Optional[List[List[float]]]:
    if not sentences:
        return []
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_name = os.getenv("MBERT_MODEL", "bert-base-multilingual-cased")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        out: List[List[float]] = []
        batch_size = int(os.getenv("MBERT_BATCH_SIZE", "8"))
        with torch.no_grad():
            for start in range(0, len(sentences), batch_size):
                batch = sentences[start:start + batch_size]
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                out.extend(pooled.cpu().tolist())
        return out
    except Exception:
        return None


def _semantic_parts(text: str, max_words: int = 220) -> List[str]:
    sentences = _sentences(text)
    if not sentences:
        return []
    engine = os.getenv("CHUNKING_ENGINE", "section_semantic").lower()
    threshold = float(os.getenv("MBERT_BREAK_THRESHOLD", "0.48"))
    vectors = _mbert_vectors(sentences) if engine == "mbert" else None
    out: List[str] = []
    buf: List[str] = []
    words = 0
    for i, sentence in enumerate(sentences):
        sw = len(sentence.split())
        semantic_break = False
        if vectors and i > 0:
            semantic_break = _cosine(vectors[i - 1], vectors[i]) < threshold
        if buf and (words + sw > max_words or semantic_break):
            out.append(" ".join(buf).strip())
            buf = []
            words = 0
        buf.append(sentence)
        words += sw
    if buf:
        out.append(" ".join(buf).strip())
    return out


def _chunk(name: str, page: int, text: str, kind: str) -> List[Chunk]:
    section, buf, out = "Document", [], []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        block = " ".join(buf)
        parts = _semantic_parts(block)
        if not parts:
            words = block.split()
            parts = [" ".join(words[max(0, i - 35): i + 220]).strip() for i in range(0, len(words), 220)]
        for part in parts:
            if part:
                out.append(Chunk(name, part, page, section, kind))
        buf = []

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if kind == "text" and _section(line):
            flush()
            section = line
        else:
            buf.append(line)
    flush()
    if not out:
        out.append(Chunk(name, text[:1200] or f"No extractable text in {name}", page, section, kind))
    return out


def build_corpus(path: Path, max_docs: int = 40, max_pages: int = 20) -> Tuple[List[Dict[str, Any]], str]:
    chunks: List[Chunk] = []
    for name, raw in _members(path)[:max_docs]:
        for page, text, kind in _text(raw, name, max_pages):
            chunks.extend(_chunk(name, page, text, kind))
    rows = [asdict(c) | {"numbers": c.numbers} for c in chunks]
    return rows, f"Indexed {len(rows)} chunks from {path.name}."


def build_corpus_from_paths(paths: List[Path], max_docs: int = 40, max_pages: int = 20) -> Tuple[List[Dict[str, Any]], str]:
    rows, notes = [], []
    for p in paths:
        part, note = build_corpus(p, max_docs, max_pages)
        rows.extend(part)
        notes.append(note)
    return rows, f"Indexed {len(rows)} chunks from {len(paths)} upload(s). " + " ".join(notes)


URL_RE = re.compile(r"https?://[^\s<>'\"`{}|\\^]+", re.IGNORECASE)


def extract_urls_from_text(text: str) -> List[str]:
    """Find valid HTTP(S) URLs in free text without preserving trailing punctuation."""

    urls: List[str] = []
    seen = set()
    for match in URL_RE.finditer(text or ""):
        url = match.group(0).strip().rstrip(".,;:!?)]}>\"'")
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"} and parsed.netloc and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def extract_urls_from_paths(paths: List[Path], max_docs: int = 40, max_pages: int = 20, max_urls: int = 50) -> List[str]:
    """Extract URLs from uploaded files, including ZIP members and OCR/table text."""

    urls: List[str] = []
    seen = set()
    for path in paths:
        try:
            for name, raw in _members(path)[:max_docs]:
                for _, text, _ in _text(raw, name, max_pages):
                    for url in extract_urls_from_text(text):
                        if url in seen:
                            continue
                        seen.add(url)
                        urls.append(url)
                        if len(urls) >= max_urls:
                            return urls
        except Exception:
            continue
    return urls


def youtube_video_id(url: str) -> Optional[str]:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "youtu.be":
        return parsed.path.strip("/").split("/")[0] or None
    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        qs_id = parse_qs(parsed.query).get("v", [""])[0]
        if qs_id:
            return qs_id
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
            return parts[1]
    return None


def _stamp(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _transcript_token_count(text: str) -> int:
    if os.getenv("TRANSCRIPT_USE_BERT_TOKENIZER", "false").lower() == "true":
        try:
            from transformers import BertTokenizer

            tokenizer = BertTokenizer.from_pretrained(os.getenv("TRANSCRIPT_TOKENIZER", "bert-base-uncased"))
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return len(re.findall(r"\w+|[^\w\s]", text or ""))


def fetch_youtube_transcript(url: str) -> Dict[str, Any]:
    video_id = youtube_video_id(url)
    if not video_id:
        return {"ok": False, "url": url, "video_id": "", "segments": [], "note": "Not a supported YouTube URL."}
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        languages = [x.strip() for x in os.getenv("YOUTUBE_TRANSCRIPT_LANGS", "en,hi,ur").split(",") if x.strip()]
        try:
            raw_segments = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        except AttributeError:
            raw_segments = YouTubeTranscriptApi().fetch(video_id, languages=languages)
        segments = []
        for item in raw_segments:
            if isinstance(item, dict):
                text_value = item.get("text", "")
                start_value = item.get("start", 0.0)
                duration_value = item.get("duration", 0.0)
            else:
                text_value = getattr(item, "text", "")
                start_value = getattr(item, "start", 0.0)
                duration_value = getattr(item, "duration", 0.0)
            if text_value:
                segments.append({"text": re.sub(r"\s+", " ", str(text_value)).strip(), "start": float(start_value or 0), "duration": float(duration_value or 0)})
        if not segments:
            return {"ok": False, "url": url, "video_id": video_id, "segments": [], "note": "No public transcript segments were found."}
        return {"ok": True, "url": url, "video_id": video_id, "segments": segments, "note": f"Fetched {len(segments)} public YouTube transcript segments."}
    except Exception as exc:
        return {"ok": False, "url": url, "video_id": video_id, "segments": [], "note": f"YouTube transcript unavailable: {exc}"}


def youtube_transcript_chunks(url: str, segments: List[Dict[str, Any]], max_tokens: int = 512) -> List[Dict[str, Any]]:
    video_id = youtube_video_id(url) or "youtube"
    chunks: List[Dict[str, Any]] = []
    buf: List[str] = []
    tokens = 0
    start = 0.0

    def flush() -> None:
        nonlocal buf, tokens, start
        if not buf:
            return
        seconds = int(start)
        watch = f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"
        text = f"Timestamp: {_stamp(start)} ({seconds}s)\nWatch: {watch}\nTranscript:\n" + " ".join(buf)
        chunk = Chunk(f"YouTube transcript {video_id}", text, max(1, seconds), f"Transcript {_stamp(start)}", "transcript")
        row = asdict(chunk) | {"numbers": chunk.numbers, "video_id": video_id, "start_time": start, "url": watch}
        chunks.append(row)
        buf, tokens, start = [], 0, 0.0

    for segment in segments:
        sentence = str(segment.get("text", "")).strip()
        if not sentence:
            continue
        count = max(1, _transcript_token_count(sentence))
        if buf and tokens + count > max_tokens:
            flush()
        if not buf:
            start = float(segment.get("start", 0.0) or 0.0)
        buf.append(sentence)
        tokens += count
    flush()
    return chunks


def jurisdiction_policy(jurisdiction: str) -> Dict[str, Any]:
    policies = {
        "India": ["DPDP controls", "lawful purpose/consent", "privacy notice", "security safeguards", "data principal rights"],
        "EU/EEA": ["GDPR lawful basis", "purpose limitation", "data minimisation", "storage limitation", "data subject rights"],
        "California": ["CCPA/CPRA notice", "access/delete/correct/opt-out rights", "sensitive data limits"],
        "UK": ["UK GDPR/Data Protection Act principles", "lawful basis", "rights handling"],
        "Global/Unknown": ["robots.txt", "terms review", "copyright review", "personal-data minimisation", "local-law review"],
    }
    return {"jurisdiction": jurisdiction, "checks": policies.get(jurisdiction, policies["Global/Unknown"])}


def robots_allowed(url: str) -> Tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False, "Only http/https URLs are supported."
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(USER_AGENT, url), robots_url
    except Exception as exc:
        return False, f"Could not verify robots.txt: {exc}"


def fetch_url_text(url: str, jurisdiction: str = "Global/Unknown", max_bytes: int = 800_000) -> Dict[str, Any]:
    allowed, note = robots_allowed(url)
    if not allowed:
        return {"ok": False, "url": url, "text": "", "note": f"Blocked by compliance check: {note}"}
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=15) as resp:
            raw = resp.read(max_bytes + 1)
            ctype = resp.headers.get("content-type", "")
        if len(raw) > max_bytes:
            return {"ok": False, "url": url, "text": "", "note": "Blocked: page exceeded size limit."}
        text = _decode(raw)
        if "html" in ctype.lower() or "<html" in text[:1000].lower():
            parser = TextHTMLParser()
            parser.feed(text)
            text = parser.text
        text = redact_personal_data(text) if os.getenv("DPDP_REDACT", "true").lower() == "true" else text
        return {"ok": True, "url": url, "text": text[:120_000], "note": json.dumps(jurisdiction_policy(jurisdiction))}
    except Exception as exc:
        return {"ok": False, "url": url, "text": "", "note": f"Fetch failed: {exc}"}


def build_corpus_from_urls(urls: List[str], jurisdiction: str = "Global/Unknown") -> Tuple[List[Dict[str, Any]], str]:
    rows: List[Dict[str, Any]] = []
    notes = []
    clean_urls: List[str] = []
    seen = set()
    for item in urls:
        candidates = extract_urls_from_text(item) or [item.strip()]
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                clean_urls.append(candidate)
    for url in clean_urls[:20]:
        video_id = youtube_video_id(url)
        if video_id:
            fetched_video = fetch_youtube_transcript(url)
            notes.append(f"{url}: {fetched_video['note']}")
            if fetched_video["ok"]:
                rows.extend(youtube_transcript_chunks(url, fetched_video["segments"], int(os.getenv("TRANSCRIPT_MAX_TOKENS", "512"))))
            continue
        fetched = fetch_url_text(url, jurisdiction)
        notes.append(f"{url}: {fetched['note']}")
        if fetched["ok"]:
            rows.extend(asdict(c) | {"numbers": c.numbers} for c in _chunk(url, 1, fetched["text"], "web"))
    detail = " ".join(notes[:5])
    return rows, f"Indexed {len(rows)} compliant web chunks from {len(clean_urls[:20])} URL(s)." + (f" Notes: {detail[:900]}" if detail else "")


def tavily_search(query: str, max_results: int = 5, topic: str = "general", search_depth: str = "basic") -> Dict[str, Any]:
    """Live search via Tavily, returning source snippets only when TAVILY_API_KEY is set."""

    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"ok": False, "answer": "", "results": [], "note": "TAVILY_API_KEY is not set."}
    try:
        payload = {
            "query": query,
            "topic": topic,
            "search_depth": search_depth,
            "max_results": max(1, min(max_results, 10)),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        req = Request(
            "https://api.tavily.com/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": USER_AGENT},
            method="POST",
        )
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read(2_000_000).decode("utf-8"))
        return {"ok": True, "answer": data.get("answer", ""), "results": data.get("results", []), "note": data.get("request_id", "")}
    except Exception as exc:
        return {"ok": False, "answer": "", "results": [], "note": f"Tavily search failed: {exc}"}


def build_corpus_from_tavily(query: str, max_results: int = 5, topic: str = "general") -> Tuple[List[Dict[str, Any]], str]:
    data = tavily_search(query, max_results=max_results, topic=topic)
    rows: List[Dict[str, Any]] = []
    if not data.get("ok"):
        return rows, data.get("note", "Tavily unavailable.")
    for r in data.get("results", []):
        url = r.get("url", "tavily-result")
        title = r.get("title", "Live search result")
        content = f"{title}\nURL: {url}\n{r.get('content', '')}"
        rows.extend(asdict(c) | {"numbers": c.numbers, "url": url} for c in _chunk(url, 1, content, "live_web"))
    return rows, f"Indexed {len(rows)} Tavily live-search chunks from {len(data.get('results', []))} result(s)."


def ingest_latest_updates(
    query: str,
    corpus_id_value: str = "latest_updates",
    max_results: int = 8,
    jurisdiction: str = "Global/Unknown",
    urls: Optional[List[str]] = None,
    store_postgres: bool = True,
    store_pinecone: bool = True,
) -> Dict[str, Any]:
    """Fetch compliant latest updates and persist to configured vector/database stores."""

    rows: List[Dict[str, Any]] = []
    notes = []
    if urls:
        url_rows, url_note = build_corpus_from_urls(urls, jurisdiction)
        rows.extend(url_rows)
        notes.append(url_note)
    tav_rows, tav_note = build_corpus_from_tavily(query, max_results=max_results)
    rows.extend(tav_rows)
    notes.append(tav_note)
    saved_pg = save_corpus_pg(rows, corpus_id_value) if store_postgres and rows else False
    saved_pc = pinecone_upsert(rows, corpus_id_value) if store_pinecone and rows else False
    return {
        "query": query,
        "corpus_id": corpus_id_value,
        "jurisdiction": jurisdiction,
        "policy": jurisdiction_policy(jurisdiction),
        "notes": notes,
        "chunks": len(rows),
        "postgres_saved": saved_pg,
        "pinecone_saved": saved_pc,
        "guardrails": [
            "Use only Tavily snippets and user-provided URLs that pass compliance checks.",
            "Respect robots.txt, website terms, copyright/database rights, and institutional policies.",
            "Do not bypass paywalls, logins, CAPTCHAs, or access controls.",
            "Redact personal identifiers when enabled.",
            "Treat updates as source evidence requiring human review, not legal advice.",
        ],
        "sources": [{"source": r.get("source"), "section": r.get("section"), "kind": r.get("kind")} for r in rows[:20]],
    }


def needs_live_search(query: str) -> bool:
    return bool(re.search(r"\b(latest|today|current|recent|live|now|new|updated|2026|price|news|guideline|rule|law|model list|free model)\b", query or "", re.I))


AI_POLICY_PROFILES = [
    {
        "name": "ChatGPT / OpenAI",
        "type": "chat_provider",
        "provider": "openai",
        "official_urls": [
            "https://openai.com/policies/",
            "https://openai.com/policies/usage-policies/",
            "https://openai.com/policies/privacy-policy/",
        ],
        "institution_notes": "Use business/enterprise terms, DPA, privacy, usage policy, and local law for government/institutional deployment.",
    },
    {
        "name": "Claude / Anthropic",
        "type": "chat_provider",
        "provider": "claude",
        "official_urls": [
            "https://www.anthropic.com/legal/consumer-terms",
            "https://www.anthropic.com/legal/privacy",
            "https://www.anthropic.com/legal/aup",
            "https://support.anthropic.com/en/collections/4078534-privacy-and-legal",
        ],
        "institution_notes": "Use commercial terms/API or Claude for Work controls for institutional data; review retention, training, and DPA requirements.",
    },
    {
        "name": "Microsoft Copilot",
        "type": "policy_profile",
        "provider": "custom",
        "official_urls": [
            "https://www.microsoft.com/en-us/microsoft-copilot/for-individuals/termsofuse",
            "https://www.microsoft.com/en-us/microsoft-copilot/for-individuals/privacy",
            "https://learn.microsoft.com/en-us/microsoft-365/copilot/enterprise-data-protection",
        ],
        "institution_notes": "Microsoft Copilot consumer and Microsoft 365 Copilot have different terms. For government/institutional use, review Product Terms, DPA, enterprise data protection, tenant controls, and admin policies.",
    },
]


def ai_policy_profiles() -> List[Dict[str, Any]]:
    return [dict(x) for x in AI_POLICY_PROFILES]


def ai_policy_scan(profile_name: str = "All", jurisdiction: str = "Global/Unknown") -> Dict[str, Any]:
    profiles = AI_POLICY_PROFILES if profile_name == "All" else [p for p in AI_POLICY_PROFILES if p["name"] == profile_name]
    rows = []
    for profile in profiles:
        fetched = []
        for url in profile["official_urls"]:
            item = fetch_url_text(url, jurisdiction)
            fetched.append({"url": url, "ok": item.get("ok", False), "note": item.get("note", ""), "excerpt": item.get("text", "")[:1500]})
        rows.append({"profile": profile, "fetched": fetched})
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "jurisdiction": jurisdiction,
        "legal_note": "Policy scan is a technical aid, not legal advice. Use official URLs, current contracts, institutional policy, and qualified counsel.",
        "profiles": rows,
    }


def corpus_id(paths: List[Path], extra_sources: Optional[List[str]] = None) -> str:
    path_bits = [f"{p.name}:{p.stat().st_size if p.exists() else 0}" for p in paths]
    source_bits = [f"url:{src}" for src in (extra_sources or [])]
    raw = "|".join(path_bits + source_bits)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def corpus_metadata(corpus: List[Dict[str, Any]], cid: str = "") -> Dict[str, Any]:
    sources: Dict[str, Dict[str, Any]] = {}
    for c in corpus:
        name = str(c.get("source", "unknown"))
        item = sources.setdefault(name, {"chunks": 0, "pages": set(), "kinds": set(), "sections": set()})
        item["chunks"] += 1
        item["pages"].add(c.get("page", 1))
        item["kinds"].add(c.get("kind", "text"))
        item["sections"].add(c.get("section", "Document"))
    clean_sources = {
        k: {
            "chunks": v["chunks"],
            "pages": sorted(v["pages"]),
            "kinds": sorted(v["kinds"]),
            "sections": sorted(list(v["sections"]))[:30],
        }
        for k, v in sources.items()
    }
    return {
        "corpus_id": cid,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(corpus),
        "source_count": len(sources),
        "sources": clean_sources,
        "selected_provider": os.getenv("LLM_PROVIDER", "local"),
        "ocr_engine": os.getenv("OCR_ENGINE", "tesseract"),
        "ocr_lang": os.getenv("OCR_LANG", "eng"),
        "chunking_engine": os.getenv("CHUNKING_ENGINE", "section_semantic"),
        "mbert_model": os.getenv("MBERT_MODEL", "bert-base-multilingual-cased"),
        "stt_engine": os.getenv("STT_ENGINE", "manual"),
        "transliteration_engine": os.getenv("TRANSLITERATION_ENGINE", "auto_llm"),
        "response_language": os.getenv("RESPONSE_LANGUAGE", "auto"),
        "compliance_jurisdiction": os.getenv("COMPLIANCE_JURISDICTION", "Global/Unknown"),
        "human_review_confirmed": os.getenv("HUMAN_REVIEW_CONFIRMED", "false"),
        "export_approval_required": os.getenv("REQUIRE_HUMAN_EXPORT_APPROVAL", "true"),
        "keys_exported": "never",
    }


def encrypt_secret_label(value: str) -> str:
    """One-way-ish display helper: hide secrets while proving a value exists."""

    if not value:
        return ""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return "set:" + base64.urlsafe_b64encode(digest[:9]).decode("ascii").rstrip("=")


def pg_conn() -> Any:
    url = os.getenv(DATABASE_URL)
    if not url:
        return None
    try:
        import psycopg

        return psycopg.connect(url)
    except Exception:
        return None


def pg_init(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            create table if not exists rag_chunks (
                corpus_id text not null,
                idx integer not null,
                source text,
                page integer,
                section text,
                kind text,
                score double precision default 0,
                text text,
                numbers jsonb,
                primary key (corpus_id, idx)
            )
            """
        )
        cur.execute(
            """
            create table if not exists rag_queries (
                id bigserial primary key,
                corpus_id text,
                question text,
                answer text,
                provider text,
                model text,
                created_at timestamptz default now()
            )
            """
        )
        cur.execute(
            """
            create table if not exists rag_integrations (
                name text primary key,
                category text,
                pricing text,
                use_case text,
                base_url text,
                model text,
                api_key_env text,
                score double precision default 0,
                updated_at timestamptz default now()
            )
            """
        )
    conn.commit()


def save_corpus_pg(corpus: List[Dict[str, Any]], cid: str) -> bool:
    conn = pg_conn()
    if conn is None:
        return False
    try:
        pg_init(conn)
        with conn.cursor() as cur:
            cur.execute("delete from rag_chunks where corpus_id = %s", (cid,))
            for i, c in enumerate(corpus):
                cur.execute(
                    """
                    insert into rag_chunks (corpus_id, idx, source, page, section, kind, score, text, numbers)
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        cid,
                        i,
                        c.get("source"),
                        c.get("page"),
                        c.get("section"),
                        c.get("kind"),
                        c.get("score", 0.0),
                        c.get("text"),
                        json.dumps(c.get("numbers", [])),
                    ),
                )
        conn.commit()
        conn.close()
        return True
    except Exception:
        conn.rollback()
        conn.close()
        return False


def log_query_pg(cid: str, question: str, answer: str, provider: str, model: str) -> bool:
    conn = pg_conn()
    if conn is None:
        return False


def upsert_integrations_pg(items: List[Dict[str, str]]) -> bool:
    conn = pg_conn()
    if conn is None:
        return False
    try:
        pg_init(conn)
        with conn.cursor() as cur:
            for item in items:
                cur.execute(
                    """
                    insert into rag_integrations (name, category, pricing, use_case, base_url, model, api_key_env, score)
                    values (%s,%s,%s,%s,%s,%s,%s,%s)
                    on conflict (name) do update set
                        category = excluded.category,
                        pricing = excluded.pricing,
                        use_case = excluded.use_case,
                        base_url = excluded.base_url,
                        model = excluded.model,
                        api_key_env = excluded.api_key_env,
                        score = excluded.score,
                        updated_at = now()
                    """,
                    (
                        item.get("name"),
                        item.get("category", "custom"),
                        item.get("pricing", "unknown"),
                        item.get("use", item.get("use_case", "")),
                        item.get("base_url", ""),
                        item.get("model", ""),
                        item.get("api_key_env", ""),
                        float(item.get("score", 0) or 0),
                    ),
                )
        conn.commit()
        conn.close()
        return True
    except Exception:
        conn.rollback()
        conn.close()
        return False


def load_integrations_pg() -> List[Dict[str, str]]:
    conn = pg_conn()
    if conn is None:
        return []
    try:
        pg_init(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                select name, category, pricing, use_case, base_url, model, api_key_env, score
                from rag_integrations
                order by score desc, updated_at desc, name asc
                limit 200
                """
            )
            rows = cur.fetchall()
        conn.close()
        return [
            {
                "name": r[0],
                "category": r[1] or "",
                "pricing": r[2] or "",
                "use": r[3] or "",
                "base_url": r[4] or "",
                "model": r[5] or "",
                "api_key_env": r[6] or "",
                "score": str(r[7] or 0),
            }
            for r in rows
        ]
    except Exception:
        conn.close()
        return []
    try:
        pg_init(conn)
        with conn.cursor() as cur:
            cur.execute(
                "insert into rag_queries (corpus_id, question, answer, provider, model) values (%s,%s,%s,%s,%s)",
                (cid, question, answer, provider, model),
            )
        conn.commit()
        conn.close()
        return True
    except Exception:
        conn.rollback()
        conn.close()
        return False


def retrieve(corpus: List[Dict[str, Any]], query: str, k: int = 8) -> List[Dict[str, Any]]:
    if not corpus:
        return []
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [f"{c['source']} {c['section']} {c['kind']} {c['text']}" for c in corpus]
        mat = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(texts + [query])
        scores = (mat[:-1] @ mat[-1].T).toarray().ravel()
        qnums = set(re.findall(r"\d+(?:\.\d+)?", query))
        science_terms = set(re.findall(r"\b(?:pdb|rmsd|angstrom|Å|resolution|domain|residue|mutation|assay|binding|affinity|ic50|ec50|kd|ph|cryo-em|x-ray|structure|glycoprotein|protein|genome|sequence|table|figure)\b", query.lower()))
        for i, c in enumerate(corpus):
            scores[i] += 0.15 * len(qnums & set(re.findall(r"\d+(?:\.\d+)?", c["text"])))
            scores[i] += 0.10 if c.get("kind") == "table" and ("table" in query.lower() or qnums) else 0
            scores[i] += 0.05 * len(science_terms & set(re.findall(r"\w+", (c["section"] + " " + c["text"]).lower())))
        order = np.argsort(scores)[::-1][:k]
        return [dict(corpus[int(i)], score=float(scores[int(i)])) for i in order]
    except Exception:
        terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for c in corpus:
            score = len(terms & set(re.findall(r"\w+", c["text"].lower())))
            scored.append(dict(c, score=float(score)))
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:k]


def ask_suggestions(corpus: List[Dict[str, Any]], n: int = 8) -> List[str]:
    """Generate simple grounded question suggestions from source sections and numeric evidence."""

    suggestions = []
    seen = set()
    for c in corpus:
        section = str(c.get("section", "Document"))
        source = str(c.get("source", "source"))
        kind = str(c.get("kind", "text"))
        nums = c.get("numbers") or []
        candidates = [
            f"What are the key findings in `{source}` section `{section}`?",
            f"What evidence supports the main claim in `{source}`?",
            f"What limitations or missing evidence are visible in `{source}`?",
        ]
        if kind == "table" or nums:
            candidates.append(f"Compare the numerical values reported in `{source}` and explain their units.")
        if re.search(r"\b(method|assay|experiment|protocol|procedure)\b", c.get("text", ""), re.I):
            candidates.append(f"What methods or experimental procedures are described in `{source}`?")
        if re.search(r"\b(figure|image|diagram|structure|table)\b", c.get("text", ""), re.I):
            candidates.append(f"What figures, tables, structures, or visual evidence are described in `{source}`?")
        for q in candidates:
            if q not in seen:
                suggestions.append(q)
                seen.add(q)
            if len(suggestions) >= n:
                return suggestions
    return suggestions or ["Summarize the uploaded evidence with citations.", "What is not found in the uploaded documents?"]


def vector_space_knowledge(corpus: List[Dict[str, Any]], query: str = "entire corpus", k: int = 25) -> Dict[str, Any]:
    """Expose a broad, auditable view of the indexed vector/lexical evidence space."""

    hits = retrieve(corpus, query or "entire corpus", min(k, max(1, len(corpus))))
    sections: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    numbers: List[str] = []
    for c in corpus:
        sections[str(c.get("section", "Document"))] = sections.get(str(c.get("section", "Document")), 0) + 1
        sources[str(c.get("source", "source"))] = sources.get(str(c.get("source", "source")), 0) + 1
        numbers.extend(c.get("numbers") or [])
    return {
        "summary": {
            "chunks": len(corpus),
            "sources": sources,
            "sections": sections,
            "sample_numbers": numbers[:60],
        },
        "top_evidence": hits,
        "suggested_questions": ask_suggestions(corpus),
    }


def agent_metrics_session(
    query: str,
    corpus: List[Dict[str, Any]],
    provider: str = "local",
    retrieval_engine: str = "TF-IDF",
    jurisdiction: str = "India",
) -> Dict[str, Any]:
    """Course-inspired metrics view for RAG, API, feedback, and MCP readiness."""

    hits = retrieve(corpus, query or "metrics rag api mcp", 10)
    source_count = len({str(c.get("source", "")) for c in corpus if c.get("source")})
    table_chunks = sum(1 for c in corpus if c.get("kind") == "table")
    media_chunks = sum(1 for c in corpus if str(c.get("kind", "")) in {"image", "ocr", "media"})
    numeric_chunks = sum(1 for c in corpus if c.get("numbers"))
    total_chars = sum(len(str(c.get("text", ""))) for c in corpus)
    top_score = round(float(hits[0].get("score", 0.0)), 4) if hits else 0.0
    citation_ready = bool(hits and source_count)
    table_aware = table_chunks > 0 or any("|" in str(h.get("text", "")) for h in hits)
    has_api_keys = any(os.getenv(k) for k in ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN", "GOOGLE_API_KEY", "GROK_API_KEY", "ANTHROPIC_API_KEY"])
    has_storage = bool(os.getenv("DATABASE_URL") or os.getenv("PINECONE_API_KEY") or os.getenv("SUPABASE_URL"))
    metrics = [
        {"metric": "Corpus chunks", "value": len(corpus), "target": ">= 1 for document RAG", "status": "ok" if corpus else "needs evidence"},
        {"metric": "Unique sources", "value": source_count, "target": ">= 1 cited source", "status": "ok" if source_count else "needs sources"},
        {"metric": "Top retrieval score", "value": top_score, "target": "higher is better", "status": "ok" if top_score > 0 else "review"},
        {"metric": "Numeric chunks", "value": numeric_chunks, "target": "detect numerical evidence", "status": "ok" if numeric_chunks else "not found"},
        {"metric": "Table chunks", "value": table_chunks, "target": "preserve structure", "status": "ok" if table_aware else "not found"},
        {"metric": "Media/OCR chunks", "value": media_chunks, "target": "image/pdf evidence support", "status": "ok" if media_chunks else "optional"},
        {"metric": "Average chunk chars", "value": round(total_chars / max(len(corpus), 1), 1), "target": "section-aware, not arbitrary 1000-char split", "status": "ok"},
        {"metric": "API key readiness", "value": "configured" if has_api_keys else "local only", "target": "optional external LLM/API integration", "status": "ok" if has_api_keys else "local"},
        {"metric": "Vector/storage readiness", "value": "configured" if has_storage else "local only", "target": "Postgres/Pinecone/Supabase", "status": "ok" if has_storage else "local"},
    ]
    gates = [
        {"gate": "Grounding", "check": "Every model claim must cite retrieved source/page/section.", "status": "ready" if citation_ready else "blocked until evidence is uploaded"},
        {"gate": "Tables and numbers", "check": "Preserve tables, units, denominators, totals, and numerical context.", "status": "ready" if numeric_chunks or table_aware else "watch"},
        {"gate": "Planner -> executor -> verifier", "check": "Route complex queries through agent pipeline and verification.", "status": "ready"},
        {"gate": "Feedback loop", "check": "Capture thumbs up/down and comments for later eval tuning.", "status": "ready through monitoring.py"},
        {"gate": "API integration", "check": "Keep provider keys secret-backed and route calls through approved adapters.", "status": "ready" if has_api_keys else "local-only mode"},
        {"gate": "MCP server readiness", "check": "Expose typed tools/resources after metrics prove stable behavior.", "status": "planned"},
        {"gate": "Human authority", "check": "Human remains final approver for exports, cloud use, and external actions.", "status": "required"},
    ]
    api_plan = [
        "Keep the Streamlit chat as the human-facing shell; keep orchestration inside the pipeline.",
        "Expose one internal API boundary per capability: retrieve, answer, quiz, website, visual map, compliance, feedback.",
        "Use provider adapters for OpenAI/OpenRouter/Hugging Face/Gemini/Grok/Ollama/custom endpoints without showing stored keys.",
        "Log latency, source count, retrieval mode, top score, table/numeric hit counts, and user feedback.",
        "Run LangSmith/eval datasets only when keys are configured; otherwise keep JSONL metrics local.",
    ]
    mcp_plan = [
        {"tool": "retrieve_evidence", "input": "query, top_k, corpus_id", "output": "ranked chunks with citations"},
        {"tool": "answer_grounded", "input": "query, provider, retrieval_mode", "output": "answer, citations, limitations"},
        {"tool": "create_quiz", "input": "exam, topic, difficulty, count", "output": "items, answer key, remarks"},
        {"tool": "build_website", "input": "brief, brand, evidence_ids", "output": "html, seo, critic notes"},
        {"tool": "visual_map", "input": "query, style, top_k", "output": "svg, mermaid, outline"},
        {"tool": "log_feedback", "input": "question, answer, rating, comment", "output": "stored feedback event"},
        {"resource": "corpus_metadata", "input": "corpus_id", "output": "sources, sections, chunks, privacy metadata"},
    ]
    evidence = [
        {
            "source": h.get("source"),
            "page": h.get("page"),
            "section": h.get("section"),
            "kind": h.get("kind"),
            "score": round(float(h.get("score", 0.0)), 4),
            "snippet": str(h.get("text", ""))[:280],
        }
        for h in hits[:6]
    ]
    markdown = (
        "# Metrics, RAG/API, and MCP Readiness\n\n"
        "Sameer's session insight is implemented as a practical loop: measure retrieval and answer quality first, "
        "then integrate agents with RAG/APIs, then expose stable capabilities through an MCP server.\n\n"
        "## Immediate Priority\n\n"
        "1. Metrics are the operating dashboard for the whole course project.\n"
        "2. RAG + API integration must stay grounded, secret-safe, and observable.\n"
        "3. MCP server work should expose only stable, typed tools after feedback and eval checks.\n\n"
        f"**Provider:** {provider}  \n"
        f"**Retrieval:** {retrieval_engine}  \n"
        f"**Jurisdiction:** {jurisdiction}\n"
    )
    return {
        "markdown": markdown,
        "metrics": metrics,
        "quality_gates": gates,
        "rag_api_plan": api_plan,
        "mcp_server_plan": mcp_plan,
        "top_evidence": evidence,
    }


def orchestration_manager_plan(
    query: str,
    corpus: List[Dict[str, Any]],
    provider: str = "local",
    retrieval_engine: str = "TF-IDF",
    live_search_enabled: bool = False,
    jurisdiction: str = "India",
) -> Dict[str, Any]:
    """Route a user query to the smallest useful agent/tool chain."""

    q = (query or "").lower()
    has_docs = bool(corpus)
    has_web = any(str(c.get("kind", "")).startswith("web") or str(c.get("kind", "")) == "live_web" for c in corpus)
    has_media = any(str(c.get("kind", "")) in {"image", "ocr", "media"} for c in corpus)
    selected_action = "Agent chat"
    confidence = 0.62
    rationale = "General evidence-grounded request; use planner, retriever, executor, verifier."

    rules = [
        ("Metrics", ["metrics", "metric", "eval", "evaluation", "observability", "monitoring", "feedback loop", "langsmith", "wandb", "evidently", "rag api", "rag and api", "mcp", "mcp server", "next session", "course"], "Metrics/course insight detected; inspect RAG quality, API readiness, feedback loop, and MCP server plan."),
        ("School clerk", ["school clerk", "clerk", "result", "marksheet", "mark sheet", "report card", "attendance", "fee reminder", "bonafide", "transfer certificate", "tc", "admission register", "roll list"], "School-office automation intent detected; use clerk workflow with result generation and human review."),
        ("Study quiz", ["quiz", "exam", "question paper", "mcq", "flashcard", "physics wallah", "textbook", "student"], "Study/exam intent detected; generate grounded learning items."),
        ("Visual maps", ["mindmap", "mind map", "flowchart", "flow chart", "concept map", "visual", "diagram", "graphic"], "Visual explanation requested; create evidence maps and Mermaid/SVG outputs."),
        ("Website", ["website", "landing page", "seo", "web page", "site builder", "html"], "Website-building intent detected; use website builder, critic, SEO, and evidence."),
        ("App blueprint", ["create app", "build app", "app as per prompt", "application blueprint", "emergent"], "App-generation intent detected; produce an implementation blueprint."),
        ("WhatsApp automation", ["whatsapp", "wa automation", "broadcast", "message campaign"], "WhatsApp/service outreach intent detected; draft compliant automation assets."),
        ("Voiceover", ["voiceover", "audio", "tts", "speech", "mp3", "narration"], "Audio-generation or narration intent detected; prepare safe voiceover guidance."),
        ("Marketing", ["marketing", "campaign", "promotion", "ad copy", "social media", "lead"], "Marketing intent detected; prepare grounded campaign plan."),
        ("Media inventory", ["media inventory", "image inventory", "asset", "gallery"], "Media-management intent detected; inspect uploaded media and metadata."),
        ("AI policy scan", ["policy", "chatgpt", "claude", "copilot", "terms", "legal norms"], "AI policy/compliance scan requested."),
        ("Compliance", ["dpdp", "privacy", "compliance", "lawful", "consent", "guideline", "government rule"], "Compliance/legal guardrail intent detected."),
        ("Ingest latest updates", ["ingest latest", "store latest", "update vector", "latest update into"], "Latest-update ingestion intent detected."),
        ("Live search", ["latest", "current", "today", "live search", "recent", "new update"], "Fresh information requested; use live search when the toggle and key are configured."),
        ("Vector knowledge", ["vector space", "knowledge graph", "all evidence", "scrap vector", "scrape vector"], "Vector-space exploration requested."),
        ("Ask suggestions", ["suggest question", "try asking", "what can i ask"], "Suggestion intent detected."),
        ("Swarm", ["swarm", "orchestrator", "agent promotion", "agent demotion", "topology"], "Agent governance/topology intent detected."),
    ]
    for action, keywords, why in rules:
        if any(k in q for k in keywords):
            selected_action = action
            rationale = why
            confidence = 0.88
            break

    routing_mode = "rule-based"
    llm_route = _llm_select_workflow(
        query=query,
        actions=[r[0] for r in rules] + ["Agent chat", "Chat"],
        evidence_state={
            "has_documents": has_docs,
            "has_url_or_live_evidence": has_web,
            "has_media_or_ocr": has_media,
            "live_search_enabled": live_search_enabled,
        },
        provider_hint=provider,
    )
    if llm_route:
        selected_action = llm_route["selected_action"]
        rationale = llm_route["rationale"]
        confidence = llm_route["confidence"]
        routing_mode = "llm-assisted"

    if selected_action == "Live search" and not live_search_enabled:
        selected_action = "Agent chat" if has_docs else "Chat"
        rationale += " Live search is disabled, so the manager falls back to grounded chat."
        confidence = 0.72
    if selected_action in {"Agent chat", "Chat"} and not has_docs and live_search_enabled and needs_live_search(query):
        selected_action = "Live search"
        confidence = 0.8

    agents = [
        {"agent": "human_supervisor", "role": "approval, policy, final authority", "rank": 0},
        {"agent": "smart_router", "role": "classify intent and select tools", "rank": 1},
        {"agent": "planner", "role": "break query into tool steps", "rank": 2},
        {"agent": "retriever", "role": "retrieve uploaded/web/vector evidence", "rank": 3},
        {"agent": "school_clerk", "role": "school office/result workflow when selected", "rank": 4},
        {"agent": "executor", "role": f"run {selected_action}", "rank": 5},
        {"agent": "verifier", "role": "check grounding, citations, and missing evidence", "rank": 6},
        {"agent": "compliance_guard", "role": f"apply {jurisdiction} privacy/legal controls", "rank": 7},
    ]
    tools = [
        {"tool": "mic_or_text_query", "selected": bool(query), "why": "User query enters through text or transcribed mic."},
        {"tool": "document_ingestion", "selected": has_docs, "why": "Uploaded files/ZIP/PDF/images/spreadsheets form the evidence base."},
        {"tool": "url_ingestion", "selected": has_web, "why": "Permitted URLs/live snippets are present in the corpus."},
        {"tool": "retrieval", "selected": has_docs or has_web, "why": f"Using {retrieval_engine} to ground the response."},
        {"tool": "llm_provider", "selected": provider != "local", "why": f"Selected provider: {provider}."},
        {"tool": selected_action, "selected": True, "why": rationale},
        {"tool": "human_review", "selected": True, "why": "Human remains above every agent and approves exports/actions."},
    ]
    return {
        "selected_action": selected_action,
        "confidence": confidence,
        "rationale": rationale,
        "query_channel": "mic/text",
        "evidence_state": {
            "chunks": len(corpus),
            "has_documents": has_docs,
            "has_url_or_live_evidence": has_web,
            "has_media_or_ocr": has_media,
            "live_search_enabled": live_search_enabled,
        },
        "agents": agents,
        "tools": tools,
        "provider": provider,
        "retrieval_engine": retrieval_engine,
        "jurisdiction": jurisdiction,
        "routing_mode": routing_mode,
    }


def mermaid_mindmap(corpus: List[Dict[str, Any]], query: str = "Study Mindmap", k: int = 12) -> str:
    hits = retrieve(corpus, query or "mindmap", k)
    root = re.sub(r"[^A-Za-z0-9 _-]", "", query or "Evidence Mindmap").strip() or "Evidence Mindmap"
    lines = ["mindmap", f"  root(({root[:60]}))"]
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for h in hits:
        by_source.setdefault(str(h.get("source", "Source")), []).append(h)
    for source, rows in list(by_source.items())[:6]:
        clean_source = re.sub(r"[^A-Za-z0-9 _.-]", "", source)[:50] or "Source"
        lines.append(f"    {clean_source}")
        for h in rows[:4]:
            section = re.sub(r"[^A-Za-z0-9 _.-]", "", str(h.get("section", "Section")))[:44] or "Section"
            snippet = re.sub(r"[^A-Za-z0-9 _.-]", "", str(h.get("text", ""))[:70]).strip() or "Evidence"
            lines.append(f"      {section}")
            lines.append(f"        {snippet}")
    return "\n".join(lines)


def _diagram_label(value: Any, limit: int = 72) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"[\[\]{}<>|`]", " ", text).replace('"', "'")
    return (text[: limit - 3] + "...") if len(text) > limit else text


def mermaid_flowchart(corpus: List[Dict[str, Any]], query: str = "Evidence Flowchart", k: int = 10) -> str:
    hits = retrieve(corpus, query or "flowchart", k)
    if not hits:
        return 'flowchart TD\n  A["No uploaded evidence"]'
    lines = ["flowchart TD", f'  Q["{_diagram_label(query or "Question", 60)}"]']
    for i, h in enumerate(hits[:k], start=1):
        source = _diagram_label(h.get("source", "Source"), 48)
        section = _diagram_label(h.get("section", "Section"), 48)
        snippet = _diagram_label(h.get("text", "Evidence"), 70)
        lines.extend(
            [
                f'  S{i}["{source}"]',
                f'  C{i}["{section}"]',
                f'  E{i}["{snippet}"]',
                f"  Q --> S{i}",
                f"  S{i} --> C{i}",
                f"  C{i} --> E{i}",
            ]
        )
    return "\n".join(lines)


def mermaid_concept_map(corpus: List[Dict[str, Any]], query: str = "Concept Map", k: int = 12) -> str:
    hits = retrieve(corpus, query or "concept map", k)
    if not hits:
        return 'graph LR\n  A["No uploaded evidence"]'
    lines = ["graph LR", f'  Q(("{_diagram_label(query or "Central question", 54)}"))']
    sections: Dict[str, List[Dict[str, Any]]] = {}
    for h in hits:
        sections.setdefault(str(h.get("section", "Document")), []).append(h)
    for i, (section, rows) in enumerate(list(sections.items())[:7], start=1):
        lines.append(f'  T{i}["{_diagram_label(section, 46)}"]')
        lines.append(f"  Q --- T{i}")
        for j, h in enumerate(rows[:3], start=1):
            node = f"N{i}_{j}"
            cite = f"{h.get('source', 'source')} p.{h.get('page', 1)}"
            label = _diagram_label(f"{h.get('text', '')} ({cite})", 74)
            lines.append(f'  {node}["{label}"]')
            lines.append(f"  T{i} --- {node}")
    return "\n".join(lines)


def evidence_graph_svg(corpus: List[Dict[str, Any]], query: str = "Evidence Graphic", k: int = 10) -> str:
    hits = retrieve(corpus, query or "evidence graphic", k)
    if not hits:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="260" viewBox="0 0 900 260">'
            '<rect width="900" height="260" fill="#f8fafc"/>'
            '<text x="450" y="130" text-anchor="middle" font-family="Arial" font-size="22" fill="#334155">'
            "No uploaded evidence available</text></svg>"
        )

    width = 1100
    row_h = 94
    height = max(440, 170 + row_h * min(len(hits), k))
    palette = ["#0f766e", "#2563eb", "#a16207", "#be123c", "#7c3aed", "#047857"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="18" fill="#f8fafc"/>',
        '<rect x="360" y="24" width="380" height="72" rx="14" fill="#111827"/>',
        f'<text x="550" y="55" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="#ffffff">{escape(_diagram_label(query or "Evidence Map", 54))}</text>',
        '<text x="550" y="78" text-anchor="middle" font-family="Arial" font-size="12" fill="#cbd5e1">Grounded visual map from uploaded/permitted evidence</text>',
    ]

    for i, h in enumerate(hits[:k], start=1):
        y = 128 + (i - 1) * row_h
        color = palette[(i - 1) % len(palette)]
        source = _diagram_label(f"{h.get('source', 'source')} p.{h.get('page', 1)}", 42)
        section = _diagram_label(h.get("section", "Section"), 44)
        snippet = _diagram_label(h.get("text", "Evidence"), 120)
        parts.extend(
            [
                f'<line x1="550" y1="96" x2="550" y2="{y + 32}" stroke="#cbd5e1" stroke-width="2"/>',
                f'<line x1="550" y1="{y + 32}" x2="214" y2="{y + 32}" stroke="#cbd5e1" stroke-width="2"/>',
                f'<circle cx="214" cy="{y + 32}" r="14" fill="{color}"/>',
                f'<text x="214" y="{y + 37}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700" fill="#ffffff">{i}</text>',
                f'<rect x="242" y="{y}" width="760" height="68" rx="12" fill="#ffffff" stroke="#e2e8f0"/>',
                f'<rect x="242" y="{y}" width="7" height="68" rx="4" fill="{color}"/>',
                f'<text x="268" y="{y + 24}" font-family="Arial" font-size="14" font-weight="700" fill="#0f172a">{escape(section)}</text>',
                f'<text x="760" y="{y + 24}" text-anchor="end" font-family="Arial" font-size="12" fill="#64748b">{escape(source)}</text>',
            ]
        )
        for line_i, wrapped in enumerate(textwrap.wrap(snippet, width=104)[:2]):
            parts.append(
                f'<text x="268" y="{y + 46 + (line_i * 16)}" font-family="Arial" font-size="12" fill="#334155">{escape(wrapped)}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def visual_map_pack(
    corpus: List[Dict[str, Any]],
    query: str = "Evidence Visual Map",
    style: str = "NotebookLM mindmap",
    k: int = 12,
) -> Dict[str, Any]:
    if style == "Flowchart":
        mermaid = mermaid_flowchart(corpus, query, k)
    elif style == "Concept map":
        mermaid = mermaid_concept_map(corpus, query, k)
    else:
        mermaid = mermaid_mindmap(corpus, query, k)
    hits = retrieve(corpus, query or "visual map", k)
    outline = [
        {
            "source": h.get("source"),
            "page": h.get("page"),
            "section": h.get("section"),
            "evidence": _diagram_label(h.get("text", ""), 180),
        }
        for h in hits
    ]
    return {
        "style": style,
        "mermaid": mermaid,
        "svg": evidence_graph_svg(corpus, query, k),
        "outline": outline,
        "note": "NotebookLM/Google-LM-style map is generated only from retrieved uploaded or permitted evidence.",
    }


def study_quiz_generator(
    corpus: List[Dict[str, Any]],
    exam: str,
    topic: str,
    count: int = 10,
    difficulty: str = "medium",
    mode: str = "question_paper",
) -> str:
    """NotebookLM/PW/textbook-style grounded quiz and question-paper generator."""

    hits = retrieve(corpus, f"{exam} {topic} {difficulty}", min(max(count, 5), 25))
    evidence = "\n".join(f"- {h['source']} p.{h['page']} [{h['section']}]: {h['text'][:260]}" for h in hits)
    if not hits:
        return "# Study Generator\n\nNot found in uploaded documents. Upload syllabus, notes, textbook chapters, or previous papers first."
    questions = []
    weak_topics: Dict[str, int] = {}
    for i, h in enumerate(hits[:count], start=1):
        stem = re.sub(r"\s+", " ", h["text"])[:180]
        cite = f"`{h['source']}` p.{h['page']} [{h['section']}]"
        weak_topics[h.get("section", "Document")] = weak_topics.get(h.get("section", "Document"), 0) + 1
        if mode == "flashcards":
            questions.append(f"**Card {i}**\n\nFront: What should a student remember from {cite}?\n\nBack: {stem}\n")
        elif mode == "pw_practice":
            questions.append(
                f"**Q{i}. Single Correct MCQ**\n\n"
                f"Question: Which option is directly supported by {cite}?\n\n"
                f"A. {stem}\nB. A claim not stated in the uploaded source\nC. A formula/result from outside the document\nD. Cannot be determined from any source\n\n"
                f"**Correct Answer:** A\n\n"
                f"**Why:** Option A is copied from the cited evidence. Options B-D are distractors requiring unsupported inference.\n\n"
                f"**PW-style feedback:** Revise `{h.get('section', 'Document')}` and underline exact words/numbers in the source before answering.\n"
            )
        elif mode == "textbook_solution":
            questions.append(
                f"**Problem {i}. Textbook-style worked solution**\n\n"
                f"**Given from source:** {stem}\n\n"
                f"**Step 1:** Identify the known fact/value/method from {cite}.\n\n"
                f"**Step 2:** Restate the concept without adding outside assumptions.\n\n"
                f"**Step 3:** Final answer must cite {cite}.\n\n"
                f"**Common mistake:** Do not use a formula, value, or theorem unless it appears in uploaded evidence.\n"
            )
        elif mode == "assertion_reason":
            questions.append(
                f"**Q{i}. Assertion-Reason**\n\n"
                f"Assertion (A): {stem}\n\n"
                f"Reason (R): This is supported by {cite}.\n\n"
                "Choose: (1) A and R true, R explains A (2) A and R true, R does not explain A (3) A true, R false (4) A false.\n\n"
                "**Answer:** 1, if the student cites the source exactly.\n"
            )
        elif mode == "quiz":
            questions.append(
                f"**Q{i}.** Based on {cite}, which statement is best supported?\n\n"
                f"A. {stem}\nB. Not found in uploaded documents\nC. Unsupported inference\nD. Outside syllabus claim\n\n"
                f"**Answer:** A\n**Explanation:** Supported by {cite}.\n"
            )
        else:
            marks = 1 if difficulty == "easy" else 3 if difficulty == "medium" else 5
            questions.append(
                f"**Q{i}. ({marks} marks)** Explain the following using only the cited source: {stem}\n\n"
                f"**Source:** {cite}\n**Expected answer points:** cite the source, preserve terms/numbers, avoid unsupported claims.\n"
            )
    return (
        f"# {exam} {mode.replace('_', ' ').title()}\n\n"
        f"**Topic:** {topic or 'Uploaded document corpus'}\n\n"
        f"**Difficulty:** {difficulty}\n\n"
        f"**Questions:** {len(questions)}\n\n"
        "## Exam-Prep Style\n\n"
        "- Physics Wallah-style: fast MCQ practice, feedback, weak-topic revision.\n"
        "- Textbook-style: step-by-step source-backed explanations.\n"
        "- NotebookLM-style: generated from uploaded material only.\n\n"
        "## Student Instructions\n\n"
        "- Answer only from the uploaded source material.\n"
        "- Cite the provided source reference in your answer.\n"
        "- If evidence is missing, write: Not found in uploaded documents.\n\n"
        "## Questions\n\n"
        + "\n".join(questions)
        + "\n## Weak Topic Signals\n\n"
        + "\n".join(f"- {k}: {v} generated item(s)" for k, v in sorted(weak_topics.items(), key=lambda x: x[1], reverse=True))
        + "\n\n## Teacher / Student Pro Tips\n\n"
        "- Convert wrong answers into flashcards.\n"
        "- Reattempt weak sections after 24 hours and 7 days.\n"
        "- For numericals, write given, required, formula/source, substitution, final unit.\n"
        "- For theory, answer in points and cite exact document section.\n"
        + "\n## Evidence Basis\n\n"
        + evidence
    )


def study_quiz_items(
    corpus: List[Dict[str, Any]],
    exam: str,
    topic: str,
    count: int = 10,
    difficulty: str = "medium",
    mode: str = "quiz",
) -> Dict[str, Any]:
    """Return structured MCQ items for the Streamlit live-exam UI."""

    hits = retrieve(corpus, f"{exam} {topic} {difficulty}", min(max(count, 5), 25))
    if not hits:
        return {
            "title": f"{exam} Live Exam",
            "items": [],
            "weak_topics": {},
            "message": "Not found in uploaded documents. Upload syllabus, notes, textbook chapters, or previous papers first.",
        }

    points = 1 if difficulty == "easy" else 2 if difficulty == "medium" else 4
    weak_topics: Dict[str, int] = {}
    items: List[Dict[str, Any]] = []

    for i, h in enumerate(hits[:count], start=1):
        source = str(h.get("source", "uploaded source"))
        page = h.get("page", 1)
        section = str(h.get("section", "Document"))
        cite = f"{source} p.{page} [{section}]"
        stem = re.sub(r"\s+", " ", str(h.get("text", ""))).strip()
        stem = stem[:260] if stem else "The cited source contains the supported statement."
        weak_topics[section] = weak_topics.get(section, 0) + 1

        if mode == "assertion_reason":
            question = f"Assertion-Reason from {cite}: Assertion (A): {stem}"
            options = [
                "A and R are true, and R explains A.",
                "A and R are true, but R does not explain A.",
                "A is true, but R is false.",
                "A is false according to the uploaded evidence.",
            ]
            correct = options[0]
            explanation = f"The assertion is copied from the cited evidence, and the reason is its explicit source: {cite}."
            feedback_map = {
                options[0]: "Correct: the assertion is grounded in the cited text, and the reason identifies the supporting source.",
                options[1]: "Incorrect here: the reason is not separate from the evidence; it directly explains why the assertion is accepted.",
                options[2]: "Incorrect here: the reason is the cited uploaded source, so it is not false.",
                options[3]: "Incorrect here: the assertion is taken from uploaded evidence, so it should not be marked false.",
            }
        else:
            question = f"Based only on {cite}, which statement is directly supported?"
            options = [
                stem,
                "Not found in uploaded documents.",
                "An outside-syllabus claim that needs external evidence.",
                "A conclusion that cannot be verified from the cited source.",
            ]
            correct = options[0]
            explanation = f"The correct option is supported by the uploaded evidence at {cite}."
            feedback_map = {
                options[0]: f"Correct: this statement is grounded directly in {cite}.",
                options[1]: "Incorrect here: the uploaded documents do contain the cited evidence for the correct option.",
                options[2]: "Incorrect: this option requires external evidence and is outside the uploaded source boundary.",
                options[3]: "Incorrect: this is a distractor because the cited source verifies the correct option.",
            }

        seed = hashlib.sha256(f"{exam}|{topic}|{source}|{page}|{section}|{i}".encode("utf-8")).hexdigest()
        order = sorted(range(len(options)), key=lambda idx: hashlib.sha256(f"{seed}|{idx}".encode("utf-8")).hexdigest())
        shuffled = [options[idx] for idx in order]
        correct_index = shuffled.index(correct)
        option_feedback = [feedback_map.get(option, "Review this option against the cited uploaded evidence.") for option in shuffled]

        items.append(
            {
                "id": hashlib.sha256(f"{seed}|item".encode("utf-8")).hexdigest()[:16],
                "number": i,
                "question": question,
                "options": shuffled,
                "correct_index": correct_index,
                "points": points,
                "explanation": explanation,
                "option_feedback": option_feedback,
                "source": source,
                "page": page,
                "section": section,
                "difficulty": difficulty,
                "mode": mode,
            }
        )

    return {
        "title": f"{exam} {mode.replace('_', ' ').title()} Live Exam",
        "topic": topic or "Uploaded document corpus",
        "difficulty": difficulty,
        "items": items,
        "weak_topics": weak_topics,
        "instructions": [
            "Choose one option per question.",
            "Answers reveal only after submission.",
            "Points are awarded only for correct choices.",
            "Every correct answer is grounded in uploaded or permitted evidence.",
        ],
    }


def embedding_retrieve(corpus: List[Dict[str, Any]], query: str, k: int = 8, model: str = "text-embedding-3-large") -> List[Dict[str, Any]]:
    """Semantic retrieval with OpenAI embeddings, falling back to TF-IDF."""

    key = os.getenv("OPENAI_API_KEY")
    if not key or not corpus:
        return retrieve(corpus, query, k)
    try:
        import numpy as np
        from openai import OpenAI

        client = OpenAI(api_key=key)
        texts = [f"{c['source']} {c['section']} {c['kind']} {c['text'][:5000]}" for c in corpus]
        vecs = client.embeddings.create(model=model, input=texts + [query]).data
        arr = np.array([v.embedding for v in vecs], dtype="float32")
        docs, q = arr[:-1], arr[-1]
        docs = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = docs @ q
        qnums = set(re.findall(r"\d+(?:\.\d+)?", query))
        for i, c in enumerate(corpus):
            scores[i] += 0.08 * len(qnums & set(re.findall(r"\d+(?:\.\d+)?", c["text"])))
            scores[i] += 0.05 if c.get("kind") == "table" and qnums else 0
        order = np.argsort(scores)[::-1][:k]
        return [dict(corpus[int(i)], score=float(scores[int(i)]), embedding_model=model) for i in order]
    except Exception:
        return retrieve(corpus, query, k)


def pinecone_ready() -> bool:
    return bool(os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_INDEX") and os.getenv("OPENAI_API_KEY"))


def pinecone_upsert(corpus: List[Dict[str, Any]], namespace: str = "") -> bool:
    if not pinecone_ready() or not corpus:
        return False
    try:
        from openai import OpenAI
        from pinecone import Pinecone

        ns = namespace or os.getenv("PINECONE_NAMESPACE", "default")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX", ""))
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        texts = [f"{c['source']} {c['section']} {c['kind']} {c['text'][:5000]}" for c in corpus]
        vecs = client.embeddings.create(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"), input=texts).data
        payload = []
        for i, (c, v) in enumerate(zip(corpus, vecs)):
            payload.append(
                {
                    "id": hashlib.sha256(f"{c.get('source')}:{c.get('page')}:{i}".encode()).hexdigest(),
                    "values": v.embedding,
                    "metadata": {
                        "idx": i,
                        "source": str(c.get("source", "")),
                        "page": int(c.get("page", 1) or 1),
                        "section": str(c.get("section", ""))[:200],
                        "kind": str(c.get("kind", "")),
                        "text": str(c.get("text", ""))[:3000],
                    },
                }
            )
        for start in range(0, len(payload), 100):
            index.upsert(vectors=payload[start:start + 100], namespace=ns)
        return True
    except Exception:
        return False


def pinecone_retrieve(corpus: List[Dict[str, Any]], query: str, k: int = 8, namespace: str = "") -> List[Dict[str, Any]]:
    if not pinecone_ready():
        return embedding_retrieve(corpus, query, k)
    try:
        from openai import OpenAI
        from pinecone import Pinecone

        ns = namespace or os.getenv("PINECONE_NAMESPACE", "default")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        qvec = client.embeddings.create(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"), input=[query]).data[0].embedding
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        res = pc.Index(os.getenv("PINECONE_INDEX", "")).query(vector=qvec, top_k=k, include_metadata=True, namespace=ns)
        out = []
        for match in res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", []):
            md = match.get("metadata", {}) if isinstance(match, dict) else match.metadata
            score = match.get("score", 0) if isinstance(match, dict) else match.score
            out.append(
                {
                    "source": md.get("source", "pinecone"),
                    "page": md.get("page", 1),
                    "section": md.get("section", "Vector"),
                    "kind": md.get("kind", "vector"),
                    "text": md.get("text", ""),
                    "score": float(score or 0),
                }
            )
        return out or embedding_retrieve(corpus, query, k)
    except Exception:
        return embedding_retrieve(corpus, query, k)


def supabase_ready() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def supabase_log_metadata(metadata: Dict[str, Any]) -> bool:
    if not supabase_ready():
        return False
    try:
        from supabase import create_client

        client = create_client(os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""))
        client.table(os.getenv("SUPABASE_METADATA_TABLE", "rag_metadata")).upsert(metadata).execute()
        return True
    except Exception:
        return False


lexical_retrieve = retrieve
llamaindex_retrieve = retrieve
tfidf_ann_cnn_retrieve = retrieve


def format_context(chunks: List[Dict[str, Any]], max_chars: int = 9000) -> str:
    blocks = [f"[{c['source']} p.{c['page']} {c['section']} {c['kind']} score={c.get('score', 0):.3f}]\n{c['text']}" for c in chunks]
    return "\n\n".join(blocks)[:max_chars]


def redacted_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if os.getenv("DPDP_REDACT", "true").lower() != "true":
        return chunks
    out = []
    for c in chunks:
        x = dict(c)
        x["text"] = redact_personal_data(str(x.get("text", "")))
        out.append(x)
    return out


def media_inventory(corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a compact media/library view from uploaded documents."""

    items = []
    for c in corpus:
        if c.get("kind") in {"image", "table"} or re.search(r"\b(?:figure|fig\.|table|chart|diagram|image|logo|asset|video|media)\b", c.get("text", ""), re.I):
            items.append(
                {
                    "source": c.get("source", ""),
                    "page": c.get("page", 1),
                    "section": c.get("section", "Document"),
                    "type": c.get("kind", "text"),
                    "description": c.get("text", "")[:500],
                }
            )
    return items[:80]


def integration_registry(extra: str = "", include_pg: bool = True) -> List[Dict[str, str]]:
    """A neutral registry users can extend with free, paid, new, or underrated tools."""

    base = [
        {"name": "Streamlit", "category": "app", "pricing": "free/paid", "use": "deploy the chatbot and builder UI"},
        {"name": "PostgreSQL", "category": "database", "pricing": "free/paid", "use": "store chunks, media notes, and query logs"},
        {"name": "OpenRouter", "category": "llm-router", "pricing": "paid/free tiers", "use": "switch between hosted language models"},
        {"name": "Hugging Face", "category": "models", "pricing": "free/paid", "use": "open models, inference endpoints, and datasets"},
        {"name": "OpenAI", "category": "llm/embeddings", "pricing": "paid", "use": "text-embedding-3-large and chat models"},
        {"name": "Google Gemini", "category": "llm", "pricing": "free/paid", "use": "alternative reasoning and generation"},
        {"name": "GitHub", "category": "source-control", "pricing": "free/paid", "use": "version, deploy, and collaborate on generated sites"},
        {"name": "Canva", "category": "creative", "pricing": "free/paid", "use": "marketing creatives and brand kits"},
        {"name": "Mailchimp", "category": "email", "pricing": "free/paid", "use": "campaigns and audience lists"},
        {"name": "Buffer", "category": "social", "pricing": "free/paid", "use": "schedule social posts"},
        {"name": "Plausible", "category": "analytics", "pricing": "paid/free self-host", "use": "privacy-friendly web analytics"},
    ]
    if include_pg:
        seen = {i["name"].lower() for i in base}
        for item in load_integrations_pg():
            if item["name"].lower() not in seen:
                base.append(item)
                seen.add(item["name"].lower())
    for line in extra.splitlines():
        parts = [p.strip() for p in re.split(r"[,|]", line) if p.strip()]
        if parts:
            base.append(
                {
                    "name": parts[0],
                    "category": parts[1] if len(parts) > 1 else "custom",
                    "pricing": parts[2] if len(parts) > 2 else "unknown",
                    "use": parts[3] if len(parts) > 3 else "user supplied integration",
                    "base_url": parts[4] if len(parts) > 4 else "",
                    "model": parts[5] if len(parts) > 5 else "",
                    "api_key_env": parts[6] if len(parts) > 6 else "",
                    "score": parts[7] if len(parts) > 7 else "0",
                }
            )
    return base


def website_brief_features(brief: str) -> Dict[str, Any]:
    text = (brief or "").lower()
    urls = re.findall(r"https?://[^\s)>\"]+", brief or "")
    return {
        "audio": bool(re.search(r"\b(audio|mp3|voice|podcast|sound|music)\b", text)),
        "video": bool(re.search(r"\b(video|youtube|reel|short)\b", text)),
        "contact": bool(re.search(r"\b(contact|form|lead|enquiry|inquiry|whatsapp)\b", text)),
        "shop": bool(re.search(r"\b(shop|store|payment|buy|checkout|product)\b", text)),
        "urls": urls,
    }


def build_website(query: str, corpus: List[Dict[str, Any]], brand: str = "Scientific RAG", goal: str = "Convert visitors") -> Dict[str, str]:
    """Generate a prompt-shaped single-file website with SEO, critique, and tips."""

    hits = retrieve(corpus, query or brand, 8)
    evidence = [h["text"][:280] for h in hits]
    features = website_brief_features(query)
    title = escape(brand.strip() or "Scientific RAG")
    offer = escape((query or goal).strip()[:180] or "Evidence-grounded intelligence")
    description = escape(f"{brand}: {goal}. Built from user brief and retrieved evidence."[:155])
    cards = "\n".join(f"<article><p>{escape(t)}</p></article>" for t in evidence[:3]) or "<article><p>No uploaded evidence was available; review claims before publishing.</p></article>"
    audio = ""
    if features["audio"]:
        src = features["urls"][0] if features["urls"] else ""
        audio = f"""
    <section>
      <h2>Audio</h2>
      <p>Add your voice note, MP3, podcast, or outreach recording here.</p>
      <audio controls src="{escape(src)}"></audio>
    </section>"""
    contact = """
    <section>
      <h2>Contact</h2>
      <form>
        <label>Name<input name="name" autocomplete="name"></label>
        <label>Email<input name="email" type="email" autocomplete="email"></label>
        <label>Message<textarea name="message" rows="4"></textarea></label>
        <button type="button">Send</button>
      </form>
    </section>""" if features["contact"] else ""
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="{description}">
  <meta property="og:title" content="{title}">
  <meta property="og:description" content="{description}">
  <title>{title}</title>
  <style>
    body{{margin:0;font-family:Inter,Arial,sans-serif;color:#17202a;background:#f7f9fb;line-height:1.55}}
    header{{padding:64px 8vw;background:#0d1b2a;color:white}}
    h1{{font-size:clamp(36px,6vw,72px);margin:0 0 12px}}
    main{{padding:36px 8vw;display:grid;gap:24px}}
    section{{max-width:1120px}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}
    article{{background:white;border:1px solid #d8dee6;border-radius:8px;padding:18px}}
    input,textarea{{width:100%;padding:10px;margin:6px 0 12px;border:1px solid #cbd5e1;border-radius:6px}}
    audio{{width:100%;max-width:720px}}
    a.button{{display:inline-block;margin-top:16px;background:#12b886;color:#06110d;padding:12px 16px;border-radius:6px;text-decoration:none;font-weight:700}}
  </style>
  <script type="application/ld+json">{{"@context":"https://schema.org","@type":"WebSite","name":"{title}","description":"{description}"}}</script>
</head>
<body>
  <header><h1>{title}</h1><p>{offer}</p><a class="button" href="#evidence">Explore Evidence</a></header>
  <main>
    <section id="evidence"><h2>Evidence Highlights</h2><div class="grid">{cards}</div></section>
    <section><h2>What This Page Does</h2><p>{offer}</p></section>
    {audio}
    {contact}
    <section><h2>Action</h2><p>Use the uploaded corpus, media assets, integrations, and human review checklist to publish, test, and improve this page.</p></section>
  </main>
</body>
</html>"""
    seo = [
        "Use a specific page title under 60 characters.",
        "Keep meta description near 150 characters.",
        "Add one H1, descriptive H2s, alt text for images, and canonical URL before publishing.",
        "Use Open Graph tags for WhatsApp, LinkedIn, and social sharing.",
        "Compress images/audio and test mobile speed.",
    ]
    tips = [
        "Map every public claim to a source or remove it.",
        "Add analytics only after privacy notice and consent needs are reviewed.",
        "Use a visible contact/CTA section if the page is for outreach.",
        "For audio, host MP3 on a permitted URL and paste it in the brief.",
        "Review DPDP/GDPR/CCPA needs before collecting form data.",
    ]
    critique = [
        "Evidence coverage is limited." if len(hits) < 3 else "Evidence coverage is acceptable for a draft.",
        "Audio requested but no audio URL was detected." if features["audio"] and not features["urls"] else "Media request is represented where possible.",
        "Human review is required before publishing.",
    ]
    return {"html": html, "sources": json.dumps(hits, indent=2), "seo": seo, "tips": tips, "critique": critique}


TEMPLATE_LIBRARY = [
    {"name": "Landing Page", "format": "HTML", "category": "website"},
    {"name": "Documentation Site", "format": "HTML", "category": "website"},
    {"name": "Portfolio / Organization Page", "format": "HTML", "category": "website"},
    {"name": "Evidence Report", "format": "Markdown", "category": "report"},
    {"name": "Scientific Summary", "format": "Markdown", "category": "report"},
    {"name": "Compliance Report", "format": "Markdown", "category": "compliance"},
    {"name": "DPDP Privacy Notice", "format": "Markdown", "category": "compliance"},
    {"name": "Marketing Plan", "format": "Markdown", "category": "marketing"},
    {"name": "Email Campaign", "format": "Markdown", "category": "marketing"},
    {"name": "Social Media Pack", "format": "Markdown", "category": "marketing"},
    {"name": "Proposal", "format": "Markdown", "category": "business"},
    {"name": "Pitch Deck Outline", "format": "Markdown", "category": "business"},
    {"name": "Invoice / Quotation", "format": "HTML", "category": "business"},
    {"name": "Intake Form", "format": "HTML", "category": "form"},
    {"name": "Survey Form", "format": "HTML", "category": "form"},
    {"name": "Media Asset Sheet", "format": "JSON", "category": "media"},
    {"name": "Integration Matrix", "format": "CSV", "category": "integration"},
    {"name": "RAG Evaluation Sheet", "format": "CSV", "category": "evaluation"},
]


EMERGENT_STYLE_FEATURES = [
    {"feature": "Prompt-to-app building", "description": "Turn a plain-language idea into a structured app spec, screens, data models, and workflows."},
    {"feature": "Web and mobile app planning", "description": "Plan responsive web, PWA, Android, and iOS-ready experiences from one brief."},
    {"feature": "End-to-end full-stack structure", "description": "Generate UI, backend, database, authentication, and deployment checklist together."},
    {"feature": "Data and backend management", "description": "Define collections, tables, relationships, records, and scaling notes."},
    {"feature": "Authentication and access control", "description": "Plan email/OTP/social login, roles, permissions, and tenant isolation."},
    {"feature": "Workflow automation", "description": "Define triggers, conditions, approvals, notifications, and background jobs."},
    {"feature": "Integrations and APIs", "description": "Map payment gateways, CRMs, analytics, storage, notifications, and custom APIs."},
    {"feature": "One-click deployment readiness", "description": "Prepare hosting, domains, environment variables, secrets, and release checks."},
    {"feature": "GitHub and handoff", "description": "Keep code export, versioning, review, and developer extension paths explicit."},
    {"feature": "Analytics and growth", "description": "Plan SEO/ASO, campaigns, push/email engagement, metrics, and iteration loops."},
    {"feature": "Advanced agent controls", "description": "Support system prompt edits, custom agents, long-context planning, and high-compute tasks when available."},
    {"feature": "Security and compliance", "description": "Keep human approval, privacy gates, RBAC, audit metadata, and jurisdiction checks visible."},
]


CODEX_STYLE_FEATURES = [
    {"feature": "Workspace-first workflow", "tool": "files + metadata", "use": "read uploaded/local evidence before acting"},
    {"feature": "Patch-based editing", "tool": "change plan", "use": "keep edits scoped, reviewable, and reversible"},
    {"feature": "Terminal verification", "tool": "compile/tests/checks", "use": "run verification before export"},
    {"feature": "Tool readiness catalog", "tool": "toolbox", "use": "show packages, keys, and configured status"},
    {"feature": "Git handoff", "tool": "GitHub/Git", "use": "prepare branch/commit/PR workflow when credentials allow"},
    {"feature": "Review mode", "tool": "findings", "use": "prioritize risks, bugs, compliance gaps, and missing tests"},
    {"feature": "Human approval gate", "tool": "approval controls", "use": "human stays above agents and orchestrator"},
    {"feature": "Agent delegation pattern", "tool": "swarm", "use": "planner, retriever, verifier, guard, orchestrator"},
    {"feature": "Evidence grounding", "tool": "RAG verifier", "use": "answers cite uploaded or permitted web evidence only"},
    {"feature": "Deploy package", "tool": "zip/runtime/requirements", "use": "produce portable Streamlit deployment bundle"},
]


def template_options() -> List[Dict[str, str]]:
    return [dict(x) for x in TEMPLATE_LIBRARY]


def emergent_features() -> List[Dict[str, str]]:
    return [dict(x) for x in EMERGENT_STYLE_FEATURES]


def codex_features() -> List[Dict[str, str]]:
    return [dict(x) for x in CODEX_STYLE_FEATURES]


def codex_workflow_brief(task: str, corpus: List[Dict[str, Any]]) -> str:
    hits = retrieve(corpus, task or "implementation workflow", 5)
    evidence = "\n".join(f"- {h['source']} p.{h['page']} [{h['section']}]: {h['text'][:220]}" for h in hits)
    features = "\n".join(f"- **{f['feature']}** using `{f['tool']}`: {f['use']}" for f in CODEX_STYLE_FEATURES)
    return (
        "# Codex-Style Workflow\n\n"
        f"**Task:** {task}\n\n"
        "## Reference Evidence\n\n"
        f"{evidence or '- No uploaded evidence found.'}\n\n"
        "## Capability Pattern\n\n"
        f"{features}\n\n"
        "## Operating Rules\n\n"
        "1. Read evidence first.\n"
        "2. Keep changes small and reviewable.\n"
        "3. Prefer built-in tools before adding heavy dependencies.\n"
        "4. Run verification before packaging.\n"
        "5. Keep human approval above all agents.\n"
        "6. Export metadata and audit trail.\n"
    )


def emergent_app_blueprint(idea: str, corpus: List[Dict[str, Any]], app_type: str = "Web + Mobile") -> str:
    hits = retrieve(corpus, idea or app_type, 6)
    evidence = "\n".join(f"- {h['source']} p.{h['page']} [{h['section']}]: {h['text'][:240]}" for h in hits[:5])
    features = "\n".join(f"- **{x['feature']}**: {x['description']}" for x in EMERGENT_STYLE_FEATURES)
    return (
        f"# App Builder Blueprint\n\n"
        f"**App type:** {app_type}\n\n"
        f"**Idea:** {idea}\n\n"
        "## Evidence From Uploaded Corpus\n\n"
        f"{evidence or '- No supporting uploaded evidence found.'}\n\n"
        "## Emergent-Style Feature Coverage\n\n"
        f"{features}\n\n"
        "## Build Plan\n\n"
        "1. Define users, roles, permissions, and compliance requirements.\n"
        "2. Draft screens, navigation, and responsive layouts.\n"
        "3. Define data models, relationships, and PostgreSQL persistence.\n"
        "4. Specify workflows, triggers, notifications, and integrations.\n"
        "5. Add RAG/OCR/STT/media/template capabilities where evidence supports them.\n"
        "6. Configure secrets, deployment, metadata, and human approval gates.\n"
        "7. Review with a human before publishing or exporting.\n\n"
        "## Human Review Checklist\n\n"
        "- Claims are grounded in uploaded evidence.\n"
        "- Privacy/compliance jurisdiction is selected.\n"
        "- Cloud processing and paid APIs are approved.\n"
        "- Metadata and source citations are present.\n"
        "- Final output is reviewed by a responsible human.\n"
    )


def _cell_float(value: Any) -> Optional[float]:
    text = str(value or "").strip().replace(",", "")
    text = re.sub(r"[%₹$]", "", text)
    if not re.search(r"\d", text):
        return None
    try:
        return float(text)
    except Exception:
        return None


def _table_records_from_corpus(corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for chunk in corpus:
        if chunk.get("kind") != "table":
            continue
        header: List[str] = []
        sheet = "Sheet1"
        for raw_line in str(chunk.get("text", "")).splitlines():
            line = raw_line.strip()
            if line.lower().startswith("sheet:"):
                sheet = line.split(":", 1)[-1].strip() or sheet
                header = []
                continue
            if not (line.startswith("|") and line.endswith("|")):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not cells or all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells):
                continue
            if not header:
                header = cells
                continue
            if len(cells) == len(header):
                row = {header[i] or f"Column {i + 1}": cells[i] for i in range(len(header))}
                row["_source"] = chunk.get("source", "")
                row["_page"] = chunk.get("page", 1)
                row["_sheet"] = sheet
                records.append(row)
    return records


def _find_column(columns: List[str], options: List[str]) -> Optional[str]:
    lowered = {c.lower().replace("_", " ").strip(): c for c in columns}
    for opt in options:
        for key, original in lowered.items():
            if opt in key:
                return original
    return None


def _grade(percent: float) -> str:
    if percent >= 90:
        return "A+"
    if percent >= 75:
        return "A"
    if percent >= 60:
        return "B"
    if percent >= 45:
        return "C"
    if percent >= 33:
        return "D"
    return "E"


def school_clerk_automation(query: str, corpus: List[Dict[str, Any]]) -> Dict[str, Any]:
    """School-office automations with human approval and DPDP-minded outputs."""

    q = (query or "").lower()
    records = _table_records_from_corpus(corpus)
    task = "result_generation" if any(x in q for x in ["result", "marksheet", "mark sheet", "report card", "grade"]) else "school_office_packet"
    automation_catalog = [
        "result sheet / marksheet generation",
        "student roll list and class register",
        "attendance summary",
        "fee reminder draft",
        "parent WhatsApp/SMS notice draft",
        "transfer certificate draft checklist",
        "bonafide / character certificate draft",
        "exam seating and invigilation checklist",
        "admission inquiry register",
        "document verification checklist",
    ]
    pro_tips = [
        "Keep one student per row and one subject per column for automatic result generation.",
        "Use columns such as Roll No, Student Name, Class, Section, Hindi, English, Maths, Science, SST.",
        "Review totals, pass/fail, spelling, roll numbers, and personal data before export.",
        "Do not send student personal data to cloud LLMs unless lawful basis/consent and school policy allow it.",
        "For WhatsApp notices, send only to opted-in parents/guardians and keep messages minimal.",
    ]
    human_checklist = [
        "Human clerk/teacher verifies uploaded data source.",
        "Human confirms lawful basis and school authorization.",
        "Human reviews marks, totals, grades, and pass/fail before publishing.",
        "Human approves exports/downloads and parent communications.",
        "Sensitive personal data is redacted/minimized when not required.",
    ]

    if task == "result_generation" and records:
        columns = [c for c in records[0].keys() if not c.startswith("_")]
        name_col = _find_column(columns, ["student name", "name", "candidate"])
        roll_col = _find_column(columns, ["roll", "admission", "adm no", "enrol", "id"])
        class_col = _find_column(columns, ["class", "grade", "standard"])
        section_col = _find_column(columns, ["section", "sec"])
        excluded = {"total", "percentage", "percent", "grade", "result", "rank", "mobile", "phone", "aadhaar", "aadhar", "email", "age", "roll", "id", "admission", "class", "section"}
        subject_cols = []
        for col in columns:
            key = col.lower()
            if any(word in key for word in excluded):
                continue
            nums = [_cell_float(r.get(col)) for r in records]
            valid = [n for n in nums if n is not None and 0 <= n <= 100]
            if valid and len(valid) >= max(1, len(records) // 3):
                subject_cols.append(col)

        result_rows: List[Dict[str, Any]] = []
        max_per_subject = 100
        pass_mark = 33
        for row in records:
            marks = [_cell_float(row.get(col)) for col in subject_cols]
            clean_marks = [m for m in marks if m is not None]
            if not clean_marks:
                continue
            total = round(sum(clean_marks), 2)
            max_total = max_per_subject * len(subject_cols)
            percent = round((total / max_total) * 100, 2) if max_total else 0
            passed = all(m >= pass_mark for m in clean_marks)
            out = {
                "Roll No": row.get(roll_col, "") if roll_col else "",
                "Student Name": row.get(name_col, "") if name_col else row.get(columns[0], ""),
                "Class": row.get(class_col, "") if class_col else "",
                "Section": row.get(section_col, "") if section_col else "",
            }
            for col in subject_cols:
                out[col] = row.get(col, "")
            out.update({"Total": total, "Max Marks": max_total, "Percentage": percent, "Grade": _grade(percent), "Result": "PASS" if passed else "FAIL"})
            result_rows.append(out)

        csv_lines: List[str] = []
        if result_rows:
            headers = list(result_rows[0].keys())
            csv_lines.append(",".join(headers))
            for row in result_rows:
                csv_lines.append(",".join('"' + str(row.get(h, "")).replace('"', '""') + '"' for h in headers))
        pass_count = sum(1 for r in result_rows if r.get("Result") == "PASS")
        fail_count = sum(1 for r in result_rows if r.get("Result") == "FAIL")
        markdown = (
            "# School Result Generation\n\n"
            f"**Students processed:** {len(result_rows)}\n\n"
            f"**Subjects detected:** {', '.join(subject_cols) or 'None'}\n\n"
            f"**Pass:** {pass_count} | **Fail:** {fail_count}\n\n"
            "## Preview\n\n"
            + "\n".join(
                f"- {r.get('Roll No', '')} {r.get('Student Name', '')}: {r.get('Total')}/{r.get('Max Marks')} ({r.get('Percentage')}%) {r.get('Grade')} {r.get('Result')}"
                for r in result_rows[:25]
            )
            + "\n\n## Human Approval Required\n\n"
            + "\n".join(f"- {x}" for x in human_checklist)
        )
        return {
            "task": task,
            "markdown": markdown,
            "csv": "\n".join(csv_lines),
            "rows": result_rows,
            "records_detected": len(records),
            "pro_tips": pro_tips,
            "human_checklist": human_checklist,
            "automation_catalog": automation_catalog,
            "note": "Result generation is computed locally from uploaded table evidence. Review before publication.",
        }

    templates = {
        "Attendance Summary": "Date, Class, Section, Total Students, Present, Absent, Leave, Remarks",
        "Fee Reminder": "Student Name, Class, Section, Due Amount, Due Date, Parent Contact, Message Status",
        "Parent Notice": "Audience, Notice Title, Date, Message, Approved By, Dispatch Channel",
        "Transfer Certificate Checklist": "Student Name, Admission No, Class, Dues Clear, Library Clear, Principal Approval, TC Number",
        "Admission Register": "Admission No, Student Name, DOB, Class, Guardian, Contact, Address, Documents Verified",
    }
    md = (
        "# School Clerk Automation Packet\n\n"
        f"**Request:** {query or 'School office automation'}\n\n"
        "## Available Automations\n\n"
        + "\n".join(f"- {x}" for x in automation_catalog)
        + "\n\n## Clerk Templates\n\n"
        + "\n".join(f"### {name}\n`{cols}`\n" for name, cols in templates.items())
        + "\n## Pro Tips\n\n"
        + "\n".join(f"- {x}" for x in pro_tips)
        + "\n\n## Human In The Loop\n\n"
        + "\n".join(f"- {x}" for x in human_checklist)
    )
    return {
        "task": task,
        "markdown": md,
        "csv": "",
        "rows": [],
        "records_detected": len(records),
        "pro_tips": pro_tips,
        "human_checklist": human_checklist,
        "automation_catalog": automation_catalog,
        "note": "Upload a CSV/XLSX marks table and ask for result generation to compute results.",
    }


def render_template(name: str, query: str, corpus: List[Dict[str, Any]], brand: str = "Evidence Studio") -> Dict[str, str]:
    hits = retrieve(corpus, query or name, 6)
    evidence = "\n".join(f"- {h['source']} p.{h['page']} [{h['section']}]: {h['text'][:260]}" for h in hits[:5])
    safe_brand = escape(brand or "Evidence Studio")
    safe_query = escape(query or name)
    if name in {"Landing Page", "Documentation Site", "Portfolio / Organization Page"}:
        page = build_website(query or name, corpus, brand, f"{name} generated from uploaded evidence")
        return {"content": page["html"], "filename": f"{name.lower().replace(' ', '_').replace('/', '')}.html", "mime": "text/html"}
    if name in {"Invoice / Quotation", "Intake Form", "Survey Form"}:
        html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{safe_brand} - {escape(name)}</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;color:#17202a}}label,input,textarea{{display:block;width:100%;margin:8px 0}}section{{max-width:760px}}</style></head>
<body><section><h1>{escape(name)}</h1><p>{safe_query}</p><label>Name<input></label><label>Email<input></label><label>Details<textarea rows="6"></textarea></label><h2>Evidence Notes</h2><pre>{escape(evidence)}</pre></section></body></html>"""
        return {"content": html, "filename": f"{name.lower().replace(' ', '_').replace('/', '')}.html", "mime": "text/html"}
    if name == "Media Asset Sheet":
        return {"content": json.dumps(media_inventory(corpus), indent=2), "filename": "media_asset_sheet.json", "mime": "application/json"}
    if name in {"Integration Matrix", "RAG Evaluation Sheet"}:
        rows = ["item,type,status,notes"]
        if name == "Integration Matrix":
            rows += [f"{i['name']},{i['category']},{i['pricing']},{i['use']}" for i in integration_registry()[:30]]
        else:
            rows += [f"{h['source']},evidence,review,{h['section']} p.{h['page']}" for h in hits]
        return {"content": "\n".join(rows), "filename": f"{name.lower().replace(' ', '_')}.csv", "mime": "text/csv"}
    md = (
        f"# {name}\n\n"
        f"**Brand/Project:** {brand}\n\n"
        f"**Brief:** {query or name}\n\n"
        "## Evidence Basis\n\n"
        f"{evidence or '- Not found in uploaded documents.'}\n\n"
        "## Draft\n\n"
        "Use the evidence above. Do not add unsupported claims. Mark missing facts clearly.\n\n"
        "## Limitations\n\n"
        "This template is grounded only in uploaded documents and should be reviewed before publication."
    )
    return {"content": md, "filename": f"{name.lower().replace(' ', '_').replace('/', '')}.md", "mime": "text/markdown"}


def marketing_plan(query: str, corpus: List[Dict[str, Any]], integrations: List[Dict[str, str]]) -> str:
    hits = retrieve(corpus, query, 5)
    tools = ", ".join(i["name"] for i in integrations[:8])
    proof = "\n".join(f"- {h['source']} p.{h['page']}: {h['text'][:220]}" for h in hits[:4])
    return (
        "## Marketing Plan\n\n"
        f"**Campaign goal:** {query}\n\n"
        "**Positioning:** Lead with claims supported by uploaded evidence. Keep any general-market claims separate unless the open-source toggle is enabled.\n\n"
        "**Channels:** Website landing page, email, social posts, short-form media, and analytics feedback loop.\n\n"
        f"**Suggested integrations:** {tools}.\n\n"
        "**Media workflow:** Extract tables, images, figures, and quoted evidence from the media library; convert them into page sections, posts, and downloadable assets.\n\n"
        "**Evidence to reuse:**\n" + (proof or "- No relevant uploaded evidence found yet.") + "\n\n"
        "**Feedback loop:** track visits, clicks, questions, weak answers, and conversions; update the corpus and regenerate page/campaign copy when evidence changes."
    )


def _local_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "I could not find relevant evidence in the uploaded documents."
    bullets = [f"- `{c['source']}` p.{c['page']} [{c['section']}]: {c['text'][:520]}" for c in chunks[:6]]
    return (
        "### Evidence-grounded scientific answer\n\n"
        "Interpret this only from the uploaded evidence below. I do not infer beyond it, fabricate values, or convert uncertainty into certainty.\n\n"
        "### Retrieved evidence\n\n" + "\n".join(bullets) +
        "\n\n### Limitations\n\nIf a required value, method, figure, table, or structural comparison is absent above, it is not supported by the uploaded corpus."
    )


def _grounding_guard(answer: str, chunks: List[Dict[str, Any]]) -> str:
    """Add a conservative guardrail when an LLM answer lacks visible citations."""

    if not chunks:
        return "Not found in uploaded documents. I need relevant uploaded evidence before answering."
    source_names = {str(c.get("source", "")) for c in chunks}
    has_source = any(name and name in answer for name in source_names)
    has_page = bool(re.search(r"\bp\.?\s*\d+|\bpage\s+\d+", answer, re.I))
    if has_source and has_page:
        return answer
    evidence = "\n".join(f"- `{c['source']}` p.{c['page']} [{c['section']}]: {c['text'][:360]}" for c in chunks[:5])
    return (
        "### Grounded Answer\n\n"
        "The generated response did not include enough explicit source citations, so I am returning only the retrieved evidence instead of risking hallucination.\n\n"
        "### Retrieved Evidence\n\n"
        f"{evidence}\n\n"
        "### Limitation\n\n"
        "A final answer is not supported unless each claim can be tied to the uploaded sources above."
    )


def _message_role(role: str) -> str:
    aliases = {"human": "user", "ai": "assistant", "model": "assistant"}
    role = aliases.get((role or "user").lower(), (role or "user").lower())
    return role if role in {"system", "user", "assistant", "tool"} else "user"


def _message_content(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    content = item.get("content", item.get("message", item.get("text", item.get("value", ""))))
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", part.get("content", part))))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def router_messages(rule: str, question: str, context: str, history: Optional[List[Any]] = None) -> List[Dict[str, str]]:
    """Build OpenAI-compatible messages from any common message shape."""

    messages: List[Dict[str, str]] = [{"role": "system", "content": rule}]
    for item in history or []:
        role = _message_role(item.get("role", item.get("type", item.get("kind", "user"))) if isinstance(item, dict) else "user")
        content = _message_content(item).strip()
        if not content:
            continue
        if role == "tool" or (isinstance(item, dict) and item.get("name")):
            name = item.get("name", "tool") if isinstance(item, dict) else "tool"
            content = f"{name} message:\n{content}"
            role = "user"
        messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nRetrieved uploaded-document evidence:\n{context}\n\nFormat: concise answer, evidence bullets with citations, limitations.",
        }
    )
    return messages


def _provider() -> Tuple[str, str, str, str | None]:
    p = os.getenv("LLM_PROVIDER", "local").lower()
    if p == "gemini":
        return p, os.getenv("GEMINI_MODEL", "gemini-1.5-flash"), "", os.getenv("GOOGLE_API_KEY")
    if p == "grok":
        return p, os.getenv("GROK_MODEL", "grok-2-latest"), os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"), os.getenv("GROK_API_KEY")
    if p == "claude":
        return p, os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"), os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages"), os.getenv("ANTHROPIC_API_KEY")
    if p == "huggingface":
        return p, os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"), os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1"), os.getenv("HF_TOKEN")
    if p == "openrouter":
        return p, os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"), os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"), os.getenv("OPENROUTER_API_KEY")
    if p == "ollama":
        return p, os.getenv("OLLAMA_MODEL", "llama3.1"), os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), os.getenv("OLLAMA_API_KEY", "ollama")
    if p == "custom":
        return p, os.getenv("CUSTOM_LLM_MODEL", "model-name"), os.getenv("CUSTOM_LLM_BASE_URL", ""), os.getenv(os.getenv("CUSTOM_LLM_API_KEY_ENV", "CUSTOM_LLM_API_KEY"))
    return p, os.getenv("OPENAI_MODEL", "gpt-4o-mini"), os.getenv("OPENAI_BASE_URL", ""), os.getenv("OPENAI_API_KEY")


def _has_non_latin(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0600-\u06FF]", text or ""))


def transliteration_instruction(question: str, context: str, provider: str, key: Optional[str]) -> str:
    engine = os.getenv("TRANSLITERATION_ENGINE", "auto_llm").lower()
    if engine == "none" or not _has_non_latin(f"{question}\n{context}"):
        return ""
    if engine in {"auto_llm", "llm"} and (provider == "local" or not key):
        return (
            "Automatic transliteration requested, but no approved LLM transliteration provider is available. "
            "Preserve the original script exactly and state that transliteration was not performed."
        )
    if engine in {"auto_llm", "llm", "bhashini", "indic_rules", "indic_nlp", "aksharamukha", "inltk", "google_input_tools"}:
        tool_name = {
            "indic_nlp": "Indic NLP Library",
            "aksharamukha": "Aksharamukha",
            "inltk": "iNLTK",
            "google_input_tools": "Google Input Tools/manual phonetic input",
            "bhashini": "Bhashini/Indic transliteration",
            "indic_rules": "Indic transliteration rules",
        }.get(engine, "the selected LLM")
        return (
            f"Automatic transliteration rule using {tool_name}: when non-Latin text appears in the user query, OCR, tables, or retrieved evidence, "
            "keep the original script and add roman transliteration in parentheses on first mention. "
            "For Hindi or Hinglish, prefer clear Devanagari Hindi text first, then roman transliteration in parentheses where useful. "
            "Do not translate meaning unless the user explicitly asks for translation. "
            "Mark transliteration as approximate when OCR quality, handwriting, spelling, or language detection is uncertain. "
            "Never change numeric values, names, roll numbers, legal identifiers, citations, or units during transliteration."
        )
    return ""


RESPONSE_LANGUAGE_OPTIONS = {
    "auto": "the user's language; if unclear, use English",
    "english": "English",
    "hindi": "Hindi in Devanagari script",
    "urdu": "Urdu in Urdu/Nastaliq script",
    "arabic": "Arabic",
    "bengali": "Bengali",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "marathi": "Marathi in Devanagari script",
    "gujarati": "Gujarati",
    "punjabi": "Punjabi in Gurmukhi script",
    "french": "French",
    "spanish": "Spanish",
    "german": "German",
}


def response_language_instruction(question: str) -> str:
    configured = os.getenv("RESPONSE_LANGUAGE", "auto").lower().strip()
    q = (question or "").lower()
    if configured == "auto":
        if re.search(r"\b(hindi|हिंदी|हिन्दी|देवनागरी|हिंदी में|hindi me)\b", q):
            configured = "hindi"
        elif re.search(r"\b(urdu|اردو)\b", q):
            configured = "urdu"
        elif re.search(r"\b(arabic|عربي|العربية)\b", q):
            configured = "arabic"
        elif re.search(r"\b(english|अंग्रेजी|अंग्रेज़ी)\b", q):
            configured = "english"
    target = RESPONSE_LANGUAGE_OPTIONS.get(configured, os.getenv("RESPONSE_LANGUAGE", "auto"))
    if configured == "auto":
        return (
            "Response language rule: answer in the user's language when clear; otherwise use English. "
            "This is translation, not transliteration."
        )
    return (
        f"Response language rule: translate the final answer into {target}. "
        "Keep source citations, filenames, page numbers, section labels, numeric values, names, and quoted evidence exact. "
        "Do not translate source filenames or citations. If a source phrase is important, show translated meaning and keep the original phrase in parentheses. "
        "This is semantic translation of the answer, not mere transliteration."
    )


def _llm_select_workflow(
    query: str,
    actions: List[str],
    evidence_state: Dict[str, Any],
    provider_hint: str = "local",
) -> Optional[Dict[str, Any]]:
    """Use the selected LLM as an internal router when policy and keys allow it."""

    provider, model, base_url, key = _provider()
    if provider == "local" or not key:
        return None
    if provider != "ollama" and os.getenv("DPDP_CLOUD_CONSENT", "false").lower() != "true":
        return None

    system = (
        "You are an internal workflow router, not an answering assistant. "
        "Choose exactly one workflow from the allowed list. "
        "Prefer the smallest workflow that satisfies the user's request. "
        "Return only JSON with keys: selected_action, confidence, rationale."
    )
    user = {
        "query": query,
        "allowed_actions": actions,
        "evidence_state": evidence_state,
        "selected_provider": provider_hint,
        "routing_policy": [
            "Use School clerk for marksheets, result generation, attendance, certificates, fees, school notices.",
            "Use Study quiz for exam practice, MCQ, question papers, flashcards.",
            "Use Visual maps for mindmaps, flowcharts, concept maps, diagrams.",
            "Use Chat or Agent chat for ordinary document questions.",
            "Use Live search only when live_search_enabled is true and query needs fresh information.",
            "Keep human review above every workflow.",
        ],
    }
    try:
        if provider == "gemini":
            import google.generativeai as genai

            genai.configure(api_key=key)
            out = genai.GenerativeModel(model).generate_content(system + "\n\n" + json.dumps(user))
            text = getattr(out, "text", "") or ""
        elif provider == "claude":
            payload = {
                "model": model,
                "max_tokens": 300,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": json.dumps(user)}],
            }
            req = Request(
                base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-api-key": key or "", "anthropic-version": "2023-06-01", "User-Agent": USER_AGENT},
                method="POST",
            )
            with urlopen(req, timeout=18) as resp:
                data = json.loads(resp.read(500_000).decode("utf-8"))
            text = "\n".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
        else:
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=base_url or None)
            out = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user)}],
                temperature=0,
                max_tokens=300,
            )
            text = out.choices[0].message.content or ""
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start: end + 1] if start != -1 and end != -1 else text)
        action = str(data.get("selected_action", "")).strip()
        if action not in actions:
            return None
        confidence = float(data.get("confidence", 0.78))
        return {
            "selected_action": action,
            "confidence": max(0.5, min(confidence, 0.98)),
            "rationale": str(data.get("rationale", "LLM selected the most relevant workflow."))[:400],
        }
    except Exception:
        return None


def generate(question: str, chunks: List[Dict[str, Any]], external: bool = False, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    provider, model, base_url, key = _provider()
    cloud_blocked = provider not in {"local"} and os.getenv("DPDP_CLOUD_CONSENT", "false").lower() != "true"
    if cloud_blocked:
        note = "\n\nCloud LLM was blocked because DPDP cloud-processing consent/lawful basis was not enabled."
        if os.getenv("TRANSLITERATION_ENGINE", "auto_llm").lower() != "none" and _has_non_latin(question + "\n" + format_context(chunks, 2500)):
            note += " Automatic LLM transliteration was also blocked; original script is preserved."
        return {"answer": _local_answer(question, chunks) + note, "provider": "local", "model": "dpdp-privacy-gate"}
    safe_chunks = redacted_chunks(chunks) if provider != "local" else chunks
    context = format_context(safe_chunks)
    translit_rule = transliteration_instruction(question, context, provider, key)
    lang_rule = response_language_instruction(question)
    rule = (
        "You are a scientific RAG assistant with strict research temperament. Use only uploaded-document evidence. Do not use memory, assumptions, or outside knowledge. Every factual claim must cite source filename and page/section from the evidence. Preserve units, numeric values, denominators, sample sizes, protein/gene names, methods, table/figure context, uncertainty, OCR text, transliteration uncertainty, and citations. Separate observation from interpretation. Do not overclaim causality, novelty, safety, clinical relevance, or statistical significance unless the evidence states it. If evidence is insufficient, answer: 'Not found in uploaded documents' and list the missing evidence."
        if not external else
        "You are a scientific RAG assistant with strict research temperament. Use uploaded evidence first. Every document-supported claim must cite source filename and page/section. Label any outside/open-source knowledge separately and never mix it with document-supported claims. Mark transliteration as approximate unless directly supported by OCR text. Do not overclaim causality, safety, clinical relevance, or statistical significance."
    )
    rule += "\n\n" + lang_rule
    if translit_rule:
        rule += "\n\n" + translit_rule
    if provider == "local" or not key:
        answer = _local_answer(question, chunks)
        if os.getenv("RESPONSE_LANGUAGE", "auto").lower() not in {"auto", "english"} or re.search(r"\b(hindi|urdu|arabic|translate|अनुवाद|हिंदी|اردو)\b", question or "", re.I):
            answer += "\n\nTranslation note: semantic translation requires an approved LLM provider. The local fallback preserves retrieved evidence in its original language."
        if translit_rule and _has_non_latin(f"{question}\n{context}"):
            answer += "\n\nTransliteration note: automatic LLM transliteration was requested, but no approved LLM provider/key is active. Original script is preserved."
        return {"answer": answer, "provider": "local", "model": "evidence-only"}
    if provider == "gemini":
        try:
            import google.generativeai as genai

            genai.configure(api_key=key)
            msg = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in router_messages(rule, question, context, history))
            out = genai.GenerativeModel(model).generate_content(msg)
            return {"answer": _grounding_guard(getattr(out, "text", "") or "", chunks), "provider": provider, "model": model}
        except Exception as exc:
            return {"answer": _local_answer(question, chunks) + f"\n\nProvider failed: {exc}", "provider": "local", "model": "fallback"}
    if provider == "claude":
        try:
            payload = {
                "model": model,
                "max_tokens": 1200,
                "temperature": 0,
                "system": rule,
                "messages": [{"role": "user", "content": f"Question:\n{question}\n\nEvidence:\n{context}\n\nFormat: concise answer, evidence bullets with citations, limitations."}],
            }
            req = Request(
                base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-api-key": key or "", "anthropic-version": "2023-06-01", "User-Agent": USER_AGENT},
                method="POST",
            )
            with urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read(2_000_000).decode("utf-8"))
            text = "\n".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
            return {"answer": _grounding_guard(text, chunks), "provider": provider, "model": model}
        except Exception as exc:
            return {"answer": _local_answer(question, chunks) + f"\n\nProvider failed: {exc}", "provider": "local", "model": "fallback"}
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=base_url or None)
        out = client.chat.completions.create(model=model, messages=router_messages(rule, question, context, history), temperature=0.0)
        return {"answer": _grounding_guard(out.choices[0].message.content or "", chunks), "provider": provider, "model": model}
    except Exception as exc:
        return {"answer": _local_answer(question, chunks) + f"\n\nProvider failed: {exc}", "provider": "local", "model": "fallback"}


async def answer_rag_chat(
    question: str,
    corpus: List[Dict[str, Any]],
    provider: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 8,
    use_llamaindex: bool = True,
    allow_external_knowledge: bool = False,
    retrieval_engine: str = "tfidf",
) -> Dict[str, Any]:
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    hits = embedding_retrieve(corpus, question, top_k) if retrieval_engine == "openai_embeddings" else retrieve(corpus, question, top_k)
    ans = generate(question, hits, allow_external_knowledge, history)
    return {"answer": ans["answer"], "sources": hits, "latency_s": 0.0, "langchain_document_count": len(hits), **ans}


async def answer_with_agent_pipeline_from_corpus(
    question: str,
    corpus: List[Dict[str, Any]],
    corpus_summary: str,
    provider: Optional[str],
    max_iterations: int = 3,
) -> Dict[str, Any]:
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    turns = [
        {"agent": "planner", "message": "Break the query into retrieve, answer, verify."},
        {"agent": "executor", "message": "Retrieve evidence from uploaded documents only."},
    ]
    hits = retrieve(corpus, question, 8)
    ans = generate(question, hits)
    turns.append({"agent": "verifier", "message": "Check that the answer cites retrieved evidence.", "payload": {"sources": len(hits)}})
    return {"answer": ans["answer"], "sources": hits, "conversation": turns, "latency_s": 0.0, **ans}


async def run_multi_agent(goal: str, provider: Optional[str] = None, data_zip: Optional[Path] = None, max_docs: int = 40, max_pages: int = 20, max_iterations: int = 3) -> Dict[str, Any]:
    if not data_zip:
        raise ValueError("Provide a data path.")
    corpus, summary = build_corpus(data_zip, max_docs, max_pages)
    return await answer_with_agent_pipeline_from_corpus(goal, corpus, summary, provider, max_iterations)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-zip", type=Path, required=True)
    ap.add_argument("--goal", default="Summarize the uploaded documents")
    ap.add_argument("--inspect-data", action="store_true")
    args = ap.parse_args()
    corpus, summary = build_corpus(args.data_zip)
    print(summary if args.inspect_data else json.dumps({"summary": summary, "answer": _local_answer(args.goal, retrieve(corpus, args.goal))}, indent=2))


if __name__ == "__main__":
    main()
