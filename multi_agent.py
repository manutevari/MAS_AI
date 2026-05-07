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
import zipfile
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from html import escape
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_: Any, **__: Any) -> bool:
        return False


EXTS = {".pdf", ".txt", ".md", ".csv", ".tsv", ".xlsx", ".xls", ".json", ".png", ".jpg", ".jpeg", ".webp"}
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


TRANSLITERATION_MODELS = [
    {"label": "No transliteration", "engine": "none", "pricing": "free", "key_required": "no"},
    {"label": "Indic transliteration rules - free/local", "engine": "indic_rules", "pricing": "free", "key_required": "no"},
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
        ("Speech to text", "free/paid", "openai", "OPENAI_API_KEY", "manual transcript + Whisper API + integration targets"),
        ("Text to speech", "free/paid", "none required", "", "safe script + external/free/local model registry"),
        ("Website builder", "free/local", "streamlit", "", "HTML preview and download"),
        ("Templates", "free/local", "streamlit", "", "HTML/Markdown/JSON/CSV template generation"),
        ("Marketing", "free/local", "streamlit", "", "evidence-grounded campaign planning"),
        ("Media management", "free/local", "pandas/Pillow", "", "image/table/figure inventory"),
        ("Compliant web ingestion", "free/local", "stdlib", "", "robots.txt, URL confirmation, redaction, size limits"),
        ("International compliance", "free/local", "stdlib", "", "India DPDP + GDPR/UK/CCPA safe controls"),
        ("Swarm topology", "free/local", "streamlit", "", "hybrid/hierarchy/mesh/star/pipeline/ring/tree/blackboard/committee"),
        ("Human review", "free/local", "streamlit", "", "approval gates, metadata, audit trail"),
        ("Codex-style workflow", "free/local", "streamlit", "", "workspace-first actions, verification, review, package handoff"),
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


def _text(raw: bytes, name: str, max_pages: int) -> List[Tuple[int, str, str]]:
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return [(p, t, "text") for p, t in _pdf(raw, max_pages)]
    if ext in {".csv", ".tsv", ".xlsx", ".xls"}:
        return [(1, _table(raw, name), "table")]
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return [(1, _image(raw, name), "image")]
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


def _chunk(name: str, page: int, text: str, kind: str) -> List[Chunk]:
    section, buf, out = "Document", [], []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        words = " ".join(buf).split()
        for i in range(0, len(words), 220):
            part = " ".join(words[max(0, i - 35): i + 220]).strip()
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
    for url in urls[:20]:
        fetched = fetch_url_text(url.strip(), jurisdiction)
        notes.append(f"{url}: {fetched['note']}")
        if fetched["ok"]:
            rows.extend(asdict(c) | {"numbers": c.numbers} for c in _chunk(url, 1, fetched["text"], "web"))
    return rows, f"Indexed {len(rows)} compliant web chunks from {len(urls[:20])} URL(s)."


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


def corpus_id(paths: List[Path]) -> str:
    raw = "|".join(f"{p.name}:{p.stat().st_size if p.exists() else 0}" for p in paths)
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
        "stt_engine": os.getenv("STT_ENGINE", "manual"),
        "transliteration_engine": os.getenv("TRANSLITERATION_ENGINE", "none"),
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


def study_quiz_generator(
    corpus: List[Dict[str, Any]],
    exam: str,
    topic: str,
    count: int = 10,
    difficulty: str = "medium",
    mode: str = "question_paper",
) -> str:
    """NotebookLM-inspired grounded quiz/question-paper generator."""

    hits = retrieve(corpus, f"{exam} {topic} {difficulty}", min(max(count, 5), 25))
    evidence = "\n".join(f"- {h['source']} p.{h['page']} [{h['section']}]: {h['text'][:260]}" for h in hits)
    if not hits:
        return "# Study Generator\n\nNot found in uploaded documents. Upload syllabus, notes, textbook chapters, or previous papers first."
    questions = []
    for i, h in enumerate(hits[:count], start=1):
        stem = re.sub(r"\s+", " ", h["text"])[:180]
        cite = f"`{h['source']}` p.{h['page']} [{h['section']}]"
        if mode == "flashcards":
            questions.append(f"**Card {i}**\n\nFront: What should a student remember from {cite}?\n\nBack: {stem}\n")
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
        "## Student Instructions\n\n"
        "- Answer only from the uploaded source material.\n"
        "- Cite the provided source reference in your answer.\n"
        "- If evidence is missing, write: Not found in uploaded documents.\n\n"
        "## Questions\n\n"
        + "\n".join(questions)
        + "\n## Evidence Basis\n\n"
        + evidence
    )


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


def build_website(query: str, corpus: List[Dict[str, Any]], brand: str = "Scientific RAG", goal: str = "Convert visitors") -> Dict[str, str]:
    """Generate a simple single-file website from retrieved evidence."""

    hits = retrieve(corpus, query or brand, 6)
    evidence = [h["text"][:260] for h in hits]
    title = escape(brand.strip() or "Scientific RAG")
    offer = escape(goal.strip() or "Evidence-grounded intelligence")
    cards = "\n".join(f"<article><p>{escape(t)}</p></article>" for t in evidence[:3]) or "<article><p>Upload source material to ground this section.</p></article>"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>
    body{{margin:0;font-family:Inter,Arial,sans-serif;color:#17202a;background:#f7f9fb}}
    header{{padding:64px 8vw;background:#0d1b2a;color:white}}
    h1{{font-size:clamp(36px,6vw,72px);margin:0 0 12px}}
    main{{padding:36px 8vw;display:grid;gap:24px}}
    section{{max-width:1120px}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}
    article{{background:white;border:1px solid #d8dee6;border-radius:8px;padding:18px}}
    a.button{{display:inline-block;margin-top:16px;background:#12b886;color:#06110d;padding:12px 16px;border-radius:6px;text-decoration:none;font-weight:700}}
  </style>
</head>
<body>
  <header><h1>{title}</h1><p>{offer}</p><a class="button" href="#evidence">Explore Evidence</a></header>
  <main>
    <section id="evidence"><h2>Evidence Highlights</h2><div class="grid">{cards}</div></section>
    <section><h2>Action</h2><p>Use the uploaded corpus, media assets, and integrations to publish, test, and improve this page.</p></section>
  </main>
</body>
</html>"""
    return {"html": html, "sources": json.dumps(hits, indent=2)}


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


def generate(question: str, chunks: List[Dict[str, Any]], external: bool = False, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    provider, model, base_url, key = _provider()
    cloud_blocked = provider not in {"local"} and os.getenv("DPDP_CLOUD_CONSENT", "false").lower() != "true"
    if cloud_blocked:
        return {"answer": _local_answer(question, chunks) + "\n\nCloud LLM was blocked because DPDP cloud-processing consent/lawful basis was not enabled.", "provider": "local", "model": "dpdp-privacy-gate"}
    safe_chunks = redacted_chunks(chunks) if provider != "local" else chunks
    context = format_context(safe_chunks)
    rule = (
        "You are a scientific RAG assistant with strict research temperament. Use only uploaded-document evidence. Do not use memory, assumptions, or outside knowledge. Every factual claim must cite source filename and page/section from the evidence. Preserve units, numeric values, denominators, sample sizes, protein/gene names, methods, table/figure context, uncertainty, OCR text, transliteration uncertainty, and citations. Separate observation from interpretation. Do not overclaim causality, novelty, safety, clinical relevance, or statistical significance unless the evidence states it. If evidence is insufficient, answer: 'Not found in uploaded documents' and list the missing evidence."
        if not external else
        "You are a scientific RAG assistant with strict research temperament. Use uploaded evidence first. Every document-supported claim must cite source filename and page/section. Label any outside/open-source knowledge separately and never mix it with document-supported claims. Mark transliteration as approximate unless directly supported by OCR text. Do not overclaim causality, safety, clinical relevance, or statistical significance."
    )
    if provider == "local" or not key:
        return {"answer": _local_answer(question, chunks), "provider": "local", "model": "evidence-only"}
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
