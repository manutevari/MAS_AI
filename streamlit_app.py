"""Compact Streamlit shell for the MAS scientific RAG agent."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import tempfile
import textwrap
import zipfile
from html import escape
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components

from monitoring import log_feedback, log_rag_event, monitoring_summary, write_evidently_report
from multi_agent import (
    agent_metrics_session,
    ai_policy_profiles,
    ai_policy_scan,
    answer_rag_chat,
    answer_with_agent_pipeline_from_corpus,
    ask_suggestions,
    build_corpus_from_paths,
    build_corpus_from_tavily,
    build_corpus_from_urls,
    build_website,
    codex_workflow_brief,
    compliance_report,
    corpus_id,
    corpus_metadata,
    emergent_app_blueprint,
    embedding_retrieve,
    extract_urls_from_paths,
    extract_urls_from_text,
    format_context,
    ingest_latest_updates,
    integration_registry,
    llm_model_catalog,
    load_integrations_pg,
    log_query_pg,
    marketing_plan,
    media_inventory,
    needs_live_search,
    ocr_language_options,
    ocr_model_options,
    orchestration_manager_plan,
    pinecone_retrieve,
    pinecone_upsert,
    render_template,
    retrieve,
    relationship_manager_agent,
    save_corpus_pg,
    school_clerk_automation,
    sentence_transformer_retrieve,
    speech_to_text_options,
    study_quiz_generator,
    study_quiz_items,
    synthesize_speech,
    supabase_log_metadata,
    swarm_initial_state,
    swarm_mermaid,
    template_options,
    text_to_speech_options,
    toolbox_catalog,
    transcribe_audio,
    transliteration_options,
    tts_guidance,
    update_swarm_feedback,
    upsert_integrations_pg,
    vector_space_knowledge,
    visual_map_pack,
    whatsapp_send_text,
    whatsapp_toolkit,
)


st.set_page_config(page_title="MAS Scientific RAG", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextArea, .stSelectbox, .stTextInput, .stButton {
    font-family: Inter, "Noto Sans Devanagari", "Nirmala UI", "Mangal", system-ui, sans-serif;
}
.block-container { max-width: 1440px; padding: .55rem .85rem .15rem .85rem; }
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
}
[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e5e7eb; }
[data-testid="stSidebar"] .block-container { padding: .75rem .7rem; }
div[data-testid="stVerticalBlock"] { gap: .32rem; }
div[data-testid="stHorizontalBlock"] { gap: .55rem; }
h1 { font-size: 1.32rem; margin: 0; letter-spacing: 0; }
h2, h3 { margin-top: .45rem; margin-bottom: .25rem; }
p, li, .stMarkdown, .stTextArea textarea { font-size: .94rem; line-height: 1.5; }
textarea { border-radius: 13px !important; border-color: #d7dde6 !important; background: #fff !important; }
.topbar {
    border: 1px solid #e5e7eb; border-radius: 14px; padding: 10px 13px; background: #fff;
    box-shadow: 0 8px 24px rgba(15, 23, 42, .04); margin-bottom: .45rem;
}
.muted { color: #64748b; }
.chip {
    display: inline-block; padding: 3px 8px; border: 1px solid #d8dee6; border-radius: 999px;
    margin: 2px 4px 2px 0; background: #fff; font-size: 11px; white-space: nowrap;
}
.ok { border-color: #9ed9b2; background: #f1fff6; }
.warn { border-color: #f5c67b; background: #fff8e8; }
.bad { border-color: #efb0b0; background: #fff5f5; }
.askbox {
    border: 1px solid #e5e7eb; border-radius: 18px; background: #fff; padding: 11px;
    box-shadow: 0 12px 34px rgba(15, 23, 42, .06);
}
.mini { color: #64748b; font-size: .75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .03em; }
[data-testid="stChatMessage"] {
    border: 1px solid #e5e7eb; border-radius: 16px; background: #fff; padding: 9px 13px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, .04);
}
[data-baseweb="tab"] { height: 34px; border-radius: 999px; padding: 0 13px; }
div.stButton > button, div.stDownloadButton > button {
    min-height: 34px; border-radius: 999px; font-weight: 650; padding: 0 .85rem;
}
div.stDownloadButton > button { background: #111827; color: #fff; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #e5e7eb; border-radius: 9px; padding: 8px; }
[data-testid="stSidebar"] { background: #050506; border-right: 1px solid #202124; display: block; }
[data-testid="stSidebar"] * { color: #d0d5dd; }
[data-testid="stSidebar"] .block-container { padding: 1.15rem .9rem; }
.block-container { padding-left: 1.2rem; padding-right: 1.2rem; }
.fake-rail {
    position: fixed; z-index: 9999; inset: 0 auto 0 0; width: 46px; background: #fff;
    border-right: 1px solid #ececf1; display: flex; flex-direction: column; align-items: center;
    padding-top: 11px; gap: 22px; color: #111827;
}
.rail-dot { width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center; font-size: 13px; }
.rail-icon { font-size: 18px; line-height: 1; color: #111827; }
.rail-avatar {
    margin-top: auto; margin-bottom: 16px; width: 22px; height: 22px; border-radius: 999px;
    display: grid; place-items: center; background: #10b981; color: #fff; font-size: 10px; font-weight: 700;
}
.chat-header {
    height: 44px; display: flex; align-items: center; justify-content: space-between; margin-bottom: .2rem;
}
.brand-select { font-weight: 600; font-size: 1rem; }
.top-actions { display: flex; gap: .35rem; justify-content: flex-end; align-items: center; }
.landing {
    min-height: 24vh; display: flex; flex-direction: column; align-items: center; justify-content: flex-end;
    text-align: center; padding-top: 4vh;
}
.landing h1 { font-size: 1.45rem; font-weight: 500; margin-bottom: 1.6rem; }
.askbox {
    width: min(720px, 100%); margin: .25rem auto .75rem auto; border-radius: 999px;
    padding: 8px 12px; box-shadow: 0 16px 46px rgba(15, 23, 42, .10);
}
.askbox textarea { min-height: 42px !important; max-height: 76px !important; border: 0 !important; box-shadow: none !important; }
.tool-strip { width: min(860px, 100%); margin: 0 auto .55rem auto; }
.soft-panel {
    border: 1px solid #ececf1; border-radius: 16px; background: #fff; padding: 12px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, .035);
}
.stPopover button { border-radius: 999px !important; }
.stApp { background: #050506; color: #d0d5dd; }
.main .block-container { background: #050506; }
h1, h2, h3, h4, p, li, label, span, .stMarkdown { color: #d0d5dd; }
.brand-select { color: #f2f4f7; }
.fake-rail { display: none; }
.topbar, .soft-panel, .askbox, [data-testid="stChatMessage"], div[data-testid="stMetric"] {
    background: #141414; border-color: #27272a; box-shadow: none;
}
.landing h1 { color: #b8c0cc; font-size: 1.55rem; }
textarea, input { background: #151515 !important; color: #f2f4f7 !important; border-color: #303036 !important; }
.chip { background: #141414; border-color: #303036; color: #d0d5dd; }
.ok { background: #102319; border-color: #236c43; }
.warn { background: #2a210d; border-color: #8a641c; }
.bad { background: #2a1212; border-color: #7a2f2f; }
div.stButton > button, div.stDownloadButton > button {
    background: #171717; border: 1px solid #303036; color: #f2f4f7;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    border-color: #6b7280; color: #ffffff;
}
div.stButton > button[kind="primary"] { background: #2f3136; color: #fff; }
.assistant-card {
    border: 1px solid #26272b; border-radius: 16px; background: #121212; padding: 14px;
}
.sidebar-brand { font-size: 1.35rem; font-weight: 800; color: #f2f4f7; margin-bottom: 1.1rem; }
.sidebar-muted { color: #8b949e; font-size: .78rem; }
</style>
""",
    unsafe_allow_html=True,
)


KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROK_API_KEY",
    "GOOGLE_API_KEY",
    "HF_TOKEN",
    "OPENROUTER_API_KEY",
    "CUSTOM_LLM_API_KEY",
    "CUSTOM_LLM_BASE_URL",
    "CUSTOM_LLM_MODEL",
    "DATABASE_URL",
    "TAVILY_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
    "PINECONE_NAMESPACE",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "WHATSAPP_TOKEN",
    "WHATSAPP_PHONE_NUMBER_ID",
    "WHATSAPP_BUSINESS_ACCOUNT_ID",
)

WORKFLOWS = [
    "Chat",
    "Agent chat",
    "Metrics",
    "Ask suggestions",
    "Vector knowledge",
    "Live search",
    "Ingest latest updates",
    "AI policy scan",
    "Relationship manager",
    "School clerk",
    "Study quiz",
    "Website",
    "App blueprint",
    "Codex workflow",
    "Template",
    "Voiceover",
    "WhatsApp automation",
    "Marketing",
    "Media inventory",
    "Mindmap",
    "Visual maps",
    "Integrations",
    "Swarm",
    "Toolbox",
    "Compliance",
    "Metadata",
]

RESPONSE_LANGUAGES = {
    "Auto": "auto",
    "English": "English",
    "Hindi": "Hindi in Devanagari script",
    "Urdu": "Urdu",
    "Arabic": "Arabic",
    "Bengali": "Bengali",
    "Tamil": "Tamil",
    "Telugu": "Telugu",
    "Marathi": "Marathi",
}


def load_secret_env() -> None:
    for key in KEYS:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])


def save_upload(file: Any) -> Path:
    root = Path(tempfile.gettempdir()) / "mas_ai_uploads"
    root.mkdir(exist_ok=True)
    path = root / file.name
    path.write_bytes(file.getbuffer())
    return path


def safe_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _export_text(content: str | bytes) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)


def _html_document(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; max-width: 920px; margin: 40px auto; padding: 0 20px; line-height: 1.58; color: #101828; }}
    pre {{ white-space: pre-wrap; background: #f6f7f9; border: 1px solid #e5e7eb; border-radius: 10px; padding: 18px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <pre>{escape(body)}</pre>
</body>
</html>"""


def _csv_document(text_value: str) -> str:
    buffer = StringIO()
    try:
        data = json.loads(text_value)
    except Exception:
        data = [{"content": line} for line in text_value.splitlines() if line.strip()] or [{"content": text_value}]
    if isinstance(data, dict):
        rows = [{"key": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v} for k, v in data.items()]
    elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
        rows = data
    else:
        rows = [{"value": json.dumps(data, ensure_ascii=False)}]
    fields = sorted({str(k) for row in rows for k in row.keys()}) or ["value"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fields})
    return buffer.getvalue()


def _svg_image(title: str, text_value: str) -> str:
    lines = []
    for line in text_value.splitlines() or [text_value]:
        lines.extend(textwrap.wrap(line, width=95) or [""])
        if len(lines) >= 52:
            break
    height = 110 + max(1, len(lines)) * 24
    text_nodes = "\n".join(
        f'<text x="48" y="{112 + i * 24}" class="body">{escape(line)}</text>' for i, line in enumerate(lines)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">
  <rect width="1200" height="{height}" fill="#ffffff"/>
  <rect x="28" y="28" width="1144" height="{height - 56}" rx="18" fill="#f8fafc" stroke="#d0d5dd"/>
  <text x="48" y="74" class="title">{escape(title)}</text>
  {text_nodes}
  <style>
    .title {{ font: 700 28px Arial, sans-serif; fill: #101828; }}
    .body {{ font: 18px Arial, sans-serif; fill: #1d2939; }}
  </style>
</svg>"""


def _png_image(title: str, text_value: str) -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.load_default()
        lines = []
        for line in text_value.splitlines() or [text_value]:
            lines.extend(textwrap.wrap(line, width=105) or [""])
            if len(lines) >= 60:
                break
        width, height = 1400, 130 + max(1, len(lines)) * 22
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=18, fill="#f8fafc", outline="#d0d5dd")
        draw.text((48, 52), title, fill="#101828", font=font)
        y = 100
        for line in lines:
            draw.text((48, y), line, fill="#1d2939", font=font)
            y += 22
        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return _svg_image(title, text_value).encode("utf-8")


def _plain_pdf(title: str, text_value: str) -> bytes:
    safe_lines = [title, ""] + [x for line in text_value.splitlines() for x in (textwrap.wrap(line, width=92) or [""])]
    pages = [safe_lines[i:i + 44] for i in range(0, len(safe_lines), 44)] or [[title]]
    objects: List[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    page_refs = []
    for page in pages:
        stream_lines = ["BT", "/F1 10 Tf", "50 770 Td", "14 TL"]
        for line in page:
            safe = line.encode("latin-1", errors="replace").decode("latin-1").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_lines.append(f"({safe}) Tj")
            stream_lines.append("T*")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
        content_id = len(objects) + 2
        page_obj = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode("ascii")
        page_refs.append(f"{len(objects) + 1} 0 R")
        objects.append(page_obj)
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(page_refs)}] /Count {len(page_refs)} >>".encode("ascii")
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode("ascii"))
    return bytes(pdf)


def _pdf_document(title: str, text_value: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        out = BytesIO()
        pdf = canvas.Canvas(out, pagesize=letter)
        width, height = letter
        y = height - 54
        pdf.setTitle(title)
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(48, y, title[:95])
        y -= 28
        pdf.setFont("Helvetica", 10)
        for line in [x for row in text_value.splitlines() for x in (textwrap.wrap(row, width=92) or [""])]:
            if y < 54:
                pdf.showPage()
                y = height - 54
                pdf.setFont("Helvetica", 10)
            pdf.drawString(48, y, line[:140])
            y -= 14
        pdf.save()
        return out.getvalue()
    except Exception:
        return _plain_pdf(title, text_value)


def _zip_bundle(label: str, content: str | bytes, filename: str, mime: str) -> bytes:
    text_value = _export_text(content)
    stem = Path(filename).stem or "generated_output"
    out = BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        original_data = content if isinstance(content, bytes) else content.encode("utf-8")
        zf.writestr(filename or f"{stem}.txt", original_data)
        zf.writestr(f"{stem}.txt", text_value)
        zf.writestr(f"{stem}.md", text_value)
        zf.writestr(f"{stem}.html", _html_document(label, text_value))
        zf.writestr(f"{stem}.csv", _csv_document(text_value))
        zf.writestr(f"{stem}.svg", _svg_image(label, text_value))
        zf.writestr(f"{stem}.pdf", _pdf_document(label, text_value))
        zf.writestr("manifest.json", json.dumps({"label": label, "original_filename": filename, "mime": mime}, indent=2))
    return out.getvalue()


def _export_variant(label: str, content: str | bytes, filename: str, mime: str, fmt: str) -> tuple[bytes, str, str]:
    text_value = _export_text(content)
    stem = Path(filename).stem or "generated_output"
    if fmt == "Original":
        return (content if isinstance(content, bytes) else content.encode("utf-8"), filename, mime)
    if fmt == "PDF":
        return _pdf_document(label, text_value), f"{stem}.pdf", "application/pdf"
    if fmt == "PNG image":
        return _png_image(label, text_value), f"{stem}.png", "image/png"
    if fmt == "SVG image":
        return _svg_image(label, text_value).encode("utf-8"), f"{stem}.svg", "image/svg+xml"
    if fmt == "ZIP bundle":
        return _zip_bundle(label, content, filename, mime), f"{stem}_bundle.zip", "application/zip"
    if fmt == "HTML":
        return _html_document(label, text_value).encode("utf-8"), f"{stem}.html", "text/html"
    if fmt == "JSON":
        try:
            data = json.loads(text_value)
        except Exception:
            data = {"label": label, "content": text_value}
        return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"), f"{stem}.json", "application/json"
    if fmt == "CSV":
        return _csv_document(text_value).encode("utf-8"), f"{stem}.csv", "text/csv"
    if fmt == "Markdown":
        return text_value.encode("utf-8"), f"{stem}.md", "text/markdown"
    return text_value.encode("utf-8"), f"{stem}.txt", "text/plain"


def download(label: str, content: str | bytes, filename: str, mime: str) -> None:
    if os.getenv("REQUIRE_HUMAN_EXPORT_APPROVAL", "true").lower() == "true":
        if not st.checkbox(f"Human approves export: {label}", key=f"approve_{safe_key(label + filename)}"):
            st.caption("Export waits for human approval.")
            return
    data = content if isinstance(content, bytes) else content.encode("utf-8")
    key = safe_key(filename + label)
    st.download_button(f"Download {label}", data, filename, mime, key=f"download_{key}")
    with st.expander("More export formats", expanded=False):
        fmt = st.selectbox(
            "Format",
            ["Original", "PDF", "PNG image", "SVG image", "ZIP bundle", "Markdown", "Text", "HTML", "JSON", "CSV"],
            key=f"export_format_{key}",
        )
        export_data, export_name, export_mime = _export_variant(label, content, filename, mime, fmt)
        st.download_button(f"Download as {fmt}", export_data, export_name, export_mime, key=f"download_more_{key}_{safe_key(fmt)}")


def render_chat(user_text: str, answer: str, meta: str = "") -> None:
    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
    with st.chat_message("assistant"):
        if meta:
            st.caption(meta)
        st.markdown(answer)


def render_sources(sources: List[Dict[str, Any]], label: str = "Evidence") -> None:
    if sources:
        with st.expander(label, expanded=False):
            st.text(format_context(sources, max_chars=14000))


def mermaid(code: str, height: int = 560) -> None:
    components.html(
        f"""
<div class="mermaid">{escape(code)}</div>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
mermaid.initialize({{startOnLoad: true, theme: "base"}});
</script>
""",
        height=height,
        scrolling=True,
    )


def browser_speak_button(text: str, key: str) -> None:
    payload = json.dumps((text or "")[:6000])
    components.html(
        f"""
<div style="display:flex;gap:8px;align-items:center;margin:6px 0 10px">
  <button id="speak_{key}" style="border:1px solid #d0d5dd;border-radius:999px;background:#101828;color:white;padding:7px 13px;cursor:pointer">Speak</button>
  <button id="stop_{key}" style="border:1px solid #d0d5dd;border-radius:999px;background:white;color:#101828;padding:7px 13px;cursor:pointer">Stop</button>
  <span style="font:12px system-ui;color:#667085">browser voice</span>
</div>
<script>
const text_{key} = {payload};
document.getElementById("speak_{key}").onclick = () => {{
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text_{key});
  utterance.rate = 0.95;
  utterance.pitch = 1;
  window.speechSynthesis.speak(utterance);
}};
document.getElementById("stop_{key}").onclick = () => window.speechSynthesis.cancel();
</script>
""",
        height=58,
    )


def render_voice_controls(text: str, label: str, engine: str, enabled: bool) -> None:
    if not enabled or not text:
        return
    key = safe_key(label + text[:80])
    with st.expander("Talk", expanded=True):
        browser_speak_button(text, key)
        if engine != "browser_speech":
            if st.button("Generate audio file", key=f"gen_audio_{key}"):
                audio = synthesize_speech(text, engine=engine, voice=os.getenv("OPENAI_TTS_VOICE", "alloy"), language=os.getenv("OCR_LANG", ""))
                if audio.get("ok"):
                    st.audio(audio["audio"], format=audio["mime"])
                    download("spoken answer", audio["audio"], f"{label}_{key}.{audio['ext']}", audio["mime"])
                else:
                    st.info(audio.get("note", "Audio generation is not available for this engine."))


def apply_provider(choice: Dict[str, str]) -> str:
    provider = choice["provider"]
    os.environ["LLM_PROVIDER"] = provider
    setters = {
        "openai": ("OPENAI_MODEL", "model"),
        "grok": ("GROK_MODEL", "model"),
        "gemini": ("GEMINI_MODEL", "model"),
        "huggingface": ("HF_MODEL", "model"),
        "openrouter": ("OPENROUTER_MODEL", "model"),
        "ollama": ("OLLAMA_MODEL", "model"),
        "custom": ("CUSTOM_LLM_MODEL", "model"),
    }
    if provider in setters:
        env_name, field = setters[provider]
        os.environ[env_name] = choice.get(field, "")
    if provider in {"huggingface", "openrouter", "ollama", "custom"}:
        base_env = {
            "huggingface": "HF_BASE_URL",
            "openrouter": "OPENROUTER_BASE_URL",
            "ollama": "OLLAMA_BASE_URL",
            "custom": "CUSTOM_LLM_BASE_URL",
        }[provider]
        os.environ[base_env] = choice.get("base_url") or os.getenv(base_env, "")
    if provider == "custom" and choice.get("key_env"):
        os.environ["CUSTOM_LLM_API_KEY_ENV"] = choice["key_env"]
    return provider


def language_query(question: str, selected: str) -> str:
    target = RESPONSE_LANGUAGES.get(selected, "auto")
    os.environ["RESPONSE_LANGUAGE"] = target
    if target == "auto":
        return question
    return (
        f"{question}\n\n"
        f"Answer in {target}. Translate the final answer, preserve citations, units, names, numbers, formulas, "
        "and quote evidence exactly where needed. Do not change the source meaning."
    )


def render_live_exam() -> None:
    exam = st.session_state.get("live_exam")
    if not exam:
        return
    items = exam.get("items", [])
    if not items:
        st.warning(exam.get("message", "No quiz items were generated from the evidence."))
        return
    submitted = exam.setdefault("submitted", {})
    total = sum(int(item.get("points", 0)) for item in items)
    earned = sum(int(row.get("points", 0)) for row in submitted.values())
    c1, c2, c3 = st.columns(3)
    c1.metric("Answered", f"{len(submitted)}/{len(items)}")
    c2.metric("Score", f"{earned}/{total}")
    c3.metric("Accuracy", f"{round((earned / total) * 100) if total else 0}%")
    st.progress(len(submitted) / len(items))

    if st.button("Reset exam"):
        st.session_state.pop("live_exam", None)
        st.rerun()

    for idx, item in enumerate(items, start=1):
        qid = item.get("id", str(idx))
        saved = submitted.get(qid)
        with st.container(border=True):
            st.markdown(f"**Q{idx}. {item['question']}**")
            st.caption(f"{item.get('points', 0)} point(s) | {item.get('source')} p.{item.get('page')} [{item.get('section')}]")
            if saved:
                selected_index = int(saved["selected_index"])
                st.radio("Options", item["options"], index=selected_index, key=f"locked_{qid}", disabled=True)
            else:
                selected = st.radio("Options", item["options"], index=None, key=f"choice_{qid}")
                if st.button("Submit answer", key=f"submit_{qid}"):
                    if selected is None:
                        st.warning("Select an answer first.")
                    else:
                        selected_index = item["options"].index(selected)
                        correct = selected_index == int(item["correct_index"])
                        submitted[qid] = {
                            "selected_index": selected_index,
                            "correct": correct,
                            "points": int(item.get("points", 0)) if correct else 0,
                        }
                        st.rerun()
            if saved:
                remarks = item.get("option_feedback", [])
                selected_index = int(saved["selected_index"])
                if saved.get("correct"):
                    st.success(f"Correct. +{item.get('points', 0)} point(s).")
                else:
                    st.error("Incorrect. +0 points.")
                if remarks and selected_index < len(remarks):
                    st.info(remarks[selected_index])
                st.markdown(f"**Correct answer:** {item['options'][int(item['correct_index'])]}")
                st.caption("Why: " + item.get("explanation", "Grounded in the cited evidence."))
                if remarks:
                    st.markdown("**Remarks for all options**")
                    for opt, note in zip(item["options"], remarks):
                        st.markdown(f"- **{opt}:** {note}")

    if len(submitted) == len(items):
        rows = []
        for idx, item in enumerate(items, start=1):
            saved = submitted.get(item.get("id", str(idx)))
            selected_index = int(saved["selected_index"]) if saved else -1
            rows.append(
                {
                    "Q": idx,
                    "Selected": item["options"][selected_index] if saved else "Not answered",
                    "Correct": item["options"][int(item["correct_index"])],
                    "Result": "Correct" if saved and saved.get("correct") else "Wrong",
                    "Points": saved.get("points", 0) if saved else 0,
                    "Reason": item.get("explanation", ""),
                }
            )
        st.success(f"Final score: {earned}/{total}")
        st.dataframe(rows, use_container_width=True)
        download("score card", json.dumps({"score": earned, "total": total, "rows": rows}, indent=2), "score_card.json", "application/json")


def status_chips(meta: Dict[str, Any]) -> None:
    chips = [
        ("ok" if os.getenv("HUMAN_REVIEW_CONFIRMED") == "true" else "warn", "Human review " + ("on" if os.getenv("HUMAN_REVIEW_CONFIRMED") == "true" else "pending")),
        ("ok" if os.getenv("DPDP_REDACT", "true") == "true" else "bad", "Redaction " + os.getenv("DPDP_REDACT", "true")),
        ("ok" if os.getenv("TAVILY_API_KEY") else "warn", "Tavily " + ("ready" if os.getenv("TAVILY_API_KEY") else "not set")),
        ("", f"Sources {meta.get('source_count', 0)}"),
    ]
    st.markdown("".join(f'<span class="chip {style}">{text}</span>' for style, text in chips), unsafe_allow_html=True)


def queue_prompt(prompt: str, action: str = "Chat", auto_run: bool = False) -> None:
    st.session_state["pending_brief_text"] = prompt
    st.session_state["forced_action"] = action
    st.session_state["pending_auto_run"] = auto_run


def remember_chat(question: str, answer: str, action: str) -> None:
    if not question and not answer:
        return
    history = st.session_state.setdefault("assistant_history", [])
    history.insert(0, {"question": question[:240], "answer": answer[:420], "action": action})
    del history[12:]


def new_chat() -> None:
    for key in ("brief_text", "pending_brief_text", "pending_auto_run", "forced_action", "live_exam"):
        st.session_state.pop(key, None)
    st.session_state["brief_text"] = ""


load_secret_env()

if st.session_state.get("pending_brief_text"):
    st.session_state["brief_text"] = st.session_state.pop("pending_brief_text")
if "brief_text" not in st.session_state:
    st.session_state["brief_text"] = ""
if "assistant_history" not in st.session_state:
    st.session_state["assistant_history"] = []

with st.sidebar:
    st.markdown('<div class="sidebar-brand">MAS AI</div>', unsafe_allow_html=True)
    if st.button("Chat", use_container_width=True):
        st.session_state["forced_action"] = "Chat"
    if st.button("Prompts", use_container_width=True):
        st.session_state["show_prompt_library"] = not st.session_state.get("show_prompt_library", False)
    if st.button("AI Specialists", use_container_width=True):
        st.session_state["show_agent_library"] = not st.session_state.get("show_agent_library", False)
    if st.session_state.get("show_prompt_library", False):
        st.caption("Prompt library")
        prompt_bank = [
            ("Summarize document", "Summarize the uploaded document with citations and limitations.", "Chat"),
            ("Summarize webpage", "Summarize the permitted webpage URL with citations.", "Chat"),
            ("Quiz from evidence", "Create a source-grounded quiz with answers and remarks.", "Study quiz"),
            ("Website copy", "Build a website page with SEO, critic review, and source evidence.", "Website"),
            ("Mindmap", "Create a mindmap and flowchart from the uploaded evidence.", "Mindmap"),
            ("Marketing post", "Create a compliant LinkedIn and X post grounded in the uploaded evidence.", "Marketing"),
        ]
        for label, prompt, action_name in prompt_bank:
            if st.button(label, key=f"side_prompt_{safe_key(label)}", use_container_width=True):
                queue_prompt(prompt, action_name)
                st.rerun()
    if st.session_state.get("show_agent_library", False):
        st.caption("Specialists")
        specialists = [
            ("Research RAG", "Agent chat"),
            ("Relationship manager", "Relationship manager"),
            ("School clerk", "School clerk"),
            ("Website builder", "Website"),
            ("Marketing", "Marketing"),
            ("Compliance", "Compliance"),
            ("Visual maps", "Visual maps"),
            ("Toolbox", "Toolbox"),
        ]
        for label, action_name in specialists:
            if st.button(label, key=f"side_agent_{safe_key(label)}", use_container_width=True):
                queue_prompt(st.session_state.get("brief_text", "") or f"Run {label.lower()} for my current task.", action_name)
                st.rerun()
    st.divider()
    if st.button("New Chat", use_container_width=True):
        new_chat()
        st.rerun()
    if st.button("History", use_container_width=True):
        st.session_state["show_history"] = not st.session_state.get("show_history", False)
    if st.session_state.get("show_history", False):
        st.caption("Recent outputs")
        if not st.session_state["assistant_history"]:
            st.markdown('<div class="sidebar-muted">No history yet.</div>', unsafe_allow_html=True)
        for i, item in enumerate(st.session_state["assistant_history"][:6], start=1):
            with st.expander(f"{i}. {item['action']}"):
                st.caption(item["question"] or "No question")
                st.write(item["answer"] or "No answer")
    st.divider()
    st.button("Subscribe", disabled=True, use_container_width=True)
    st.button("Settings", disabled=True, use_container_width=True)
    st.markdown('<div class="sidebar-muted">Human-in-loop scientific RAG workspace</div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="fake-rail">
  <div class="rail-dot">◎</div>
  <div class="rail-icon">✎</div>
  <div class="rail-icon">⌕</div>
  <div class="rail-icon">○</div>
  <div class="rail-avatar">AT</div>
</div>
""",
    unsafe_allow_html=True,
)

header_left, header_center, header_right = st.columns([2, 3, 5], vertical_alignment="center")
with header_left:
    st.markdown('<div class="brand-select">MAS AI</div>', unsafe_allow_html=True)
with header_center:
    if st.button("New Chat", use_container_width=True):
        new_chat()
        st.rerun()
with header_right:
    top_buttons = st.columns([1, 1, 1, 1, 1], gap="small")

preset = "Balanced"
density = "Compact"
defaults = {"retrieval": "TF-IDF", "chunking": "section_semantic", "k": 8}
uploads: List[Any] = []
local_path = ""
urls = ""
jurisdiction = "India"
fetch_ok = False
use_tavily = False
extra_models = ""
forced_action = st.session_state.get("forced_action", "")
manual = bool(forced_action)
manual_action = forced_action if forced_action in WORKFLOWS else "Chat"
response_language = "Auto"
auto_mic_run = True
voice_reply = False
assistant_tts_engine = "browser_speech"
provider = os.getenv("LLM_PROVIDER", "local")
retrieval = "TF-IDF"
top_k = 8

with top_buttons[0].popover("Files", use_container_width=True):
    preset = st.selectbox("Preset", ["Balanced", "Fast", "Deep Research", "Offline"], index=0)
    density = st.radio("Density", ["Compact", "Comfortable", "Ultra"], horizontal=True)
    defaults = {
        "Fast": {"retrieval": "TF-IDF", "chunking": "section_semantic", "k": 5},
        "Balanced": {"retrieval": "TF-IDF", "chunking": "section_semantic", "k": 8},
        "Deep Research": {"retrieval": "OpenAI text-embedding-3-large", "chunking": "mbert", "k": 12},
        "Offline": {"retrieval": "TF-IDF", "chunking": "section_semantic", "k": 6},
    }[preset]
    uploads = st.file_uploader(
        "Evidence files",
        type=["zip", "pdf", "txt", "md", "csv", "tsv", "xlsx", "xls", "json", "png", "jpg", "jpeg", "webp", "srt", "vtt"],
        accept_multiple_files=True,
    )
    local_path = st.text_input("Local path", placeholder="Optional local file/folder")
    urls = st.text_area("URLs", placeholder="https://example.org/page", height=76)
    jurisdiction = st.selectbox("Jurisdiction", ["India", "EU/EEA", "California", "UK", "Global/Unknown"])
    os.environ["COMPLIANCE_JURISDICTION"] = jurisdiction
    fetch_ok = st.checkbox("URL fetch permitted by law, robots.txt, and terms")
    use_tavily = st.checkbox("Use Tavily live search", value=preset == "Deep Research")
    st.caption("Tavily key is read from Streamlit secrets only; no key field is shown.")

with top_buttons[1].popover("Tools", use_container_width=True):
    manual = st.toggle("Manual node selection", value=manual)
    manual_action = st.selectbox("Node / function", WORKFLOWS, index=WORKFLOWS.index(manual_action), disabled=not manual)
    if manual:
        st.session_state["forced_action"] = manual_action
    elif st.session_state.get("forced_action"):
        st.session_state.pop("forced_action", None)
    st.caption("When manual mode is off, the orchestration manager selects the node from the query, files, mic, URL, and live-search state.")
    with st.expander("Accessible node map", expanded=False):
        st.markdown("\n".join(f"- {node}" for node in WORKFLOWS))

with top_buttons[2].popover("Model", use_container_width=True):
    extra_models = st.text_area("Extra LLMs", placeholder="Label, provider, model, base_url, key_env", height=68)
    models = llm_model_catalog(extra_models)
    free = [m for m in models if m.get("requires_key") == "no"]
    paid = [m for m in models if m.get("requires_key") != "no"]
    model_group = st.radio("Group", ["Free / no key", "Paid / key required"], horizontal=True)
    choices = free if model_group.startswith("Free") else paid
    model_search = st.text_input("Search models", placeholder="Search")
    if model_search:
        choices = [m for m in choices if model_search.lower() in m["label"].lower() or model_search.lower() in m.get("provider", "").lower()]
    st.caption("Basic Models" if model_group.startswith("Free") else "Advanced / key-required models")
    if not choices:
        st.warning("No model matches the search.")
        choices = free or models[:1]
    selected = st.selectbox("Model", [m["label"] for m in choices])
    selected_model = choices[[m["label"] for m in choices].index(selected)]
    provider = apply_provider(selected_model)
    key_env = selected_model.get("key_env", "")
    if model_group.startswith("Paid") and key_env:
        if os.getenv(key_env):
            st.caption(f"{key_env} is loaded from secrets/env and remains hidden.")
        else:
            pasted = st.text_input(f"Paste {key_env}", type="password")
            if pasted:
                os.environ[key_env] = pasted
    else:
        st.caption("No key is required for this model.")
    with st.expander("Custom/Ollama endpoint"):
        os.environ["OLLAMA_BASE_URL"] = st.text_input("Ollama base URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
        os.environ["OLLAMA_MODEL"] = st.text_input("Ollama model", os.getenv("OLLAMA_MODEL", "llama3.1"))
        os.environ["CUSTOM_LLM_BASE_URL"] = st.text_input("Custom base URL", os.getenv("CUSTOM_LLM_BASE_URL", ""))
        os.environ["CUSTOM_LLM_MODEL"] = st.text_input("Custom model", os.getenv("CUSTOM_LLM_MODEL", ""))
        os.environ["CUSTOM_LLM_API_KEY_ENV"] = st.text_input("Custom key env", os.getenv("CUSTOM_LLM_API_KEY_ENV", "CUSTOM_LLM_API_KEY"))

with top_buttons[3].popover("Process", use_container_width=True):
    c1, c2 = st.columns(2)
    retrieval = c1.selectbox("Search", ["TF-IDF", "MiniLM semantic", "OpenAI text-embedding-3-large", "Pinecone"], index=["TF-IDF", "MiniLM semantic", "OpenAI text-embedding-3-large", "Pinecone"].index(defaults["retrieval"]))
    os.environ["ENABLE_MINILM_RETRIEVER"] = str(retrieval == "MiniLM semantic").lower()
    chunking = c2.selectbox("Chunking", ["section_semantic", "mbert"], index=["section_semantic", "mbert"].index(defaults["chunking"]))
    os.environ["CHUNKING_ENGINE"] = chunking
    ocr_rows = ocr_model_options()
    ocr_choice = st.selectbox("OCR", [f"{m['label']} | {m['pricing']}" for m in ocr_rows])
    os.environ["OCR_ENGINE"] = ocr_rows[[f"{m['label']} | {m['pricing']}" for m in ocr_rows].index(ocr_choice)]["engine"]
    lang_rows = ocr_language_options()
    lang_choice = st.selectbox("OCR language", [f"{m['label']} ({m['code']})" for m in lang_rows])
    lang_row = lang_rows[[f"{m['label']} ({m['code']})" for m in lang_rows].index(lang_choice)]
    os.environ["OCR_LANG"] = st.text_input("Custom OCR code", os.getenv("OCR_LANG", "eng+hin+urd")) if lang_row["code"] == "custom" else lang_row["code"]
    trans_rows = transliteration_options()
    trans = st.selectbox("Transliteration", [m["label"] for m in trans_rows])
    os.environ["TRANSLITERATION_ENGINE"] = trans_rows[[m["label"] for m in trans_rows].index(trans)]["engine"]
    response_language = st.selectbox("Answer language", list(RESPONSE_LANGUAGES), index=0)
    stt_rows = speech_to_text_options()
    stt = st.selectbox("Speech to text", [m["label"] for m in stt_rows])
    os.environ["STT_ENGINE"] = stt_rows[[m["label"] for m in stt_rows].index(stt)]["engine"]
    auto_mic_run = st.checkbox("Auto-run mic transcript", value=True)
    voice_reply = st.checkbox("Talk back", value=False)
    tts_rows = text_to_speech_options()
    tts_choice = st.selectbox("Assistant voice", [m["label"] for m in tts_rows], index=0)
    assistant_tts_engine = tts_rows[[m["label"] for m in tts_rows].index(tts_choice)]["engine"]
    os.environ["ASSISTANT_TTS_ENGINE"] = assistant_tts_engine
    if assistant_tts_engine == "openai_tts":
        os.environ["OPENAI_TTS_VOICE"] = st.selectbox("OpenAI voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], index=0)
    if assistant_tts_engine == "edge_tts":
        os.environ["EDGE_TTS_VOICE"] = st.text_input("Edge voice", os.getenv("EDGE_TTS_VOICE", "en-IN-NeerjaNeural"))
    top_k = st.slider("Evidence depth", 3, 15, defaults["k"])

with top_buttons[4].popover("Guardrails", use_container_width=True):
    lawful = st.checkbox("Lawful basis/consent confirmed")
    cloud = st.checkbox("Allow cloud processing")
    os.environ["DPDP_LAWFUL_BASIS"] = str(lawful).lower()
    os.environ["DPDP_CLOUD_CONSENT"] = str(lawful and cloud).lower()
    os.environ["DPDP_REDACT"] = str(st.checkbox("Redact identifiers", value=True)).lower()
    os.environ["HUMAN_REVIEW_CONFIRMED"] = str(st.checkbox("Human reviewer remains responsible")).lower()
    os.environ["REQUIRE_HUMAN_EXPORT_APPROVAL"] = str(st.checkbox("Require approval before export", value=True)).lower()

paths = [save_upload(f) for f in uploads] if uploads else []
if local_path:
    paths.append(Path(local_path))
explicit_urls = extract_urls_from_text(urls)
attachment_urls = extract_urls_from_paths(paths) if paths else []
detected_urls = list(dict.fromkeys([*explicit_urls, *attachment_urls]))
web_urls = detected_urls[:] if fetch_ok else []
detected_url_note = ""
if detected_urls and not fetch_ok:
    detected_url_note = f" Detected {len(detected_urls)} URL(s) in pasted text or attachments; enable URL permission to transcribe and chunk them."

with st.spinner("Indexing evidence"):
    corpus, summary = build_corpus_from_paths(paths) if paths else ([], "No uploaded files.")
    if web_urls:
        web_corpus, web_summary = build_corpus_from_urls(web_urls, jurisdiction)
        corpus.extend(web_corpus)
        summary += " " + web_summary
    if detected_url_note:
        summary += detected_url_note
    cid = corpus_id(paths, web_urls)
    if corpus:
        save_corpus_pg(corpus, cid)
        supabase_log_metadata(corpus_metadata(corpus, cid))
        if retrieval == "Pinecone":
            pinecone_upsert(corpus, cid)

metadata = corpus_metadata(corpus, cid)

st.markdown('<div class="tool-strip">', unsafe_allow_html=True)
status_chips(metadata)
st.markdown("</div>", unsafe_allow_html=True)

chat_tab, evidence_tab, studio_tab = st.tabs(["Chat", "Evidence", "Studio"])

with chat_tab:
    st.markdown('<div class="landing"><h1>How can I help you today?</h1></div>', unsafe_allow_html=True)
    quick_cols = st.columns(6)
    quick_actions = [
        ("Summarize Webpage", "Summarize the permitted webpage URL with citations.", "Chat"),
        ("Summarize Document", "Summarize the uploaded document with citations, key findings, and limitations.", "Chat"),
        ("Chat with Webpage", "Answer my question using only the permitted webpage URL evidence.", "Chat"),
        ("LinkedIn Post", "Create a LinkedIn post grounded only in uploaded or permitted evidence.", "Marketing"),
        ("X Post", "Create a concise X post thread grounded only in uploaded or permitted evidence.", "Marketing"),
        ("View All Agents", "Show all available agents and tools.", "Toolbox"),
    ]
    for idx, (label, prompt, action_name) in enumerate(quick_actions):
        if quick_cols[idx].button(label, key=f"quick_{safe_key(label)}", use_container_width=True):
            queue_prompt(prompt, action_name)
            st.rerun()
    prompt_cols = st.columns([.55, 8, .75, .75], vertical_alignment="center")
    with prompt_cols[0].popover("+", use_container_width=True):
        st.caption("Attach evidence without opening the setup drawer.")
        extra_files = st.file_uploader(
            "Add files",
            type=["zip", "pdf", "txt", "md", "csv", "tsv", "xlsx", "xls", "json", "png", "jpg", "jpeg", "webp", "srt", "vtt"],
            accept_multiple_files=True,
            key="extra_files",
        )
        if extra_files:
            extra_paths = [save_upload(f) for f in extra_files]
            more_corpus, more_summary = build_corpus_from_paths(extra_paths)
            extra_urls = extract_urls_from_paths(extra_paths)
            if extra_urls and fetch_ok:
                extra_web, extra_web_summary = build_corpus_from_urls(extra_urls, jurisdiction)
                more_corpus.extend(extra_web)
                more_summary += " " + extra_web_summary
                web_urls.extend([u for u in extra_urls if u not in web_urls])
            elif extra_urls:
                more_summary += f" Detected {len(extra_urls)} URL(s); enable URL permission to transcribe and chunk them."
            corpus.extend(more_corpus)
            cid = corpus_id(paths + extra_paths, web_urls)
            metadata = corpus_metadata(corpus, cid)
            st.caption(more_summary)
    with prompt_cols[1]:
        brief = st.text_area(
            "Brief / query",
            height=68 if density != "Ultra" else 48,
            placeholder="Tell me something about this page" if urls.strip() else "Ask anything",
            key="brief_text",
            label_visibility="collapsed",
        )
    with prompt_cols[2].popover("Mic", use_container_width=True):
        mic_audio = st.audio_input("Mic") if hasattr(st, "audio_input") else None
        audio_upload = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg", "webm"])
        st.caption("Mic recordings transcribe automatically after you stop recording.")
        stt_engine = os.getenv("STT_ENGINE", "manual")
        stt_lang = os.getenv("OCR_LANG", "eng").split("+")[0]
        if mic_audio:
            mic_bytes = mic_audio.getvalue()
            mic_signature = safe_key(stt_engine + str(len(mic_bytes)) + hashlib.sha1(mic_bytes).hexdigest())
            if st.session_state.get("last_mic_transcript_signature") != mic_signature:
                st.session_state["last_mic_transcript_signature"] = mic_signature
                with st.spinner("Transcribing mic audio"):
                    transcript = transcribe_audio(mic_bytes, getattr(mic_audio, "name", "mic_input.wav"), stt_engine, stt_lang)
                if transcript.startswith("STT failed:") or transcript.startswith("STT engine"):
                    st.warning(transcript)
                else:
                    st.session_state["pending_brief_text"] = transcript
                    st.session_state["pending_auto_run"] = auto_mic_run
                    st.rerun()
            else:
                st.caption("Mic transcript is already in the prompt.")
        if audio_upload and st.button("Transcribe uploaded audio"):
            transcript = transcribe_audio(audio_upload.getvalue(), getattr(audio_upload, "name", "audio_upload.wav"), stt_engine, stt_lang)
            if transcript.startswith("STT failed:") or transcript.startswith("STT engine"):
                st.warning(transcript)
            else:
                st.session_state["pending_brief_text"] = transcript
                st.session_state["pending_auto_run"] = auto_mic_run
                st.rerun()
    with prompt_cols[3]:
        run = st.button("Run", type="primary", help="Run the smart orchestration pipeline")

    st.markdown('<div class="tool-strip">', unsafe_allow_html=True)
    suggestion_cols = st.columns(4)
    for i, suggestion in enumerate(ask_suggestions(corpus, 4)):
        label = suggestion.split("?")[0].strip()[:30] or f"Suggestion {i + 1}"
        if suggestion_cols[i].button(label, key=f"suggestion_{i}", help=suggestion):
            queue_prompt(suggestion, "Chat")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with studio_tab:
    with st.expander("Session", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chunks", len(corpus))
        c2.metric("Sources", metadata.get("source_count", 0))
        c3.metric("Provider", provider)
        c4.metric("Jurisdiction", jurisdiction)
        st.caption(summary)
    with st.expander("All available nodes and functions", expanded=False):
        st.markdown("\n".join(f"- {node}" for node in WORKFLOWS))
        st.dataframe(toolbox_catalog(), use_container_width=True)

with evidence_tab:
    search_query = brief or "summary"
    hits = pinecone_retrieve(corpus, search_query, top_k, cid) if retrieval == "Pinecone" else (
        embedding_retrieve(corpus, search_query, top_k) if corpus and retrieval.startswith("OpenAI") else (
            sentence_transformer_retrieve(corpus, search_query, top_k) if corpus and retrieval == "MiniLM semantic" else retrieve(corpus, search_query, top_k)
        )
    )
    st.caption(f"{len(hits)} retrieved chunks from {metadata.get('source_count', 0)} source(s).")
    st.text(format_context(hits) if hits else "No evidence yet. Upload files, add permitted URLs, or enable Tavily live search.")
    with st.expander("Metadata", expanded=False):
        st.json(metadata)

auto_run = bool(st.session_state.pop("pending_auto_run", False))
if not (run or auto_run or st.session_state.get("live_exam")):
    with chat_tab:
        st.info("Upload evidence or enable live search, then ask a question. Smart routing will choose the right tool.")
    st.stop()

prompt_urls = extract_urls_from_text(brief)
new_prompt_urls = [u for u in prompt_urls if u not in web_urls]
if prompt_urls and not fetch_ok:
    with chat_tab:
        st.warning("URL(s) detected in the query. Open Files and enable URL fetch permission so the app can transcribe, chunk, and cite them.")
if new_prompt_urls and fetch_ok:
    with st.spinner("Transcribing URL evidence"):
        prompt_corpus, prompt_summary = build_corpus_from_urls(new_prompt_urls, jurisdiction)
        corpus.extend(prompt_corpus)
        web_urls.extend(new_prompt_urls)
        summary += " " + prompt_summary
        cid = corpus_id(paths, web_urls)
        metadata = corpus_metadata(corpus, cid)
        if corpus:
            save_corpus_pg(corpus, cid)
            supabase_log_metadata(metadata)
            if retrieval == "Pinecone":
                pinecone_upsert(corpus, cid)
    with chat_tab:
        st.info(f"Transcribed and chunked {len(new_prompt_urls)} URL(s) from your query.")

query = language_query(brief, response_language)

if use_tavily and brief and (needs_live_search(brief) or manual and manual_action == "Live search"):
    with st.spinner("Collecting Tavily evidence"):
        live_corpus, live_summary = build_corpus_from_tavily(brief, max_results=5)
        corpus.extend(live_corpus)
        summary += " " + live_summary
        metadata = corpus_metadata(corpus, cid)

action = manual_action if manual else "Smart auto"
route: Dict[str, Any] | None = None
if action == "Smart auto":
    route = orchestration_manager_plan(
        brief,
        corpus,
        provider=provider,
        retrieval_engine=retrieval,
        live_search_enabled=use_tavily,
        jurisdiction=jurisdiction,
    )
    action = route["selected_action"]

with studio_tab:
    if route:
        with st.expander("Routing audit", expanded=False):
            st.metric("Selected", route["selected_action"])
            st.metric("Confidence", f"{int(route['confidence'] * 100)}%")
            st.caption(route["rationale"])
            st.dataframe(route.get("agents", []), use_container_width=True)
            st.dataframe(route.get("tools", []), use_container_width=True)
            st.json(route.get("evidence_state", {}))

with chat_tab:
    if auto_run:
        st.success("Mic transcript was routed through the smart workflow.")

    if action == "Chat":
        if not corpus and not use_tavily:
            result = {"answer": "No indexed evidence is available. Upload documents or enable Tavily live search for grounded answers.", "sources": [], "provider": "local", "model": "no-evidence"}
        else:
            result = asyncio.run(
                answer_rag_chat(
                    query,
                    corpus,
                    provider=provider,
                    top_k=top_k,
                    retrieval_engine="openai_embeddings" if retrieval in {"OpenAI text-embedding-3-large", "Pinecone"} else ("minilm" if retrieval == "MiniLM semantic" else "tfidf"),
                )
            )
        a_col, s_col = st.columns([3, 1])
        with a_col:
            render_chat(brief, result["answer"], f"{result.get('provider', provider)} / {result.get('model', '')}")
            render_voice_controls(result["answer"], "answer", assistant_tts_engine, voice_reply)
        with s_col:
            st.markdown("#### Evidence")
            st.caption(f"{len(result.get('sources', []))} chunks")
            render_sources(result.get("sources", []), "Retrieved")
            with st.expander("BLEU / ROUGE / METEOR", expanded=False):
                st.dataframe(result.get("eval_matrix", []), use_container_width=True)
            download("answer", json.dumps(result, indent=2), "answer.json", "application/json")
        log_query_pg(cid, brief, result["answer"], result.get("provider", ""), result.get("model", ""))
        log_rag_event(brief, result["answer"], result.get("provider", ""), result.get("model", ""), result.get("latency_s", 0.0), result.get("sources", []), retrieval)
        remember_chat(brief, result["answer"], "Chat")
        f1, f2 = st.columns(2)
        if f1.button("Good answer", key="feedback_chat_up"):
            log_feedback(brief, result["answer"], "up", provider=result.get("provider", ""), retrieval_mode=retrieval)
            st.success("Feedback saved.")
        if f2.button("Needs work", key="feedback_chat_down"):
            log_feedback(brief, result["answer"], "down", provider=result.get("provider", ""), retrieval_mode=retrieval)
            st.warning("Feedback saved for review.")

    elif action == "Agent chat":
        result = asyncio.run(answer_with_agent_pipeline_from_corpus(query, corpus, summary, provider))
        a_col, s_col = st.columns([3, 1])
        with a_col:
            render_chat(brief, result["answer"], f"{result.get('provider', provider)} / planner -> executor -> verifier")
            render_voice_controls(result["answer"], "agent_answer", assistant_tts_engine, voice_reply)
        with s_col:
            render_sources(result.get("sources", []), "Retrieved")
            with st.expander("BLEU / ROUGE / METEOR", expanded=False):
                st.dataframe(result.get("eval_matrix", []), use_container_width=True)
            with st.expander("Agent trace"):
                st.json(result.get("conversation", []))
            download("agent answer", json.dumps(result, indent=2), "agent_answer.json", "application/json")
        log_rag_event(brief, result["answer"], result.get("provider", ""), result.get("model", ""), result.get("latency_s", 0.0), result.get("sources", []), retrieval)
        remember_chat(brief, result["answer"], "Agent chat")
        f1, f2 = st.columns(2)
        if f1.button("Good agent answer", key="feedback_agent_up"):
            log_feedback(brief, result["answer"], "up", provider=result.get("provider", ""), retrieval_mode=retrieval)
            st.success("Feedback saved.")
        if f2.button("Agent needs work", key="feedback_agent_down"):
            log_feedback(brief, result["answer"], "down", provider=result.get("provider", ""), retrieval_mode=retrieval)
            st.warning("Feedback saved for review.")

    elif action == "Metrics":
        out = agent_metrics_session(brief, corpus, provider=provider, retrieval_engine=retrieval, jurisdiction=jurisdiction)
        st.markdown(out["markdown"])
        monitor = monitoring_summary()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Logged answers", monitor["events"])
        c2.metric("Feedback", monitor["feedback"])
        c3.metric("Positive", monitor["thumbs_up"])
        c4.metric("Avg latency", f"{monitor['avg_latency_s']}s")
        st.markdown("### RAG Quality Metrics")
        st.dataframe(out["metrics"], use_container_width=True)
        st.markdown("### BLEU / ROUGE / METEOR Matrix")
        st.dataframe(out["lexical_eval_matrix"], use_container_width=True)
        st.markdown("### Quality Gates")
        st.dataframe(out["quality_gates"], use_container_width=True)
        st.markdown("### RAG + API Integration Plan")
        st.markdown("\n".join(f"- {item}" for item in out["rag_api_plan"]))
        st.markdown("### MCP Server Plan")
        st.dataframe(out["mcp_server_plan"], use_container_width=True)
        with st.expander("Top evidence for metrics context", expanded=False):
            st.json(out["top_evidence"])
        report_path = write_evidently_report()
        if report_path.exists():
            download("metrics report", report_path.read_text(encoding="utf-8"), "rag_metrics_report.html", "text/html")
        download("metrics packet", json.dumps(out, indent=2), "metrics_mcp_readiness.json", "application/json")

    elif action == "Ask suggestions":
        out = ask_suggestions(corpus)
        st.markdown("\n".join(f"- {x}" for x in out))
        download("suggestions", json.dumps(out, indent=2), "suggestions.json", "application/json")

    elif action == "Vector knowledge":
        out = vector_space_knowledge(corpus, brief or "entire corpus", k=25)
        render_chat(brief, "I reviewed the indexed evidence space. Open the panels below for coverage and top evidence.", "Vector knowledge")
        st.json(out["summary"])
        render_sources(out["top_evidence"], "Top evidence")
        download("vector knowledge", json.dumps(out, indent=2), "vector_knowledge.json", "application/json")

    elif action == "Live search":
        out = vector_space_knowledge(corpus, brief or "live search", k=25)
        render_chat(brief, "Live/permitted evidence has been collected and indexed.", "Live search")
        st.json(out["summary"])
        render_sources(out["top_evidence"], "Live evidence")
        download("live search", json.dumps(out, indent=2), "live_search.json", "application/json")

    elif action == "Ingest latest updates":
        out = ingest_latest_updates(brief or "latest updates", "latest_updates", jurisdiction=jurisdiction, urls=web_urls, store_postgres=True, store_pinecone=True)
        st.json(out)
        download("latest updates", json.dumps(out, indent=2), "latest_updates.json", "application/json")

    elif action == "AI policy scan":
        profile = st.selectbox("Policy profile", ["All"] + [p["name"] for p in ai_policy_profiles()])
        out = ai_policy_scan(profile, jurisdiction)
        st.json(out)
        download("policy scan", json.dumps(out, indent=2), "ai_policy_scan.json", "application/json")

    elif action == "Relationship manager":
        out = relationship_manager_agent(brief, corpus, provider=provider)
        a_col, s_col = st.columns([3, 1])
        with a_col:
            render_chat(brief, out["answer"], f"{out['product_agent_response']['agent']} / {out['decision']['route']}")
            render_voice_controls(out["answer"], "relationship_manager", assistant_tts_engine, voice_reply)
            with st.expander("Architecture", expanded=False):
                mermaid(out["architecture"], height=520)
        with s_col:
            st.markdown("#### Decision")
            st.json(out["decision"])
            render_sources(out.get("sources", []), "Product evidence")
            with st.expander("Product agent response", expanded=False):
                st.json(out["product_agent_response"])
            with st.expander("BLEU / ROUGE / METEOR", expanded=False):
                st.dataframe(out.get("eval_matrix", []), use_container_width=True)
            download("relationship manager", json.dumps(out, indent=2), "relationship_manager.json", "application/json")
        remember_chat(brief, out["answer"], "Relationship manager")

    elif action == "School clerk":
        out = school_clerk_automation(brief, corpus)
        st.markdown(out["markdown"])
        if out.get("rows"):
            st.dataframe(out["rows"], use_container_width=True)
        with st.expander("Pro tips and approval checklist", expanded=True):
            st.markdown("\n".join(f"- {x}" for x in out.get("pro_tips", [])))
            st.markdown("\n".join(f"- {x}" for x in out.get("human_checklist", [])))
        if out.get("csv"):
            download("school result csv", out["csv"], "school_result_sheet.csv", "text/csv")
        download("school clerk packet", json.dumps(out, indent=2), "school_clerk.json", "application/json")

    elif action == "Study quiz":
        c1, c2, c3 = st.columns(3)
        exam = c1.text_input("Exam", "School / University Exam")
        difficulty = c2.selectbox("Difficulty", ["easy", "medium", "hard"])
        mode = c3.selectbox("Mode", ["question_paper", "pw_practice", "textbook_solution", "assertion_reason", "quiz", "flashcards"])
        count = st.slider("Questions", 5, 50, 10)
        live_mode = st.toggle("Live exam with scoring", value=mode in {"quiz", "pw_practice", "assertion_reason"})
        if live_mode and (run or auto_run):
            quiz = study_quiz_items(corpus, exam, brief or "uploaded syllabus", count, difficulty, mode)
            quiz["submitted"] = {}
            st.session_state["live_exam"] = quiz
        elif run or auto_run:
            st.session_state.pop("live_exam", None)
            out = study_quiz_generator(corpus, exam, brief or "uploaded syllabus", count, difficulty, mode)
            st.markdown(out)
            download("study quiz", out, f"{mode}.md", "text/markdown")
        render_live_exam()

    elif action == "Website":
        page = build_website(brief, corpus, "Evidence Studio", "Evidence-grounded publication")
        components.html(page["html"], height=580, scrolling=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Critic**")
            st.markdown("\n".join(f"- {x}" for x in page.get("critique", [])))
        with c2:
            st.markdown("**SEO**")
            st.markdown("\n".join(f"- {x}" for x in page.get("seo", [])))
        with c3:
            st.markdown("**Pro tips**")
            st.markdown("\n".join(f"- {x}" for x in page.get("tips", [])))
        with st.expander("Website source evidence"):
            st.code(page.get("sources", "[]"), language="json")
        download("website", page["html"], "index.html", "text/html")

    elif action == "App blueprint":
        out = emergent_app_blueprint(brief, corpus)
        st.markdown(out)
        download("app blueprint", out, "app_blueprint.md", "text/markdown")

    elif action == "Codex workflow":
        out = codex_workflow_brief(brief, corpus)
        st.markdown(out)
        download("codex workflow", out, "codex_workflow.md", "text/markdown")

    elif action == "Template":
        choice = st.selectbox("Template", [t["name"] for t in template_options()])
        out = render_template(choice, brief, corpus, "Evidence Studio")
        if out["mime"] == "text/html":
            components.html(out["content"], height=560, scrolling=True)
        else:
            st.code(out["content"][:12000])
        download("template", out["content"], out["filename"], out["mime"])

    elif action == "Voiceover":
        models = text_to_speech_options()
        choice = st.selectbox("TTS model", [m["label"] for m in models])
        selected_tts_engine = models[[m["label"] for m in models].index(choice)]["engine"]
        guide = tts_guidance(brief, selected_tts_engine, os.getenv("OCR_LANG", "eng"))
        st.json({k: v for k, v in guide.items() if k != "safe_text"})
        st.text_area("Safe script", guide["safe_text"], height=170)
        browser_speak_button(guide["safe_text"], safe_key("voiceover" + guide["safe_text"][:80]))
        if selected_tts_engine not in {"browser_speech", "manual_external"} and st.button("Generate voiceover audio"):
            audio = synthesize_speech(guide["safe_text"], engine=selected_tts_engine, voice=os.getenv("OPENAI_TTS_VOICE", "alloy"), language=os.getenv("OCR_LANG", ""))
            if audio.get("ok"):
                st.audio(audio["audio"], format=audio["mime"])
                download("voiceover audio", audio["audio"], f"voiceover.{audio['ext']}", audio["mime"])
            else:
                st.info(audio.get("note", "This TTS engine is a guided/external option."))
        if guide.get("url"):
            st.link_button("Open tool", guide["url"])
        download("voiceover script", guide["safe_text"], "voiceover_script.txt", "text/plain")

    elif action == "WhatsApp automation":
        service_url = st.text_input("Service / website URL", "")
        audience = st.text_input("Audience", "opted-in users")
        out = whatsapp_toolkit(brief, service_url, audience)
        st.json(out)
        with st.expander("Optional official WhatsApp Cloud API send"):
            to = st.text_input("Recipient phone in E.164", placeholder="919999999999")
            send_ok = st.checkbox("Human confirms opt-in, compliance, and message review")
            if to and send_ok and st.button("Send WhatsApp text"):
                st.json(whatsapp_send_text(to, out["safe_message"] + (f"\n{service_url}" if service_url else "")))
        download("WhatsApp automation", json.dumps(out, indent=2), "whatsapp_automation.json", "application/json")

    elif action == "Marketing":
        out = marketing_plan(brief, corpus, integration_registry())
        st.markdown(out)
        download("marketing plan", out, "marketing_plan.md", "text/markdown")

    elif action == "Media inventory":
        out = media_inventory(corpus)
        st.dataframe(out, use_container_width=True)
        download("media inventory", json.dumps(out, indent=2), "media_inventory.json", "application/json")

    elif action in {"Mindmap", "Visual maps"}:
        style = "NotebookLM mindmap" if action == "Mindmap" else st.selectbox("Visual type", ["NotebookLM mindmap", "Flowchart", "Concept map"])
        out = visual_map_pack(corpus, brief or "Evidence Visual Map", style, top_k)
        tabs = st.tabs(["Graphic", "Mermaid", "Evidence"])
        with tabs[0]:
            components.html(out["svg"], height=680, scrolling=True)
            download("visual svg", out["svg"], "visual_map.svg", "image/svg+xml")
        with tabs[1]:
            mermaid(out["mermaid"])
            st.code(out["mermaid"], language="mermaid")
            download("visual mermaid", out["mermaid"], "visual_map.mmd", "text/plain")
        with tabs[2]:
            st.dataframe(out["outline"], use_container_width=True)
            download("visual json", json.dumps(out, indent=2), "visual_map.json", "application/json")

    elif action == "Integrations":
        custom = st.text_area("Add integrations", placeholder="Tool, category, pricing, use, base_url, model, key_env, score")
        out = integration_registry(custom)
        if st.button("Save integrations"):
            st.success("Saved." if upsert_integrations_pg(out) else "PostgreSQL is not connected.")
        if load_integrations_pg():
            st.caption("Loaded registry rows from PostgreSQL.")
        st.dataframe(out, use_container_width=True)
        download("integrations", json.dumps(out, indent=2), "integrations.json", "application/json")

    elif action == "Swarm":
        if "swarm_state" not in st.session_state:
            st.session_state["swarm_state"] = swarm_initial_state()
        state = st.session_state["swarm_state"]
        topology = st.selectbox("Topology", state["available_topologies"])
        state["topology"] = topology
        mermaid(swarm_mermaid(state, topology), height=420)
        agent = st.selectbox("Agent", [a["name"] for a in state["agents"]])
        c1, c2 = st.columns(2)
        if c1.button("Positive feedback"):
            st.session_state["swarm_state"] = update_swarm_feedback(state, agent, "positive")
            st.rerun()
        if c2.button("Negative feedback"):
            st.session_state["swarm_state"] = update_swarm_feedback(state, agent, "negative")
            st.rerun()
        st.dataframe(st.session_state["swarm_state"]["agents"], use_container_width=True)
        download("swarm", json.dumps(st.session_state["swarm_state"], indent=2), "swarm_state.json", "application/json")

    elif action == "Toolbox":
        out = toolbox_catalog()
        st.dataframe(out, use_container_width=True)
        download("toolbox", json.dumps(out, indent=2), "toolbox.json", "application/json")

    elif action == "Compliance":
        out = compliance_report(corpus)
        st.json(out)
        download("compliance", json.dumps(out, indent=2), "compliance.json", "application/json")

    else:
        st.json(metadata)
        download("metadata", json.dumps(metadata, indent=2), "metadata.json", "application/json")
