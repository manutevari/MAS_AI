"""Minimal Streamlit UI for the full evidence-grounded toolkit."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
from html import escape
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components

from multi_agent import (
    advanced_strategy_pack,
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
    encrypt_secret_label,
    format_context,
    implementor_workflow,
    integration_registry,
    ingest_latest_updates,
    llm_model_catalog,
    load_integrations_pg,
    log_query_pg,
    marketing_plan,
    match_pydantic_schema_for_query,
    media_inventory,
    mermaid_mindmap,
    needs_live_search,
    orchestration_manager_plan,
    pinecone_retrieve,
    pinecone_upsert,
    render_template,
    researcher_workflow,
    retrieve,
    retrieve_auto,
    save_corpus_pg,
    school_clerk_automation,
    skill_manager_workflow,
    study_quiz_items,
    study_quiz_generator,
    summarize_corpus,
    supabase_log_metadata,
    swarm_initial_state,
    swarm_mermaid,
    storage_backends_status,
    template_options,
    text_to_speech_options,
    toolbox_catalog,
    transcribe_audio,
    tts_guidance,
    update_swarm_feedback,
    upsert_integrations_pg,
    vector_space_knowledge,
    visual_map_pack,
    whatsapp_send_text,
    whatsapp_toolkit,
)


st.set_page_config(page_title="Scientific RAG", layout="wide")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #212121 !important;
        color: #ececec !important;
        font-family: Inter, "Noto Sans Devanagari", system-ui, -apple-system, sans-serif;
    }
    .stMarkdown, .stTextArea textarea, .stChatMessage, p, li {
        line-height: 1.68;
        font-size: 0.96rem;
        color: #ececec;
    }
    .block-container {
        padding-top: 1.5rem;
        max-width: 820px;
    }
    h1, h2, h3, h4 {
        color: #f3f4f6 !important;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    [data-testid="stSidebar"] {
        background: #171717 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    div[data-testid="stMetric"] {
        background: #262626;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px;
        color: #ececec;
    }
    .hero {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        background: #262626;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .chip {
        display: inline-block;
        padding: 4px 10px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        margin: 4px 6px 4px 0;
        background: #2f2f2f;
        color: #ececec;
        font-size: 12px;
    }
    .danger {
        border-color: rgba(239, 68, 68, 0.4);
        background: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
    }
    .ok {
        border-color: rgba(16, 163, 127, 0.4);
        background: rgba(16, 163, 127, 0.15);
        color: #6ee7b7;
    }
    .muted {
        color: #8e8e8e;
    }
    textarea, input[type="text"] {
        border-radius: 14px !important;
        border-color: #383838 !important;
        background: #2f2f2f !important;
        color: #ececec !important;
    }
    [data-testid="stChatMessage"] {
        background: #262626;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 12px 16px;
        margin-bottom: 0.75rem;
    }
    [data-testid="stChatMessageContent"] {
        font-family: Inter, sans-serif;
    }
    .answer-meta {
        color: #8e8e8e;
        font-size: .82rem;
        margin: 4px 0 12px 0;
    }
    div.stButton > button {
        border-radius: 20px;
        min-height: 38px;
        font-weight: 500;
        padding: 6px 16px;
        background: #2f2f2f;
        color: #ececec;
        border: 1px solid rgba(255, 255, 255, 0.12);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: #383838;
        border-color: rgba(255, 255, 255, 0.25);
        color: #ffffff;
    }
    div.stDownloadButton > button {
        border-radius: 20px;
        background: #10a37f;
        color: white;
        border: none;
    }
    .compact-suggestions {
        margin-top: -8px;
        margin-bottom: 4px;
    }
    .compact-suggestions .stButton > button {
        min-height: 34px;
        width: 100%;
        border-color: #d8dee6;
        background: #ffffff;
        font-size: .86rem;
        line-height: 1.2;
        white-space: normal;
    }
    .compact-suggestions .stButton:first-of-type > button {
        border-color: #94a3b8;
        background: #f8fafc;
    }
</style>
""",
    unsafe_allow_html=True,
)


def secret_env() -> None:
    keys = (
        "OPENAI_API_KEY",
        "GROK_API_KEY",
        "GOOGLE_API_KEY",
        "HF_TOKEN",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
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
    try:
        for key in keys:
            value = st.secrets.get(key)
            if value and not os.getenv(key):
                os.environ[key] = str(value)
    except Exception:
        # Local/no-key mode should work even when secrets.toml is absent.
        return


def save_upload(file: Any) -> Path:
    root = Path(tempfile.gettempdir()) / "simple_rag_uploads"
    root.mkdir(exist_ok=True)
    path = root / file.name
    path.write_bytes(file.getbuffer())
    return path


def allow_download(label: str) -> bool:
    if os.getenv("REQUIRE_HUMAN_EXPORT_APPROVAL", "true").lower() != "true":
        return True
    return st.checkbox(f"Human approves {label}", key="approve_" + label)


def show_download(label: str, content: str | bytes, name: str, mime: str) -> None:
    if allow_download(label):
        data = content if isinstance(content, bytes) else content.encode()
        st.download_button("Download", data, name, mime)
    else:
        st.caption("Download locked until human approval.")


def render_mermaid(code: str, height: int = 560) -> None:
    html = f"""
<div class="mermaid">
{escape(code)}
</div>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
mermaid.initialize({{ startOnLoad: true, theme: "base" }});
</script>
"""
    components.html(html, height=height, scrolling=True)


def render_conversation(user_text: str, answer: str, meta: str = "") -> None:
    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
    with st.chat_message("assistant"):
        if meta:
            st.markdown(f'<div class="answer-meta">{escape(meta)}</div>', unsafe_allow_html=True)
        st.markdown(answer)


def render_sources(sources: List[Dict[str, Any]], label: str = "Sources") -> None:
    if not sources:
        return
    with st.expander(label, expanded=False):
        st.text(format_context(sources, max_chars=14000))


def render_summary_mindmap(pack: Dict[str, Any] | None) -> None:
    if not pack:
        return
    st.subheader("Summary mindmap")
    tabs = st.tabs(["Graphic", "Mermaid", "Evidence"])
    with tabs[0]:
        if pack.get("svg"):
            components.html(pack["svg"], height=620, scrolling=True)
        else:
            st.caption("No graphic mindmap is available for this summary.")
    with tabs[1]:
        if pack.get("mermaid"):
            render_mermaid(pack["mermaid"], height=560)
            st.code(pack["mermaid"], language="mermaid")
        else:
            st.caption("No Mermaid mindmap is available for this summary.")
    with tabs[2]:
        outline = pack.get("outline", [])
        if outline:
            st.dataframe(outline, use_container_width=True)
        else:
            st.caption("No evidence nodes are available for this summary.")


def apply_provider(choice: Dict[str, str]) -> str:
    provider = choice["provider"]
    if provider == "openrouter":
        os.environ["OPENROUTER_MODEL"] = choice["model"]
        os.environ["OPENROUTER_BASE_URL"] = choice["base_url"]
    if provider == "gemini":
        os.environ["GEMINI_MODEL"] = choice["model"]
    if provider == "huggingface":
        os.environ["HF_MODEL"] = choice["model"]
        os.environ["HF_BASE_URL"] = choice["base_url"]
    if provider == "custom":
        os.environ["CUSTOM_LLM_MODEL"] = choice["model"]
        os.environ["CUSTOM_LLM_BASE_URL"] = choice["base_url"]
        if choice.get("key_env"):
            os.environ["CUSTOM_LLM_API_KEY_ENV"] = choice["key_env"]
    if provider == "ollama":
        os.environ["OLLAMA_MODEL"] = choice["model"]
        os.environ["OLLAMA_BASE_URL"] = choice["base_url"] or "http://localhost:11434/v1"
    if provider == "openai":
        os.environ["OPENAI_MODEL"] = choice["model"]
    if provider == "grok":
        os.environ["GROK_MODEL"] = choice["model"]
    os.environ["LLM_PROVIDER"] = provider
    return provider


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_url_box(value: str) -> List[str]:
    tokens = re.split(r"[\s,;]+", value or "")
    urls: List[str] = []
    seen = set()
    for token in tokens:
        cleaned = token.strip().strip("()[]{}<>\"'")
        if not cleaned:
            continue
        if not re.match(r"^https?://", cleaned, re.I) and re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/.*)?$", cleaned):
            cleaned = "https://" + cleaned
        parsed = urlparse(cleaned)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        normalized = cleaned.rstrip(".,;")
        if normalized not in seen:
            urls.append(normalized)
            seen.add(normalized)
    return urls[:20]


def session_corpus_id(paths: List[Path], urls: List[str]) -> str:
    if paths:
        base = corpus_id(paths)
    else:
        base = "session"
    if not urls:
        return base
    digest = hashlib.sha256("|".join(urls).encode("utf-8")).hexdigest()[:12]
    return f"{base}-{digest}"


def apply_hidden_provider() -> str:
    rows = llm_model_catalog(os.getenv("MAS_EXTRA_LLMS", ""))
    preferred_provider = os.getenv("LLM_PROVIDER", "local").lower()
    provider_model_env = {
        "custom": "CUSTOM_LLM_MODEL",
        "gemini": "GEMINI_MODEL",
        "grok": "GROK_MODEL",
        "huggingface": "HF_MODEL",
        "ollama": "OLLAMA_MODEL",
        "openai": "OPENAI_MODEL",
        "openrouter": "OPENROUTER_MODEL",
    }
    preferred_model = os.getenv("LLM_MODEL") or os.getenv(provider_model_env.get(preferred_provider, ""), "")
    preferred_model = preferred_model.strip()
    fallback = next((row for row in rows if row.get("provider") == "local"), rows[0])
    matches = [
        row
        for row in rows
        if row.get("provider", "").lower() == preferred_provider
        and (not preferred_model or row.get("model") == preferred_model or row.get("label") == preferred_model)
    ]
    selected = matches[0] if matches else fallback
    key_env = selected.get("key_env", "")
    if selected.get("requires_key") == "yes" and key_env and not os.getenv(key_env):
        selected = fallback
    return apply_provider(selected)


def render_live_exam() -> None:
    exam_state = st.session_state.get("live_exam")
    if not exam_state:
        return

    items = exam_state.get("items", [])
    if not items:
        st.warning(exam_state.get("message", "No quiz items were generated."))
        return

    submitted = exam_state.setdefault("submitted", {})
    total_points = sum(int(item.get("points", 0)) for item in items)
    earned_points = sum(int(row.get("points", 0)) for row in submitted.values())
    answered = len(submitted)

    st.markdown("### Live Exam")
    st.caption("Options are vertical. Answers reveal only after you submit a selected answer.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Answered", f"{answered}/{len(items)}")
    c2.metric("Score", f"{earned_points}/{total_points}")
    c3.metric("Accuracy", f"{round((earned_points / total_points) * 100) if total_points else 0}%")
    c4.metric("Difficulty", exam_state.get("difficulty", "medium"))
    st.progress(answered / len(items))

    reset_col, finish_col = st.columns([1, 2])
    if reset_col.button("Reset live exam"):
        st.session_state.pop("live_exam", None)
        st.rerun()
    if finish_col.button("End exam and show score card"):
        exam_state["finished"] = True

    for idx, item in enumerate(items):
        qid = item.get("id", str(idx))
        saved = submitted.get(qid)
        with st.container(border=True):
            st.markdown(f"**Q{idx + 1}. {item['question']}**")
            st.caption(f"{item.get('points', 0)} point(s) | Source: {item.get('source')} p.{item.get('page')} [{item.get('section')}]")
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
                if saved.get("correct"):
                    st.success(f"Correct. +{item.get('points', 0)} point(s).")
                else:
                    st.error("Incorrect. +0 points.")
                selected_index = int(saved["selected_index"])
                remarks = item.get("option_feedback", [])
                if remarks and selected_index < len(remarks):
                    verdict = "correct" if saved.get("correct") else "incorrect"
                    st.info(f"Your answer is {verdict} because: {remarks[selected_index]}")
                st.markdown(f"**Correct answer:** {item['options'][int(item['correct_index'])]}")
                st.caption("Reason for correct answer: " + item.get("explanation", "The answer is grounded in the cited source."))
                if remarks:
                    st.markdown("**Remarks / definitions for all options**")
                    for option_i, option in enumerate(item["options"]):
                        prefix = "Correct option" if option_i == int(item["correct_index"]) else "Other option"
                        st.markdown(f"- **{prefix}:** {option} — {remarks[option_i]}")

    all_done = len(submitted) == len(items)
    if all_done or exam_state.get("finished"):
        rows = []
        for idx, item in enumerate(items, start=1):
            saved = submitted.get(item.get("id", ""))
            selected_index = int(saved["selected_index"]) if saved else -1
            remarks = item.get("option_feedback", [])
            rows.append(
                {
                    "Q": idx,
                    "Selected": item["options"][selected_index] if saved else "Not answered",
                    "Correct": item["options"][int(item["correct_index"])],
                    "Result": "Correct" if saved and saved.get("correct") else "Wrong / skipped",
                    "Points": saved.get("points", 0) if saved else 0,
                    "Remark": remarks[selected_index] if saved and remarks and selected_index < len(remarks) else "No answer selected.",
                    "Correct Reason": item.get("explanation", "Grounded in the cited source."),
                    "Source": f"{item.get('source')} p.{item.get('page')} [{item.get('section')}]",
                }
            )
        st.markdown("### Score Card")
        st.success(f"Final score: {earned_points}/{total_points} ({round((earned_points / total_points) * 100) if total_points else 0}%).")
        st.dataframe(rows, use_container_width=True)
        weak_topics = exam_state.get("weak_topics", {})
        if weak_topics:
            st.markdown("**Weak-topic revision map**")
            st.markdown("\n".join(f"- {topic}: {count} item(s)" for topic, count in sorted(weak_topics.items(), key=lambda x: x[1], reverse=True)))
        show_download("score card", json.dumps({"score": earned_points, "total": total_points, "rows": rows}, indent=2), "score_card.json", "application/json")


secret_env()

st.markdown(
    """
<div class="hero">
  <h1>Scientific RAG Studio</h1>
  <div class="muted">Evidence-grounded chat, live search, builders, templates, SWARN structure orchestration, and human approval in one focused interface.</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Setup")
    uploads = st.file_uploader(
        "Files",
        type=["zip", "pdf", "txt", "md", "csv", "tsv", "xlsx", "xls", "json", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    local_path = st.text_input("Local path")
    urls = st.text_area("URLs", placeholder="https://example.org/page")
    jurisdiction = os.getenv("COMPLIANCE_JURISDICTION", "India")
    os.environ["COMPLIANCE_JURISDICTION"] = jurisdiction
    fetch_ok = env_flag("MAS_URL_FETCH_OK", True)
    use_tavily = bool(os.getenv("TAVILY_API_KEY")) and env_flag("MAS_TAVILY_ENABLED", True)
    provider = apply_hidden_provider()

    retrieval = "Auto orchestrator"
    top_k = int(os.getenv("MAS_EVIDENCE_DEPTH", "8"))
    auto_mic_run = os.getenv("MAS_AUTO_MIC_RUN", "true").lower() == "true"
    os.environ.setdefault("CHUNKING_ENGINE", "section_semantic")
    os.environ.setdefault("OCR_ENGINE", "tesseract")
    os.environ.setdefault("OCR_LANG", "eng+hin+urd")
    os.environ.setdefault("TRANSLITERATION_ENGINE", "auto_llm")
    os.environ.setdefault("STT_ENGINE", "manual")

    with st.expander("Privacy and approval", expanded=False):
        lawful = st.checkbox("Lawful basis/consent for personal data")
        cloud = st.checkbox("Allow cloud processing")
        os.environ["DPDP_LAWFUL_BASIS"] = str(lawful).lower()
        os.environ["DPDP_CLOUD_CONSENT"] = str(lawful and cloud).lower()
        os.environ["DPDP_REDACT"] = str(st.checkbox("Redact personal identifiers", value=True)).lower()
        os.environ["HUMAN_REVIEW_CONFIRMED"] = str(st.checkbox("Human reviewer responsible")).lower()
        os.environ["REQUIRE_HUMAN_EXPORT_APPROVAL"] = str(st.checkbox("Require approval before export", value=True)).lower()

paths = [save_upload(f) for f in uploads] if uploads else ([Path(local_path)] if local_path else [])
web_urls = parse_url_box(urls) if fetch_ok else []
with st.spinner("Indexing evidence..."):
    corpus, summary = build_corpus_from_paths(paths) if paths else ([], "No files.")
    if web_urls:
        web_corpus, web_summary = build_corpus_from_urls(web_urls, jurisdiction)
        corpus.extend(web_corpus)
        summary += " " + web_summary
    cid = session_corpus_id(paths, web_urls)
    if corpus:
        save_corpus_pg(corpus, cid)

metadata = corpus_metadata(corpus, cid)
supabase_log_metadata(metadata)
if corpus and os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_INDEX") and os.getenv("OPENAI_API_KEY"):
    pinecone_upsert(corpus, cid)
st.caption(summary)
with st.expander("Session status", expanded=False):
    status_cols = st.columns(2)
    status_cols[0].metric("Chunks", len(corpus))
    status_cols[1].metric("Sources", metadata.get("source_count", 0))
st.markdown(
    "".join(
        [
            f'<span class="chip {"ok" if os.getenv("HUMAN_REVIEW_CONFIRMED") == "true" else "danger"}">Human review {"on" if os.getenv("HUMAN_REVIEW_CONFIRMED") == "true" else "pending"}</span>',
            f'<span class="chip {"ok" if os.getenv("DPDP_REDACT") == "true" else "danger"}">Redaction {os.getenv("DPDP_REDACT")}</span>',
        ]
    ),
    unsafe_allow_html=True,
)

if "brief_text" not in st.session_state:
    st.session_state["brief_text"] = ""

WORKFLOWS = [
    "Chat",
    "Summarizer",
    "Agent chat",
    "Skill manager",
    "Researcher",
    "Implementor",
    "Ask suggestions",
    "Vector knowledge",
    "Naya search",
    "Live search",
    "Ingest latest updates",
    "AI policy scan",
    "School clerk",
    "Study quiz",
    "Advanced strategies",
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
    "SWARN architecture",
    "Toolbox",
    "Compliance",
    "Metadata",
]
action = "Smart auto"
suggested_action = st.session_state.get("suggested_action", "")
if suggested_action in WORKFLOWS:
    action = suggested_action

admin_unlocked = os.getenv("MAS_ADMIN_MODE", "false").lower() == "true"
with st.expander("Admin routing override", expanded=False):
    admin_pin_expected = os.getenv("MAS_ADMIN_PIN") or os.getenv("ADMIN_PIN")
    if admin_pin_expected:
        admin_pin = st.text_input("Admin PIN", type="password")
        admin_unlocked = admin_unlocked or admin_pin == admin_pin_expected
    if admin_unlocked:
        if st.checkbox("Admin manually chooses workflow"):
            action = st.selectbox("Workflow", WORKFLOWS)
            st.session_state["suggested_action"] = action
        elif suggested_action:
            st.caption(f"Queued suggested workflow: {suggested_action}. Clear it to return to smart auto.")
            if st.button("Clear suggested workflow"):
                st.session_state.pop("suggested_action", None)
                st.rerun()
    else:
        st.caption("Smart routing is active. Manual workflow selection is available only after admin unlock. Use suggestions below for human-in-loop steering.")

if st.session_state.get("pending_brief_text"):
    st.session_state["brief_text"] = st.session_state.pop("pending_brief_text")
st.markdown("### Ask")
brief = st.text_area("Brief / query", height=120, placeholder="Ask or describe what you want.", key="brief_text")
if web_urls and not brief.strip():
    brief = "Summarizer: summarize the linked URL evidence and extract implementation-ready actions with citations."

with st.container():
    st.markdown('<div class="compact-suggestions">', unsafe_allow_html=True)
    suggestions = ask_suggestions(corpus, 4)
    summarizer_prompt = "Summarizer: summarize the uploaded evidence with citations."
    suggestions = [summarizer_prompt] + [q for q in suggestions if q != summarizer_prompt]
    suggestions = suggestions[:4]
    cols = st.columns(4)
    for i, q in enumerate(suggestions):
        label = "Summarizer" if i == 0 else (q[:42] + "..." if len(q) > 45 else q)
        if cols[i].button(label, key=f"suggest_{i}", help=q, use_container_width=True):
            st.session_state["pending_brief_text"] = q
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

suggested_workflows = [
    ("Agent chat", "Let the agent council plan, retrieve, answer, verify, and show its human-visible trace."),
    ("Skill manager", "Inspect, discover, register, and manage tools, agent capabilities, and skill selection rules."),
    ("Researcher", "Conduct evidence-grounded scientific research, hypothesis checking, literature synthesis, and evidence evaluation."),
    ("Implementor", "Translate research, blueprints, and plans into concrete code, scripts, deliverables, and packages."),
    ("SWARN architecture", "View supervisor-led workflow, agent, retrieval, reasoning, and next-action orchestration."),
    ("Advanced strategies", "Generate retrieval-ready chunking, guardrail, evaluation, and failure-mode artifacts."),
    ("Summarizer", "Create a grounded summary with citations and a mindmap."),
    ("Vector knowledge", "Inspect the evidence space while the backend is chosen automatically."),
    ("Ask suggestions", "Generate useful questions from the indexed evidence."),
    ("Naya search", "Search latest/new evidence when Tavily is configured, otherwise show what is needed."),
    ("Compliance", "Review privacy, consent, redaction, and jurisdiction guardrails."),
    ("Ingest latest updates", "Collect latest snippets and persist to configured PostgreSQL/Pinecone stores."),
    ("Toolbox", "Check tools, packages, databases, LLMs, and integration readiness."),
    ("Metadata", "Inspect corpus, source, provider, and storage metadata."),
]
workflow_help = dict(suggested_workflows)
workflow_labels = ["Smart auto"] + [workflow for workflow, _ in suggested_workflows]
quick_index = workflow_labels.index(suggested_action) if suggested_action in workflow_labels else 0
with st.expander("Quick workflow", expanded=True):
    q1, q2, q3 = st.columns([5, 1.4, 1.2])
    quick_choice = q1.selectbox(
        "Workflow",
        workflow_labels,
        index=quick_index,
        help="Use Smart auto for normal routing, or choose a specific suggested workflow.",
    )
    if q2.button("Apply", use_container_width=True):
        if quick_choice == "Smart auto":
            st.session_state.pop("suggested_action", None)
        else:
            st.session_state["suggested_action"] = quick_choice
            if not st.session_state.get("brief_text"):
                st.session_state["pending_brief_text"] = workflow_help.get(quick_choice, "")
        st.rerun()
    if suggested_action and q3.button("Clear", use_container_width=True):
        st.session_state.pop("suggested_action", None)
        st.rerun()
    if suggested_action:
        st.caption(f"Queued: {suggested_action}. Press Run, or choose Smart auto and Apply.")
    else:
        st.caption("Smart auto is active. Choose a workflow only when you want to steer the orchestrator.")
    with st.expander("Workflow guide", expanded=False):
        st.dataframe(
            [{"workflow": workflow, "use": help_text} for workflow, help_text in suggested_workflows],
            use_container_width=True,
            hide_index=True,
        )

with st.expander("Voice and extra files", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        mic_audio = st.audio_input("Mic") if hasattr(st, "audio_input") else None
        audio = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg", "webm"])
    with c2:
        extra_files = st.file_uploader(
            "Upload more files",
            type=["zip", "pdf", "txt", "md", "csv", "tsv", "xlsx", "xls", "json", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="inline_more_files",
        )
    if extra_files:
        more_paths = [save_upload(f) for f in extra_files]
        more_corpus, more_summary = build_corpus_from_paths(more_paths)
        corpus.extend(more_corpus)
        summary += " " + more_summary
        metadata = corpus_metadata(corpus, cid)
        st.caption(more_summary)
    speech = mic_audio or audio
    if speech and st.button("Transcribe"):
        name = getattr(speech, "name", "mic_input.wav")
        transcript = transcribe_audio(speech.getvalue(), name, os.getenv("STT_ENGINE", "manual"), os.getenv("OCR_LANG", "eng").split("+")[0])
        st.session_state["pending_brief_text"] = transcript
        if auto_mic_run:
            st.session_state["pending_auto_run"] = True
        st.rerun()

if use_tavily and brief and (action in {"Live search", "Naya search", "Ingest latest updates"} or needs_live_search(brief)):
    with st.spinner("Adding Tavily live evidence..."):
        live_corpus, live_summary = build_corpus_from_tavily(brief, max_results=5)
        corpus.extend(live_corpus)
        summary += " " + live_summary
        metadata = corpus_metadata(corpus, cid)
        st.caption(live_summary)

with st.expander("Evidence preview", expanded=False):
    hits, preview_retrieval = retrieve_auto(corpus, brief or "summary", top_k, cid, requested=retrieval, provider=provider)
    st.text(format_context(hits) if hits else "No indexed evidence yet. Paste URLs or upload documents for grounded evidence.")

quiz_active = action == "Study quiz" and bool(st.session_state.get("live_exam"))
auto_run = bool(st.session_state.pop("pending_auto_run", False))
run = st.button("Run", type="primary") or auto_run
if auto_run:
    st.success("Mic transcript routed to the best available workflow.")
if not run and not quiz_active:
    st.stop()

manager_plan: Dict[str, Any] | None = orchestration_manager_plan(
    brief,
    corpus,
    provider=provider,
    retrieval_engine=retrieval,
    live_search_enabled=use_tavily,
    jurisdiction=jurisdiction,
)
if action == "Smart auto":
    action = manager_plan["selected_action"]
else:
    manager_plan["selected_action"] = action
    manager_plan["rationale"] = f"Human selected the suggested workflow `{action}`; orchestrator still chooses tools, backend, and guardrails."
    manager_plan["confidence"] = max(float(manager_plan.get("confidence", 0.75)), 0.75)

with st.expander("Human-visible routing audit", expanded=False):
    st.metric("Selected workflow", manager_plan["selected_action"])
    st.metric("Confidence", f"{int(manager_plan['confidence'] * 100)}%")
    st.caption(manager_plan["rationale"])
    st.caption(f"Routing mode: {manager_plan.get('routing_mode', 'rule-based')}")
    st.dataframe(manager_plan["agents"], use_container_width=True)
    visible_tools = manager_plan["tools"] if admin_unlocked else [row for row in manager_plan["tools"] if row.get("tool") != "retrieval"]
    st.dataframe(visible_tools, use_container_width=True)
    if admin_unlocked:
        with st.expander("Admin technical routing details", expanded=False):
            st.caption(f"Retrieval: {manager_plan['retrieval_decision']['engine']} - {manager_plan['retrieval_decision']['reason']}")
            st.dataframe(manager_plan.get("storage_backends", []), use_container_width=True)
            st.dataframe(manager_plan.get("integration_hints", []), use_container_width=True)
            st.json(manager_plan["evidence_state"])

schema_match = match_pydantic_schema_for_query(brief, corpus)
with st.expander("Pydantic Schema & Grounded Validation Match", expanded=False):
    s1, s2, s3 = st.columns(3)
    s1.metric("Pydantic Schema", schema_match["matched_schema_name"])
    s2.metric("Match Confidence", f"{int(schema_match['confidence'] * 100)}%")
    s3.metric("Grounded Chunks", schema_match["grounded_evidence_count"])
    st.caption("Pydantic v2 schema auto-matched to user query intent and validated against grounded results.")
    st.markdown("#### JSON Schema Definition (OpenAPI)")
    st.json(schema_match["json_schema"])
    st.markdown("#### Pydantic Validated Model Payload")
    st.json(schema_match["validated_data"])

if action == "Chat":
    if not corpus and not use_tavily:
        result = {"answer": "Live chat is available, but no evidence is indexed. Upload documents or configure latest search for grounded answers.", "sources": [], "provider": "local", "model": "no-evidence"}
    else:
        result = asyncio.run(
            answer_rag_chat(
                brief,
                corpus,
                provider=provider,
                top_k=top_k,
                retrieval_engine=retrieval,
            )
        )
    render_conversation(brief, result["answer"], f"{result.get('provider', provider)} · {result.get('model', '')}")
    render_sources(result.get("sources", []), "Retrieved evidence")
    log_query_pg(cid, brief, result["answer"], result.get("provider", ""), result.get("model", ""))
    show_download("answer", json.dumps(result, indent=2), "answer.json", "application/json")

elif action == "Summarizer":
    if not corpus and not use_tavily:
        result = {"answer": "No indexed evidence is available yet. Upload a document, paste a permitted URL, or configure latest search before summarizing.", "sources": [], "provider": "local", "model": "no-evidence"}
    else:
        result = asyncio.run(summarize_corpus(corpus, brief or "Summarize the uploaded evidence with citations.", provider=provider, top_k=top_k))
    render_conversation(brief or "Summarize the uploaded evidence", result["answer"], f"{result.get('provider', provider)} · {result.get('model', '')}")
    render_summary_mindmap(result.get("mindmap"))
    render_sources(result.get("sources", []), "Summary evidence")
    log_query_pg(cid, brief or "Summarize", result["answer"], result.get("provider", ""), result.get("model", ""))
    show_download("summary", json.dumps(result, indent=2), "summary.json", "application/json")

elif action == "Agent chat":
    result = asyncio.run(answer_with_agent_pipeline_from_corpus(brief, corpus, summary, provider))
    render_conversation(brief, result["answer"], f"{result.get('provider', provider)} · SWARN/hybrid")
    render_sources(result.get("sources", []), "Retrieved evidence")
    with st.expander("Human-visible agent council", expanded=True):
        st.caption("This is the SWARN agent conversation for human supervision. It is not a normal-user workflow selector.")
        st.json(result.get("conversation", []))
    with st.expander("SWARN structure orchestration", expanded=False):
        render_mermaid(result.get("swarn_mermaid", result.get("swarm_mermaid", swarm_mermaid(swarm_initial_state()))))
        st.json(result.get("swarn_state", result.get("swarm_state", {})))
    show_download("agent answer", json.dumps(result, indent=2), "agent_answer.json", "application/json")

elif action == "Skill manager":
    out = skill_manager_workflow(brief, corpus)
    st.markdown(out["markdown_report"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Registered Skills", len(out["registered_skills"]))
    c2.metric("Active Triggered", len(out["active_skills"]))
    c3.metric("Missing Packages / Keys", len(out["missing_dependencies"]))
    with st.expander("Registered skill catalog", expanded=True):
        st.dataframe(out["registered_skills"], use_container_width=True)
    with st.expander("System capability matrix", expanded=False):
        st.dataframe(out["capability_matrix"], use_container_width=True)
    if out.get("missing_dependencies"):
        with st.expander("Missing dependencies warning"):
            st.markdown("\n".join(f"- {d}" for d in out["missing_dependencies"]))
    show_download("skill manager report", json.dumps(out, indent=2), "skill_manager_report.json", "application/json")

elif action == "Researcher":
    out = researcher_workflow(brief, corpus, live_search_enabled=use_tavily, provider=provider)
    st.markdown(out["markdown_synthesis"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Evidence Depth", len(out["sources"]))
    c2.metric("Numeric Findings", len(out["numeric_findings"]))
    c3.metric("Hypotheses Checked", len(out["hypothesis_eval"]))
    with st.expander("Hypothesis evaluation matrix", expanded=True):
        st.dataframe(out["hypothesis_eval"], use_container_width=True)
    with st.expander("Evidence matrix", expanded=False):
        st.dataframe(out["evidence_matrix"], use_container_width=True)
    render_sources(out.get("sources", []), "Research source evidence")
    show_download("researcher synthesis", json.dumps(out, indent=2), "researcher_synthesis.json", "application/json")

elif action == "Implementor":
    out = implementor_workflow(brief, corpus)
    st.markdown(out["implementation_plan"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Target Deliverable", out["filename"])
    c2.metric("Language", out["code_lang"].upper())
    c3.metric("Syntax Verification", out["verification"]["syntax_check"])
    st.markdown(f"### Code Deliverable: `{out['filename']}`")
    st.code(out["code_content"], language=out["code_lang"])
    with st.expander("Verification & grounding details", expanded=True):
        st.json(out["verification"])
    show_download("implementor deliverable", out["code_content"], out["filename"], out["mime"])
    show_download("implementor packet", json.dumps(out, indent=2), "implementor_packet.json", "application/json")

elif action == "Ask suggestions":
    out = ask_suggestions(corpus)
    st.markdown("\n".join(f"- {q}" for q in out))
    show_download("ask suggestions", json.dumps(out, indent=2), "ask_suggestions.json", "application/json")

elif action == "Vector knowledge":
    out = vector_space_knowledge(corpus, brief or "entire corpus", k=25)
    render_conversation(brief, "I reviewed the indexed evidence space. Open the panels below for source coverage, top evidence, and suggested questions.", "Vector knowledge")
    if admin_unlocked:
        with st.expander("Admin backend routing details", expanded=False):
            st.json(out.get("retrieval_decision", {}))
            st.dataframe(out.get("storage_backends", []), use_container_width=True)
    with st.expander("Coverage summary", expanded=True):
        st.json(out["summary"])
    render_sources(out["top_evidence"], "Top evidence")
    with st.expander("Suggested questions", expanded=False):
        st.markdown("\n".join(f"- {q}" for q in out["suggested_questions"]))
    show_download("vector knowledge", json.dumps(out, indent=2), "vector_knowledge.json", "application/json")

elif action in {"Live search", "Naya search"}:
    if not os.getenv("TAVILY_API_KEY"):
        st.warning("Naya/latest search is present but needs TAVILY_API_KEY in Streamlit secrets or environment to fetch fresh web snippets.")
    elif not use_tavily:
        st.info("Naya/latest search is available, but it is disabled by deployment configuration.")
    out = vector_space_knowledge(corpus, brief or "live search", k=25)
    render_conversation(brief, "Naya/latest or permitted evidence has been routed through the knowledge layer. Open the panels below to inspect what was retrieved.", action)
    if admin_unlocked:
        with st.expander("Admin Naya backend details", expanded=False):
            st.json(out.get("retrieval_decision", {}))
            st.dataframe(out.get("storage_backends", []), use_container_width=True)
    with st.expander("Coverage summary", expanded=True):
        st.json(out["summary"])
    render_sources(out["top_evidence"], "Live evidence")
    show_download("naya search evidence", json.dumps(out, indent=2), "naya_search_evidence.json", "application/json")

elif action == "Ingest latest updates":
    st.warning("This uses configured latest-search connectors. It stores snippets, not unrestricted scraped pages.")
    st.caption("The orchestrator stores updates only in configured, available stores.")
    namespace = st.text_input("Update corpus / namespace", "latest_updates")
    max_results = st.slider("Live results", 3, 10, 8)
    save_pg = bool(os.getenv("DATABASE_URL"))
    save_pc = bool(os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_INDEX") and os.getenv("OPENAI_API_KEY"))
    out = ingest_latest_updates(
        brief or "latest updates",
        corpus_id_value=namespace,
        max_results=max_results,
        jurisdiction=jurisdiction,
        urls=web_urls,
        store_postgres=save_pg,
        store_pinecone=save_pc,
    )
    st.json(out)
    show_download("latest updates", json.dumps(out, indent=2), "latest_updates_ingest.json", "application/json")

elif action == "AI policy scan":
    profiles = ["All"] + [p["name"] for p in ai_policy_profiles()]
    profile = st.selectbox("Policy profile", profiles)
    out = ai_policy_scan(profile, jurisdiction)
    st.json(out)
    show_download("AI policy scan", json.dumps(out, indent=2), "ai_policy_scan.json", "application/json")

elif action == "School clerk":
    out = school_clerk_automation(brief, corpus)
    st.markdown(out["markdown"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Task", out["task"].replace("_", " ").title())
    c2.metric("Table records", out["records_detected"])
    c3.metric("Human review", "Required")
    if out.get("rows"):
        st.markdown("### Result Preview")
        st.dataframe(out["rows"], use_container_width=True)
    with st.expander("Pro tips", expanded=True):
        st.markdown("\n".join(f"- {tip}" for tip in out.get("pro_tips", [])))
    with st.expander("Human approval checklist", expanded=True):
        st.markdown("\n".join(f"- {item}" for item in out.get("human_checklist", [])))
    with st.expander("Available clerk automations"):
        st.markdown("\n".join(f"- {item}" for item in out.get("automation_catalog", [])))
    if out.get("csv"):
        show_download("school result csv", out["csv"], "school_result_sheet.csv", "text/csv")
    show_download("school clerk packet", json.dumps(out, indent=2), "school_clerk_automation.json", "application/json")

elif action == "Study quiz":
    c1, c2, c3 = st.columns(3)
    exam = c1.text_input("Exam", "School / University Exam")
    difficulty = c2.selectbox("Difficulty", ["easy", "medium", "hard"])
    mode = c3.selectbox("Mode", ["question_paper", "pw_practice", "textbook_solution", "assertion_reason", "quiz", "flashcards"])
    count = st.slider("Questions", 5, 50, 10)
    live_mode = st.toggle("Live exam with scoring", value=mode in {"quiz", "pw_practice", "assertion_reason"})
    if run and live_mode:
        quiz = study_quiz_items(corpus, exam, brief or "uploaded syllabus", count, difficulty, mode)
        quiz["submitted"] = {}
        quiz["finished"] = False
        st.session_state["live_exam"] = quiz
    elif run:
        st.session_state.pop("live_exam", None)
        out = study_quiz_generator(corpus, exam, brief or "uploaded syllabus", count, difficulty, mode)
        st.markdown(out)
        show_download("study quiz", out, f"{mode}.md", "text/markdown")

    if live_mode and st.session_state.get("live_exam"):
        render_live_exam()

elif action == "Advanced strategies":
    out = advanced_strategy_pack(brief, corpus)
    st.markdown(out["report"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Strategy areas", len(out["strategies"]))
    c2.metric("Eval rows", len(out["evaluation_rows"]))
    c3.metric("Source chunks", len(out["sources"]))
    tabs = st.tabs(["Readiness", "Tokenizer", "Retrieval", "Chunking", "Evaluation", "Guardrails", "Failure modes", "Mindmap"])
    with tabs[0]:
        st.dataframe(out["readiness"], use_container_width=True)
        st.json(out["content_type_estimate"])
    with tabs[1]:
        st.dataframe(out["tokenizer_diagnostics"], use_container_width=True)
    with tabs[2]:
        st.dataframe(out["retrieval_experiment"], use_container_width=True)
    with tabs[3]:
        st.dataframe(out["chunking_experiment"], use_container_width=True)
    with tabs[4]:
        st.json(out["evaluation_run"]["summary"])
        st.dataframe(out["evaluation_run"]["rows"], use_container_width=True)
        st.dataframe(out["evaluation_rows"], use_container_width=True)
        show_download("evaluation csv", out["evaluation_csv"], "evaluation_results_starter.csv", "text/csv")
        show_download("evaluation run csv", out["evaluation_run"]["csv"], "evaluation_results_autoscored.csv", "text/csv")
    with tabs[5]:
        st.code(out["guardrail_prompt"], language="text")
        show_download("guardrail prompt", out["guardrail_prompt"], "grounding_prompt.txt", "text/plain")
    with tabs[6]:
        st.markdown(out["failure_modes_md"])
        show_download("failure modes", out["failure_modes_md"], "failure_modes.md", "text/markdown")
    with tabs[7]:
        components.html(out["mindmap"]["svg"], height=640, scrolling=True)
        with st.expander("Mermaid"):
            render_mermaid(out["mindmap"]["mermaid"], height=520)
            st.code(out["mindmap"]["mermaid"], language="mermaid")
    render_sources(out.get("sources", []), "Strategy source evidence")
    show_download("advanced strategy packet", json.dumps(out, indent=2), "advanced_strategy_pack.json", "application/json")

elif action == "Website":
    page = build_website(brief, corpus, "Evidence Studio", "Evidence-grounded publication")
    components.html(page["html"], height=600, scrolling=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Critic Review**")
        st.markdown("\n".join(f"- {x}" for x in page.get("critique", [])))
        st.markdown("**SEO Suggestions**")
        st.markdown("\n".join(f"- {x}" for x in page.get("seo", [])))
    with c2:
        st.markdown("**Pro Tips**")
        st.markdown("\n".join(f"- {x}" for x in page.get("tips", [])))
        with st.expander("Website source evidence"):
            st.code(page.get("sources", "[]"), language="json")
    show_download("website", page["html"], "index.html", "text/html")

elif action == "App blueprint":
    out = emergent_app_blueprint(brief, corpus)
    st.markdown(out)
    show_download("app blueprint", out, "app_blueprint.md", "text/markdown")

elif action == "Codex workflow":
    out = codex_workflow_brief(brief, corpus)
    st.markdown(out)
    show_download("codex workflow", out, "codex_workflow.md", "text/markdown")

elif action == "Template":
    templates = template_options()
    choice = st.selectbox("Template type", [t["name"] for t in templates])
    out = render_template(choice, brief, corpus, "Evidence Studio")
    if out["mime"] == "text/html":
        components.html(out["content"], height=560, scrolling=True)
    else:
        st.code(out["content"][:12000])
    show_download("template", out["content"], out["filename"], out["mime"])

elif action == "Voiceover":
    models = text_to_speech_options()
    choice = st.selectbox("TTS model", [m["label"] for m in models])
    guide = tts_guidance(brief, models[[m["label"] for m in models].index(choice)]["engine"], os.getenv("OCR_LANG", "eng"))
    st.json({k: v for k, v in guide.items() if k != "safe_text"})
    st.text_area("Safe script", guide["safe_text"], height=180)
    if guide.get("url"):
        st.link_button("Open tool", guide["url"])
    show_download("voiceover script", guide["safe_text"], "voiceover_script.txt", "text/plain")

elif action == "WhatsApp automation":
    service_url = st.text_input("Service / website URL", "")
    audience = st.text_input("Audience", "opted-in users")
    out = whatsapp_toolkit(brief, service_url, audience)
    st.json(out)
    with st.expander("Optional Cloud API send"):
        st.warning("Send only to opted-in recipients and only when policy/consent requirements are satisfied.")
        to = st.text_input("Recipient phone E.164", placeholder="919999999999")
        send_ok = st.checkbox("Human confirms opt-in, policy compliance, and message review")
        if to and send_ok and st.button("Send WhatsApp text"):
            st.json(whatsapp_send_text(to, out["safe_message"] + (f"\n{service_url}" if service_url else "")))
    show_download("WhatsApp automation", json.dumps(out, indent=2), "whatsapp_automation.json", "application/json")

elif action == "Marketing":
    out = marketing_plan(brief, corpus, integration_registry())
    st.markdown(out)
    show_download("marketing plan", out, "marketing_plan.md", "text/markdown")

elif action == "Media inventory":
    out = media_inventory(corpus)
    st.dataframe(out, use_container_width=True)
    show_download("media inventory", json.dumps(out, indent=2), "media_inventory.json", "application/json")

elif action == "Mindmap":
    out = visual_map_pack(corpus, brief or "Evidence Mindmap", "NotebookLM mindmap", top_k)
    st.caption(out["note"])
    tabs = st.tabs(["Graphic", "Mermaid", "Evidence"])
    with tabs[0]:
        components.html(out["svg"], height=680, scrolling=True)
        show_download("mindmap svg", out["svg"], "mindmap.svg", "image/svg+xml")
    with tabs[1]:
        render_mermaid(out["mermaid"])
        st.code(out["mermaid"], language="mermaid")
        show_download("mindmap mermaid", out["mermaid"], "mindmap.mmd", "text/plain")
    with tabs[2]:
        st.dataframe(out["outline"], use_container_width=True)
        show_download("mindmap evidence", json.dumps(out, indent=2), "mindmap.json", "application/json")

elif action == "Visual maps":
    style = st.selectbox("Visual type", ["NotebookLM mindmap", "Flowchart", "Concept map"])
    depth = st.slider("Visual evidence depth", 5, 20, top_k)
    out = visual_map_pack(corpus, brief or "Evidence Visual Map", style, depth)
    st.caption(out["note"])
    tabs = st.tabs(["Graphic image", "Mermaid visual", "Evidence outline"])
    with tabs[0]:
        components.html(out["svg"], height=720, scrolling=True)
        show_download("visual svg", out["svg"], f"{style.lower().replace(' ', '_')}.svg", "image/svg+xml")
    with tabs[1]:
        render_mermaid(out["mermaid"], height=620)
        st.code(out["mermaid"], language="mermaid")
        show_download("visual mermaid", out["mermaid"], f"{style.lower().replace(' ', '_')}.mmd", "text/plain")
    with tabs[2]:
        st.dataframe(out["outline"], use_container_width=True)
        show_download("visual map json", json.dumps(out, indent=2), "visual_map.json", "application/json")

elif action == "Integrations":
    custom = st.text_area("Add integrations", placeholder="Tool, category, pricing, use, base_url, model, key_env, score")
    out = integration_registry(custom)
    if st.button("Save integrations"):
        st.success("Saved." if upsert_integrations_pg(out) else "PostgreSQL not connected.")
    if load_integrations_pg():
        st.caption("Loaded registry rows from PostgreSQL.")
    st.dataframe(out, use_container_width=True)
    show_download("integrations", json.dumps(out, indent=2), "integrations.json", "application/json")

elif action in {"SWARN architecture", "Swarm"}:
    if "swarn_state" not in st.session_state:
        st.session_state["swarn_state"] = st.session_state.get("swarm_state", swarm_initial_state())
    state = st.session_state["swarn_state"]
    st.markdown("### SWARN Structure Orchestration")
    st.caption("Supervisor -> Workflow -> Agent Network -> Retrieval/Reasoning -> Next Action. Human approval remains above every node.")
    topology = st.selectbox("SWARN topology", state["available_topologies"])
    state["topology"] = topology
    st.markdown(f"```mermaid\n{swarm_mermaid(state, topology)}\n```")
    with st.expander("SWARN architecture layers", expanded=True):
        st.dataframe(state.get("architecture", []), use_container_width=True)
    agent = st.selectbox("Agent", [a["name"] for a in state["agents"]])
    c1, c2 = st.columns(2)
    if c1.button("Positive feedback"):
        st.session_state["swarn_state"] = update_swarm_feedback(state, agent, "positive")
        st.rerun()
    if c2.button("Negative feedback"):
        st.session_state["swarn_state"] = update_swarm_feedback(state, agent, "negative")
        st.rerun()
    st.dataframe(st.session_state["swarn_state"]["agents"], use_container_width=True)
    show_download("SWARN architecture", json.dumps(st.session_state["swarn_state"], indent=2), "swarn_architecture.json", "application/json")

elif action == "Toolbox":
    out = toolbox_catalog()
    st.dataframe(out, use_container_width=True)
    with st.expander("Integration registry", expanded=False):
        st.dataframe(integration_registry(), use_container_width=True)
    if admin_unlocked:
        with st.expander("Admin backend status", expanded=False):
            st.dataframe(storage_backends_status(), use_container_width=True)
    show_download("toolbox", json.dumps(out, indent=2), "toolbox_catalog.json", "application/json")

elif action == "Compliance":
    out = compliance_report(corpus)
    st.json(out)
    show_download("compliance", json.dumps(out, indent=2), "compliance_report.json", "application/json")

else:
    metadata_packet = {
        **metadata,
        "orchestrator": manager_plan,
        "storage_backends": storage_backends_status(),
        "integrations": integration_registry(include_pg=False)[:20],
    }
    st.json(metadata_packet)
    show_download("metadata", json.dumps(metadata_packet, indent=2), "metadata.json", "application/json")