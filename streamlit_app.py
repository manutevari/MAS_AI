"""Minimal Streamlit UI for the full evidence-grounded toolkit."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components

from multi_agent import (
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
    integration_registry,
    llm_model_catalog,
    load_integrations_pg,
    log_query_pg,
    marketing_plan,
    media_inventory,
    needs_live_search,
    ocr_model_options,
    pinecone_retrieve,
    pinecone_upsert,
    render_template,
    retrieve,
    save_corpus_pg,
    speech_to_text_options,
    study_quiz_generator,
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
)


st.set_page_config(page_title="Scientific RAG", layout="wide")

st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.4rem;
        max-width: 1280px;
    }
    h1 {
        font-size: 2.2rem;
        margin-bottom: .2rem;
    }
    [data-testid="stSidebar"] {
        background: #f7f9fb;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
    }
    .hero {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 18px 20px;
        background: linear-gradient(135deg, #ffffff, #f5fbff);
    }
    .chip {
        display: inline-block;
        padding: 4px 9px;
        border: 1px solid #d8dee6;
        border-radius: 999px;
        margin: 4px 6px 4px 0;
        background: #ffffff;
        font-size: 12px;
    }
    .danger {
        border-color: #f1b4b4;
        background: #fff5f5;
    }
    .ok {
        border-color: #a7e0bd;
        background: #f1fff6;
    }
    .muted {
        color: #64748b;
    }
    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 8px;
        min-height: 38px;
    }
    div.stDownloadButton > button {
        background: #0f766e;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


def secret_env() -> None:
    for key in (
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
    ):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])


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


secret_env()

st.markdown(
    """
<div class="hero">
  <h1>Scientific RAG Studio</h1>
  <div class="muted">Evidence-grounded chat, live search, builders, templates, swarm governance, and human approval in one focused interface.</div>
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
    jurisdiction = st.selectbox("Jurisdiction", ["India", "EU/EEA", "California", "UK", "Global/Unknown"])
    os.environ["COMPLIANCE_JURISDICTION"] = jurisdiction
    fetch_ok = st.checkbox("I confirm URL fetching is lawful and robots.txt/site terms permit it")
    use_tavily = st.checkbox("Use Tavily live search when needed")
    st.caption("Tavily: configured" if os.getenv("TAVILY_API_KEY") else "Tavily: add TAVILY_API_KEY in Streamlit secrets")

    st.divider()
    extra_models = st.text_area("Extra LLMs", placeholder="Label, provider, model, base_url, key_env")
    llm_rows = llm_model_catalog(extra_models)
    free_rows = [m for m in llm_rows if m.get("requires_key") == "no"]
    paid_rows = [m for m in llm_rows if m not in free_rows]
    model_group = st.radio("Model group", ["Free / no key", "Paid / key required"], horizontal=False)
    active_rows = free_rows if model_group.startswith("Free") else paid_rows
    llm_choice = st.selectbox("LLM model", [m["label"] for m in active_rows])
    selected_llm = active_rows[[m["label"] for m in active_rows].index(llm_choice)]
    provider = apply_provider(selected_llm)
    key_env = selected_llm.get("key_env", "")
    if model_group.startswith("Paid") and key_env:
        pasted_key = st.text_input(f"Key for {key_env}", os.getenv(key_env, ""), type="password")
        if pasted_key:
            os.environ[key_env] = pasted_key
        if os.getenv(key_env):
            st.caption(f"{key_env}: key loaded")
    else:
        st.caption("Selected model does not require a key.")

    with st.expander("Custom OpenAI-compatible endpoint"):
        os.environ["OLLAMA_BASE_URL"] = st.text_input("Ollama base URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
        os.environ["OLLAMA_MODEL"] = st.text_input("Ollama model", os.getenv("OLLAMA_MODEL", "llama3.1"))
        os.environ["CUSTOM_LLM_BASE_URL"] = st.text_input("Custom base URL", os.getenv("CUSTOM_LLM_BASE_URL", ""))
        os.environ["CUSTOM_LLM_MODEL"] = st.text_input("Custom model", os.getenv("CUSTOM_LLM_MODEL", ""))
        custom_key_env = st.text_input("Custom key env", os.getenv("CUSTOM_LLM_API_KEY_ENV", "CUSTOM_LLM_API_KEY"))
        os.environ["CUSTOM_LLM_API_KEY_ENV"] = custom_key_env

    st.divider()
    retrieval = st.selectbox("Retrieval", ["TF-IDF", "OpenAI text-embedding-3-large", "Pinecone"])
    ocr = st.selectbox("OCR", [f"{m['label']} | {m['pricing']}" for m in ocr_model_options()])
    os.environ["OCR_ENGINE"] = ocr_model_options()[[f"{m['label']} | {m['pricing']}" for m in ocr_model_options()].index(ocr)]["engine"]
    os.environ["OCR_LANG"] = st.text_input("OCR language", os.getenv("OCR_LANG", "eng"))
    trans = st.selectbox("Transliteration", [m["label"] for m in transliteration_options()])
    os.environ["TRANSLITERATION_ENGINE"] = transliteration_options()[[m["label"] for m in transliteration_options()].index(trans)]["engine"]
    stt = st.selectbox("Speech to text", [m["label"] for m in speech_to_text_options()])
    os.environ["STT_ENGINE"] = speech_to_text_options()[[m["label"] for m in speech_to_text_options()].index(stt)]["engine"]

    st.divider()
    lawful = st.checkbox("Lawful basis/consent for personal data")
    cloud = st.checkbox("Allow cloud processing")
    os.environ["DPDP_LAWFUL_BASIS"] = str(lawful).lower()
    os.environ["DPDP_CLOUD_CONSENT"] = str(lawful and cloud).lower()
    os.environ["DPDP_REDACT"] = str(st.checkbox("Redact personal identifiers", value=True)).lower()
    os.environ["HUMAN_REVIEW_CONFIRMED"] = str(st.checkbox("Human reviewer responsible")).lower()
    os.environ["REQUIRE_HUMAN_EXPORT_APPROVAL"] = str(st.checkbox("Require approval before export", value=True)).lower()
    top_k = st.slider("Evidence", 3, 15, 8)

paths = [save_upload(f) for f in uploads] if uploads else ([Path(local_path)] if local_path else [])
web_urls = [u.strip() for u in urls.splitlines() if u.strip()] if fetch_ok else []
with st.spinner("Indexing evidence..."):
    corpus, summary = build_corpus_from_paths(paths) if paths else ([], "No files.")
    if web_urls:
        web_corpus, web_summary = build_corpus_from_urls(web_urls, jurisdiction)
        corpus.extend(web_corpus)
        summary += " " + web_summary
    cid = corpus_id(paths)
    if corpus:
        save_corpus_pg(corpus, cid)

metadata = corpus_metadata(corpus, cid)
supabase_log_metadata(metadata)
if retrieval == "Pinecone" and corpus:
    st.caption("Pinecone: indexed" if pinecone_upsert(corpus, cid) else "Pinecone: not configured or indexing failed")
st.success(summary)
status_cols = st.columns(4)
status_cols[0].metric("Chunks", len(corpus))
status_cols[1].metric("Sources", metadata.get("source_count", 0))
status_cols[2].metric("Provider", provider)
status_cols[3].metric("Jurisdiction", jurisdiction)
st.markdown(
    "".join(
        [
            f'<span class="chip {"ok" if os.getenv("HUMAN_REVIEW_CONFIRMED") == "true" else "danger"}">Human review {"on" if os.getenv("HUMAN_REVIEW_CONFIRMED") == "true" else "pending"}</span>',
            f'<span class="chip {"ok" if os.getenv("DPDP_REDACT") == "true" else "danger"}">Redaction {os.getenv("DPDP_REDACT")}</span>',
            f'<span class="chip">OCR {os.getenv("OCR_ENGINE", "tesseract")}</span>',
            f'<span class="chip">STT {os.getenv("STT_ENGINE", "manual")}</span>',
        ]
    ),
    unsafe_allow_html=True,
)

if "brief_text" not in st.session_state:
    st.session_state["brief_text"] = ""

action = st.selectbox(
    "Action",
    [
        "Chat",
        "Agent chat",
        "Ask suggestions",
        "Vector knowledge",
        "Live search",
        "AI policy scan",
        "Study quiz",
        "Website",
        "App blueprint",
        "Codex workflow",
        "Template",
        "Voiceover",
        "Marketing",
        "Media inventory",
        "Integrations",
        "Swarm",
        "Toolbox",
        "Compliance",
        "Metadata",
    ],
)
if st.session_state.get("pending_brief_text"):
    st.session_state["brief_text"] = st.session_state.pop("pending_brief_text")
st.markdown("### Ask")
brief = st.text_area("Brief / query", height=120, placeholder="Ask or describe what you want.", key="brief_text")

with st.container(border=True):
    st.caption("Try asking")
    suggestions = ask_suggestions(corpus, 4)
    cols = st.columns(2)
    for i, q in enumerate(suggestions):
        if cols[i % 2].button(q, key=f"suggest_{i}"):
            brief = q
            st.session_state["suggested_brief"] = q
    brief = st.session_state.get("suggested_brief", brief)
    mic_audio = st.audio_input("Mic") if hasattr(st, "audio_input") else None
    audio = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg", "webm"])
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
        st.rerun()

if use_tavily and brief and (action == "Live search" or needs_live_search(brief)):
    with st.spinner("Adding Tavily live evidence..."):
        live_corpus, live_summary = build_corpus_from_tavily(brief, max_results=5)
        corpus.extend(live_corpus)
        summary += " " + live_summary
        metadata = corpus_metadata(corpus, cid)
        st.caption(live_summary)

with st.expander("Evidence preview", expanded=False):
    if retrieval == "Pinecone":
        hits = pinecone_retrieve(corpus, brief or "summary", top_k, cid)
    else:
        hits = embedding_retrieve(corpus, brief or "summary", top_k) if corpus and retrieval.startswith("OpenAI") else retrieve(corpus, brief or "summary", top_k)
    st.text(format_context(hits) if hits else "No indexed evidence yet. Enable Tavily live search or upload documents for grounded evidence.")

run = st.button("Run", type="primary")
if not run:
    st.stop()

if action == "Chat":
    if not corpus and not use_tavily:
        result = {"answer": "Live chat is available, but no evidence is indexed. Upload documents or enable Tavily live search for grounded answers.", "sources": [], "provider": "local", "model": "no-evidence"}
    else:
        result = asyncio.run(
            answer_rag_chat(
                brief,
                corpus,
                provider=provider,
                top_k=top_k,
                retrieval_engine="openai_embeddings" if retrieval in {"OpenAI text-embedding-3-large", "Pinecone"} else "tfidf",
            )
        )
    st.markdown(result["answer"])
    log_query_pg(cid, brief, result["answer"], result.get("provider", ""), result.get("model", ""))
    show_download("answer", json.dumps(result, indent=2), "answer.json", "application/json")

elif action == "Agent chat":
    result = asyncio.run(answer_with_agent_pipeline_from_corpus(brief, corpus, summary, provider))
    st.markdown(result["answer"])
    st.json(result.get("conversation", []))
    show_download("agent answer", json.dumps(result, indent=2), "agent_answer.json", "application/json")

elif action == "Ask suggestions":
    out = ask_suggestions(corpus)
    st.markdown("\n".join(f"- {q}" for q in out))
    show_download("ask suggestions", json.dumps(out, indent=2), "ask_suggestions.json", "application/json")

elif action == "Vector knowledge":
    out = vector_space_knowledge(corpus, brief or "entire corpus", k=25)
    st.json(out["summary"])
    with st.expander("Top evidence"):
        st.text(format_context(out["top_evidence"], max_chars=14000))
    st.markdown("**Suggested questions**")
    st.markdown("\n".join(f"- {q}" for q in out["suggested_questions"]))
    show_download("vector knowledge", json.dumps(out, indent=2), "vector_knowledge.json", "application/json")

elif action == "Live search":
    out = vector_space_knowledge(corpus, brief or "live search", k=25)
    st.json(out["summary"])
    st.text(format_context(out["top_evidence"], max_chars=14000))
    show_download("live search evidence", json.dumps(out, indent=2), "live_search_evidence.json", "application/json")

elif action == "AI policy scan":
    profiles = ["All"] + [p["name"] for p in ai_policy_profiles()]
    profile = st.selectbox("Policy profile", profiles)
    out = ai_policy_scan(profile, jurisdiction)
    st.json(out)
    show_download("AI policy scan", json.dumps(out, indent=2), "ai_policy_scan.json", "application/json")

elif action == "Study quiz":
    c1, c2, c3 = st.columns(3)
    exam = c1.text_input("Exam", "School / University Exam")
    difficulty = c2.selectbox("Difficulty", ["easy", "medium", "hard"])
    mode = c3.selectbox("Mode", ["question_paper", "quiz", "flashcards"])
    count = st.slider("Questions", 5, 50, 10)
    out = study_quiz_generator(corpus, exam, brief or "uploaded syllabus", count, difficulty, mode)
    st.markdown(out)
    show_download("study quiz", out, f"{mode}.md", "text/markdown")

elif action == "Website":
    page = build_website(brief, corpus, "Evidence Studio", "Evidence-grounded publication")
    components.html(page["html"], height=600, scrolling=True)
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

elif action == "Marketing":
    out = marketing_plan(brief, corpus, integration_registry())
    st.markdown(out)
    show_download("marketing plan", out, "marketing_plan.md", "text/markdown")

elif action == "Media inventory":
    out = media_inventory(corpus)
    st.dataframe(out, use_container_width=True)
    show_download("media inventory", json.dumps(out, indent=2), "media_inventory.json", "application/json")

elif action == "Integrations":
    custom = st.text_area("Add integrations", placeholder="Tool, category, pricing, use, base_url, model, key_env, score")
    out = integration_registry(custom)
    if st.button("Save integrations"):
        st.success("Saved." if upsert_integrations_pg(out) else "PostgreSQL not connected.")
    if load_integrations_pg():
        st.caption("Loaded registry rows from PostgreSQL.")
    st.dataframe(out, use_container_width=True)
    show_download("integrations", json.dumps(out, indent=2), "integrations.json", "application/json")

elif action == "Swarm":
    if "swarm_state" not in st.session_state:
        st.session_state["swarm_state"] = swarm_initial_state()
    state = st.session_state["swarm_state"]
    topology = st.selectbox("Topology", state["available_topologies"])
    state["topology"] = topology
    st.markdown(f"```mermaid\n{swarm_mermaid(state, topology)}\n```")
    agent = st.selectbox("Agent", [a["name"] for a in state["agents"]])
    c1, c2 = st.columns(2)
    if c1.button("Positive feedback"):
        st.session_state["swarm_state"] = update_swarm_feedback(state, agent, "positive")
        st.rerun()
    if c2.button("Negative feedback"):
        st.session_state["swarm_state"] = update_swarm_feedback(state, agent, "negative")
        st.rerun()
    st.dataframe(st.session_state["swarm_state"]["agents"], use_container_width=True)
    show_download("swarm", json.dumps(st.session_state["swarm_state"], indent=2), "swarm_state.json", "application/json")

elif action == "Toolbox":
    out = toolbox_catalog()
    st.dataframe(out, use_container_width=True)
    show_download("toolbox", json.dumps(out, indent=2), "toolbox_catalog.json", "application/json")

elif action == "Compliance":
    out = compliance_report(corpus)
    st.json(out)
    show_download("compliance", json.dumps(out, indent=2), "compliance_report.json", "application/json")

else:
    st.json(metadata)
    show_download("metadata", json.dumps(metadata, indent=2), "metadata.json", "application/json")
