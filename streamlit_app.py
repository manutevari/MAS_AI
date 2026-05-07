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
    format_context,
    integration_registry,
    llm_model_catalog,
    load_integrations_pg,
    log_query_pg,
    marketing_plan,
    media_inventory,
    needs_live_search,
    ocr_model_options,
    render_template,
    retrieve,
    save_corpus_pg,
    speech_to_text_options,
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


def secret_env() -> None:
    for key in (
        "OPENAI_API_KEY",
        "GROK_API_KEY",
        "GOOGLE_API_KEY",
        "HF_TOKEN",
        "OPENROUTER_API_KEY",
        "CUSTOM_LLM_API_KEY",
        "CUSTOM_LLM_BASE_URL",
        "CUSTOM_LLM_MODEL",
        "DATABASE_URL",
        "TAVILY_API_KEY",
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
    if provider == "openai":
        os.environ["OPENAI_MODEL"] = choice["model"]
    if provider == "grok":
        os.environ["GROK_MODEL"] = choice["model"]
    os.environ["LLM_PROVIDER"] = provider
    return provider


secret_env()

st.title("Scientific RAG")
st.caption("One screen. Upload evidence. Choose action. Human approves final output.")

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
    tavily_key = st.text_input("Tavily key", os.getenv("TAVILY_API_KEY", ""), type="password")
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
    use_tavily = st.checkbox("Use Tavily live search when needed")

    st.divider()
    extra_models = st.text_area("Extra LLMs", placeholder="Label, provider, model, base_url, key_env")
    llm_rows = llm_model_catalog(extra_models)
    llm_choice = st.selectbox("LLM model", [m["label"] for m in llm_rows])
    selected_llm = llm_rows[[m["label"] for m in llm_rows].index(llm_choice)]
    provider = apply_provider(selected_llm)
    key_env = selected_llm.get("key_env", "")
    if key_env:
        pasted_key = st.text_input(f"Key for {key_env}", os.getenv(key_env, ""), type="password")
        if pasted_key:
            os.environ[key_env] = pasted_key

    with st.expander("Custom OpenAI-compatible endpoint"):
        os.environ["CUSTOM_LLM_BASE_URL"] = st.text_input("Custom base URL", os.getenv("CUSTOM_LLM_BASE_URL", ""))
        os.environ["CUSTOM_LLM_MODEL"] = st.text_input("Custom model", os.getenv("CUSTOM_LLM_MODEL", ""))
        custom_key_env = st.text_input("Custom key env", os.getenv("CUSTOM_LLM_API_KEY_ENV", "CUSTOM_LLM_API_KEY"))
        os.environ["CUSTOM_LLM_API_KEY_ENV"] = custom_key_env

    st.divider()
    retrieval = st.selectbox("Retrieval", ["TF-IDF", "OpenAI text-embedding-3-large"])
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
st.success(summary)

action = st.selectbox(
    "Action",
    [
        "Chat",
        "Agent chat",
        "Ask suggestions",
        "Vector knowledge",
        "Live search",
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
brief = st.text_area("Brief / query", height=120, placeholder="Ask or describe what you want.")

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
        brief = transcribe_audio(speech.getvalue(), name, os.getenv("STT_ENGINE", "manual"), os.getenv("OCR_LANG", "eng").split("+")[0])
        st.session_state["suggested_brief"] = brief
        st.text_area("Transcript", brief, height=100)

if use_tavily and brief and (action == "Live search" or needs_live_search(brief)):
    with st.spinner("Adding Tavily live evidence..."):
        live_corpus, live_summary = build_corpus_from_tavily(brief, max_results=5)
        corpus.extend(live_corpus)
        summary += " " + live_summary
        metadata = corpus_metadata(corpus, cid)
        st.caption(live_summary)

with st.expander("Evidence preview"):
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
                retrieval_engine="openai_embeddings" if retrieval.startswith("OpenAI") else "tfidf",
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
