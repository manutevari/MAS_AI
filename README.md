# Scientific RAG

A small Streamlit app for evidence-grounded scientific question answering over uploaded documents.

## What It Does

- Upload one or more files: ZIP, PDF, TXT, Markdown, JSON, CSV/TSV, Excel, or images.
- Ask scientific questions about methods, values, tables, figures, structures, assays, or limitations.
- Retrieve relevant chunks with TF-IDF and numeric/table boosts.
- Optionally retrieve semantically with OpenAI `text-embedding-3-large`.
- Answer from uploaded scientific evidence by default.
- Refuse unsupported answers instead of hallucinating; every model answer must cite uploaded source evidence.
- Optionally allow general/open-source knowledge.
- Optionally show a simple Planner -> Executor -> Verifier conversation.
- Build a live single-file website from retrieved evidence and download `index.html`.
- Generate many template types from uploaded evidence.
- Generate Emergent-style app builder blueprints for web, mobile, SaaS, enterprise, and fintech apps.
- Generate a marketing plan grounded in uploaded documents.
- Manage a media inventory for tables, figures, images, diagrams, charts, and assets.
- Maintain a PostgreSQL-backed integration/model registry for free, paid, new, or underrated apps and services.
- Use a custom OpenAI-compatible LLM endpoint so newly available model routers can be added without code changes.
- Choose from segregated LLM dropdowns: `Free / no key` and `Paid / key required`.
- Key fields appear only for paid/key-required selections.
- Configure multilingual OCR with `OCR_LANG` for precise text extraction when Tesseract language packs are installed.
- Select OCR models from a dropdown with free/paid and key-required labels.
- Select transliteration strategy while keeping answers grounded in uploaded OCR/document evidence.
- Use speech-to-text as a typing helper for queries.
- Record query audio with the browser microphone and use the transcript as typed query text.
- Generate safe voiceover scripts for free and paid text-to-speech models.
- Include India DPDP-oriented privacy controls: lawful-basis confirmation, cloud-processing gate, and PII redaction.
- Ingest compliant web URLs with robots.txt checks, jurisdiction selection, redaction, and source citations.
- Apply international compliance readiness controls for India DPDP, EU/EEA GDPR, UK GDPR, California CCPA/CPRA, and global safe-fetch workflows.
- Keep human-in-the-loop approvals and metadata/audit exports.
- Use combined swarm topologies with agent attention weights, feedback-driven promotion/demotion, and immutable human authority above the orchestrator.
- Show a toolbox readiness matrix for every major feature, dependency, key, and integration target.
- Suggest grounded questions from the indexed corpus.
- Show `Try asking` suggestions directly under the query box with mic and upload controls.
- Live chat can open without uploaded documents; upload files or enable Tavily for grounded evidence.
- Explore the indexed vector/lexical evidence space as auditable knowledge, not unsupported memory.
- Use Tavily live search as an opt-in evidence source when current information is needed.
- Download answers as Markdown, JSON, CSV, HTML, or TXT.
- Persist indexed chunks and query logs in PostgreSQL.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Providers

Default provider is `local`, which needs no API key and returns evidence from the uploaded documents.

Embedding retrieval uses `OPENAI_API_KEY` and `text-embedding-3-large`. If no key is set, it falls back to TF-IDF.

Optional OpenAI-compatible providers:

- `openai`: `OPENAI_API_KEY`
- `grok`: `GROK_API_KEY`
- `huggingface`: `HF_TOKEN`
- `openrouter`: `OPENROUTER_API_KEY`
- `ollama`: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, no real key required
- `custom`: `CUSTOM_LLM_BASE_URL`, `CUSTOM_LLM_MODEL`, and `CUSTOM_LLM_API_KEY`

Set provider keys in Streamlit secrets or environment variables.

For live search, set `TAVILY_API_KEY`. The app uses Tavily only when the sidebar toggle is enabled and the query asks for current/latest/live information, or when the `Live search` action is selected.

The sidebar shows the required key input for the selected model, for example `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `HF_TOKEN`, or a custom key env var.
For no-key free/local models, the key input is hidden. Password fields mask keys, and metadata exports never include raw keys; the app only displays a short irreversible fingerprint when a key is set.

The LLM dropdown tries to load OpenRouter's live model catalog and labels entries as free or paid/key-required. If live loading is unavailable, built-in fallback choices remain available. Extra models can be added in this format:

```text
Label, provider, model, base_url, key_env
```

Free model availability changes frequently. OpenRouter’s `openrouter/free` router is included as the lowest-maintenance option because it selects from currently available free models.

The custom provider supports any OpenAI-compatible API. That is the practical way to use newly released or underrated LLM providers without constantly editing the app.

Ollama is available as a free/no-key local provider through its OpenAI-compatible endpoint:

```toml
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.1"
```

Run Ollama on the same machine or network as Streamlit and pull a model first, for example `ollama pull llama3.1`.

## Router Messages

The chat tab accepts advanced message JSON for large routers and OpenAI-compatible APIs:

```json
[
  {"role": "system", "content": "Extra instruction"},
  {"type": "human", "message": "Human context"},
  {"role": "assistant", "content": "Previous assistant answer"},
  {"name": "ocr", "text": "Tool result"},
  "Any plain message"
]
```

Any unknown role is normalized to `user`. The parser accepts `role`, `type`, or `kind`, and accepts `content`, `message`, `text`, or `value`. Plain strings are accepted as user messages. Tool or named messages are preserved as labeled context inside a user message so strict routers do not reject them.

## Grounding Rule

Model answers are guarded after generation. If the response does not cite uploaded filenames and page/section evidence, the app returns retrieved evidence plus a limitation note instead of allowing an unsupported answer. Scientific/research guardrails require units, numeric values, uncertainty, methods, table/figure context, and citation discipline; unsupported causality, safety, clinical relevance, novelty, or statistical significance must not be claimed.

## India Privacy And DPDP Readiness

The app includes technical safeguards aligned with India privacy expectations under the Digital Personal Data Protection Act, 2023 and DPDP Rules readiness:

- Cloud LLM/OCR processing is blocked unless the user confirms lawful basis/consent and separately allows cloud processing.
- Common Indian identifiers are detected and can be redacted before cloud calls: Aadhaar-like numbers, PAN, phone, email, UPI IDs, and account-like numbers.
- A `Compliance` tab shows detected personal-data categories and affected sources.
- The compliance report can be downloaded as JSON.
- PostgreSQL logging is optional and should be deployed with access control, retention, deletion, and breach-response procedures.

This is a technical compliance aid, not legal advice. The deployer remains responsible for notices, consent records, retention, data principal rights, children’s data handling, breach notification, vendor contracts, and cross-border transfer decisions.

## International Guidelines

The compliance profile supports India, EU/EEA, California, UK, and Global/Unknown. It is designed around common international safeguards:

- lawful basis or consent where required
- clear notice and purpose limitation
- data minimisation and redaction
- access control and security safeguards
- deletion/retention workflow
- data-principal/data-subject rights handling
- breach-response workflow
- cross-border/vendor review
- robots.txt and website terms for web ingestion
- no paywall, login, CAPTCHA, or access-control bypass
- copyright/database-rights review before reuse

## Compliant Web Ingestion

The app does not mass-scrape the internet. It supports controlled URL ingestion:

- User provides specific URLs.
- User confirms permission under robots.txt, site terms, copyright, and local law.
- The fetcher checks `robots.txt` before access.
- Only HTTP/HTTPS URLs are allowed.
- Large pages are blocked by a size limit.
- Personal identifiers can be redacted before indexing.
- Jurisdiction can be set to India, EU/EEA, California, UK, or Global/Unknown.

This keeps web context auditable and grounded instead of attempting unlawful or indiscriminate scraping.

## PostgreSQL

Set `DATABASE_URL` in Streamlit secrets:

```toml
DATABASE_URL = "postgresql://user:password@host:5432/database"
```

The app creates two small tables automatically:

- `rag_chunks`: uploaded corpus chunks
- `rag_queries`: questions, answers, provider, model, timestamps
- `rag_integrations`: editable tool/model registry with optional base URL, model name, key env var, and ranking score

## Files

- `streamlit_app.py`: UI
- `multi_agent.py`: compact RAG core
- `requirements.txt`: deployment dependencies

## Simple UI

The interface is intentionally one-screen:

- sidebar: files, URLs, models, OCR/STT, privacy, and human approval
- main screen: one `Action` dropdown, one brief/query box, one `Run` button
- output area: review result, approve, download

Actions include Chat, Agent chat, Ask suggestions, Vector knowledge, Live search, Website, App blueprint, Codex workflow, Template, Voiceover, Marketing, Media inventory, Integrations, Swarm, Toolbox, Compliance, and Metadata.

## Tavily Live Search

Tavily is integrated as a controlled evidence source:

- requires `TAVILY_API_KEY`
- uses `https://api.tavily.com/search`
- does not use Tavily's generated answer as final truth
- converts Tavily result snippets into cited `live_web` evidence chunks
- live search is opt-in from the sidebar
- current-information queries such as latest, current, today, news, rule, guideline, or free model list can trigger it when enabled
- grounded-answer guardrails still apply

## Codex-Style Workflow

The `Codex workflow` action adds a compact agentic coding/process pattern:

- workspace-first evidence reading
- scoped, reviewable changes
- tool readiness checks
- verification before packaging
- Git/GitHub handoff planning
- review-mode risk focus
- human approval above all agents
- metadata and audit export

The integration registry is intentionally user-editable instead of hard-coding a stale vendor list.

## Template Library

Included template families:

- Website: landing page, documentation site, portfolio/organization page
- Reports: evidence report, scientific summary
- Compliance: compliance report, DPDP privacy notice
- Marketing: marketing plan, email campaign, social media pack
- Business: proposal, pitch deck outline, invoice/quotation
- Forms: intake form, survey form
- Media: media asset sheet
- Integrations: integration matrix
- Evaluation: RAG evaluation sheet

Templates are generated from retrieved uploaded evidence and include limitation notes when facts are missing.

## Emergent-Style App Builder Features

The app builder tab incorporates these feature patterns:

- prompt-to-app planning
- web and mobile app targets
- end-to-end full-stack structure
- data and backend management
- authentication and access control
- workflow automation
- integrations and APIs
- deployment readiness
- GitHub/developer handoff planning
- analytics and growth planning
- advanced agent controls
- security, compliance, metadata, and human review

## Human In The Loop And Metadata

The app keeps the human reviewer on top:

- A human reviewer checkbox is shown in the privacy controls.
- Downloads/exports can require explicit human approval.
- Metadata is generated for each indexed corpus.
- Metadata includes source counts, chunk counts, pages, sections, OCR/STT/model settings, compliance jurisdiction, and approval status.
- Metadata can be downloaded as JSON for audit trails.

## Swarm Governance

The swarm topology keeps humans above all agents:

- Human reviewer is immutable final authority.
- Orchestrator is the highest agent level, never above human.
- Supported topologies: Hybrid, Hierarchy, Mesh, Star, Pipeline, Ring, Tree, Blackboard, Committee.
- Hybrid combines multiple communication structures at once.
- Positive feedback increases an agent attention weight.
- Repeated positive feedback can promote an agent up to orchestrator-candidate level.
- Negative feedback reduces attention weight.
- Repeated negative feedback demotes the agent level.
- Compliance guard and verifier report back to the human reviewer.
- Swarm state can be downloaded as JSON.

## Toolbox Readiness

The toolbox tab keeps the code modest while making the system data-rich:

- lists each feature
- marks free/paid/key requirements
- shows required packages
- shows environment variables/API keys
- marks package readiness
- marks key readiness
- exports the catalog as JSON

Heavy engines are exposed as integration targets instead of being forced into the default dependency set.

## OCR

Set `OCR_LANG` to a Tesseract language code such as `eng`, `hin`, `fra`, or `eng+hin`. The server must have the matching Tesseract language data installed.

OCR dropdown options include:

- IndicPhotoOCR: free/open ecosystem, best for Hindi and Indian scripts, selected/configured externally.
- PaddleOCR: free/local, multilingual detection + recognition, selected/configured externally.
- Tesseract OCR v5: free/local and included as the lightweight default path through `pytesseract`.
- Google Document AI: paid/cloud, requires Google credentials.
- Azure Document Intelligence: paid/cloud, requires Azure key.
- AWS Textract: paid/cloud, requires AWS credentials.
- VLM OCR: depends on the selected vision-capable LLM provider.

Only Tesseract is kept as a standard lightweight dependency in this deploy package. The other engines are exposed as selectable configuration targets so you can add their SDKs or connect them through a custom integration without bloating every deployment.

## Transliteration

The transliteration dropdown includes:

- No transliteration
- Indic transliteration rules
- Bhashini/Indic transliteration
- LLM-assisted transliteration

The grounding guard still applies: transliterated text must be treated as OCR-derived evidence and uncertainty must be stated when the source text is unclear.

## Speech To Text

The chat tab includes a speech input helper for typing queries:

- Browser microphone input: available when Streamlit supports `st.audio_input`.
- Manual transcript paste: free and always available.
- OpenAI Whisper API: paid/cloud, implemented through `OPENAI_API_KEY` and `OPENAI_STT_MODEL`.
- Whisper local/faster-whisper: free/local, selectable integration target.
- Google Speech-to-Text: paid/cloud, selectable integration target.
- Azure Speech: paid/cloud, selectable integration target.
- Bhashini ASR: India-focused, platform dependent, selectable integration target.

Speech transcripts are used only as query text. Answers still come from uploaded document evidence and pass through the grounding guard.

## Text To Speech / Audio Generator

The voiceover tab supports free and paid TTS planning:

- Galaxy.ai: free/external, multilingual, useful for creative outreach.
- QuillBot Voice: free/external, useful for clean narration.
- Airvoz: free/external, multilingual and useful for Hindi/community outreach.
- OpenAI TTS: paid/cloud, API-key based.
- Edge TTS: free/local package when installed.
- Coqui/Piper local TTS: free/local, privacy-preserving when installed.
- Festival Speech Synthesis: classic free/offline academic TTS.
- eSpeak NG: classic free/offline, compact and useful for accessibility.
- MaryTTS: underrated free/research TTS with prosody control.
- Tacotron 2: free/experimental neural TTS for learning and experiments.
- Mozilla TTS / Coqui legacy: free/community neural TTS.
- OpenAI Jukebox: free/research music and singing generation, experimental rather than routine narration.

The app prepares a safe voiceover script and redacts common Indian personal identifiers before third-party use. External free TTS sites are opened by link; review their terms, limits, privacy policy, and voice rights before publishing. Avoid celebrity/impersonation voices unless you have permission.
