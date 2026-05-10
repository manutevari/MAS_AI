# Scientific RAG

A small Streamlit app for evidence-grounded scientific question answering over uploaded documents.

## What It Does

Scientific RAG Studio is a compact Streamlit workspace for grounded research, study, and publishing workflows.

Core capabilities:

- evidence-grounded chat over files, permitted URLs, Tavily live evidence, or vector stores
- TF-IDF, OpenAI embeddings, and optional Pinecone retrieval
- PDF, text, spreadsheet, image/OCR, ZIP, and web evidence ingestion
- automatic LLM-assisted transliteration for non-Latin query/OCR/evidence text when an approved provider is active
- OCR presets for English, Hindi, Urdu, Indian languages, Arabic, CJK, European languages, and custom Tesseract codes
- section-aware semantic chunking with optional mBERT breakpoints through `bert-base-multilingual-cased`
- strict anti-hallucination guardrails with citations and missing-evidence handling
- NotebookLM-style, Physics Wallah-style, and textbook-style quizzes, flashcards, solutions, and question papers from uploaded sources
- school clerk automation for result sheets, attendance, fee reminders, certificates, roll lists, notices, and parent communication drafts
- website, app blueprint, marketing, media, voiceover, and template generation
- WhatsApp Business automation drafts, payloads, policy checklist, and optional Cloud API send
- prompt-aware website builder with critic review, SEO suggestions, pro tips, media/audio sections, and source evidence
- Mermaid-style mindmap, flowchart, concept-map, and SVG graphic generation from retrieved evidence
- segregated free/no-key and paid/key-required model selectors
- Ollama, OpenAI, Claude, Grok, Gemini, Hugging Face, OpenRouter, and custom endpoints
- India DPDP plus international compliance controls
- human-in-the-loop approvals, metadata exports, and swarm governance
- hidden smart router that sends mic/text/doc/URL queries to the right agent and tool workflow
- mic transcription can auto-route through smart workflow selection to execute the required task
- course-inspired metrics node for RAG quality, feedback, API integration readiness, and MCP server planning
- relationship-manager architecture with structured schemas, shared state, intent classification, RAG product answers, EMI calculation, lead drafts, and product-agent routing

Storage and search:

- PostgreSQL or Supabase Postgres for logs and chunks
- Supabase API metadata logging
- Pinecone vector retrieval
- Tavily opt-in live search from Streamlit secrets only
- compliant latest-update ingestion into PostgreSQL and Pinecone when configured

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
- `claude`: `ANTHROPIC_API_KEY`
- `grok`: `GROK_API_KEY`
- `huggingface`: `HF_TOKEN`
- `openrouter`: `OPENROUTER_API_KEY`
- `ollama`: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, no real key required
- `custom`: `CUSTOM_LLM_BASE_URL`, `CUSTOM_LLM_MODEL`, and `CUSTOM_LLM_API_KEY`

Set provider keys in Streamlit secrets or environment variables.

For live search, set `TAVILY_API_KEY` in Streamlit secrets/TOML or environment. The UI does not accept Tavily keys directly; it only shows a live-search toggle and configured/missing status.

The sidebar shows the required key input for the selected model, for example `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `HF_TOKEN`, or a custom key env var.
For no-key free/local models, the key input is hidden. Password fields mask keys, and metadata exports never include raw keys. The UI never displays keys, partial keys, or key fingerprints.

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
- URLs can be typed in the URL box, pasted in the chat prompt, or found inside uploaded TXT/PDF/CSV/XLSX/JSON/image OCR/ZIP evidence.
- YouTube watch, short, live, embed, and `youtu.be` links are treated as transcript sources when public captions are available.
- User confirms permission under robots.txt, site terms, copyright, and local law.
- The fetcher checks `robots.txt` before access.
- Only HTTP/HTTPS URLs are allowed.
- Large pages are blocked by a size limit.
- Personal identifiers can be redacted before indexing.
- Jurisdiction can be set to India, EU/EEA, California, UK, or Global/Unknown.
- Permitted URLs are transcribed into text, section/semantic chunked, and cited as `web` evidence beside uploaded documents.
- YouTube transcript chunks preserve start timestamps and watch links so answers can point back to the relevant moment.
- If a platform blocks server-side transcript access, upload the matching `.srt` or `.vtt` subtitle file and it will be indexed as transcript evidence.

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

## Pinecone And Supabase

Optional vector/database services:

```toml
PINECONE_API_KEY = ""
PINECONE_INDEX = ""
PINECONE_NAMESPACE = "default"

SUPABASE_URL = ""
SUPABASE_SERVICE_ROLE_KEY = ""
SUPABASE_METADATA_TABLE = "rag_metadata"
```

- Pinecone stores OpenAI `text-embedding-3-large` vectors and can be selected in `Retrieval`.
- Pinecone falls back to OpenAI embedding retrieval or TF-IDF if not configured.
- Supabase can be used through `DATABASE_URL` for hosted PostgreSQL.
- Supabase API metadata logging is optional through `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.

## Files

- `streamlit_app.py`: UI
- `multi_agent.py`: compact RAG core
- `requirements.txt`: deployment dependencies

## Modern Query-First UI

The interface is intentionally one-screen and chat-oriented:

- sidebar: files and URLs first; model, retrieval, OCR/STT, privacy, and approval controls are grouped in collapsed panels
- main screen: compact Chat, Evidence, and Debug tabs with short prompt chips and an icon-only run button
- output area: ChatGPT/Claude-style user and assistant messages with a right-side evidence/source column for RAG answers
- presets: Fast, Balanced, Deep Research, and Offline configure retrieval depth and chunking defaults
- density modes: Compact, Comfortable, and Ultra Compact adjust spacing for different screens
- tools stay hidden unless advanced manual override is opened
- Hindi/Devanagari content uses Devanagari font fallbacks such as Noto Sans Devanagari, Nirmala UI, and Mangal
- every generated output can be exported after human approval as original format, PDF, PNG image, SVG image, ZIP bundle, Markdown, text, HTML, JSON, or CSV

The default UI is query-first. The app chooses the workflow internally. Advanced manual override can expose Chat, Agent chat, Ask suggestions, Vector knowledge, Live search, AI policy scan, Relationship manager, School clerk, Study quiz, Website, App blueprint, Codex workflow, Template, Voiceover, Marketing, Media inventory, Mindmap, Visual maps, Integrations, Swarm, Toolbox, Compliance, and Metadata.

## Relationship Manager Architecture

The project incorporates the uploaded architecture diagram as a reusable workflow:

- offline document indexing path: load files, chunk text, create embeddings/vector records, and keep source metadata
- retrieve-only RAG service contract: `retrieve_context(query, top_k, max_chars)` returns retrieved docs and a source-block context without calling an LLM
- optional MiniLM semantic retriever uses `sentence-transformers/paraphrase-MiniLM-L6-v2` with TF-IDF fallback
- structured output schemas: `RAGDecision`, `ProductAgentResponse`, `CustomerInfo`, `EMIInput`, and `LeadPayload`
- explicit `RAGAnswer` packets preserve answer text, citations, limitations, confidence, and grounded status
- shared state: product workspaces for personal loan, two-wheeler loan, and generic relationship-manager tasks
- module map/audit trail mirrors the diagram layers: indexing, schemas, shared state, graph orchestration, RAG, EMI, lead service, and product agents
- relationship manager layer: classifies generic, product-information, EMI, lead/application, and unclear queries
- RAG retrieval service: product-information queries retrieve uploaded product evidence and generate grounded answers
- product agents: selected by product, then route to collect customer info, calculate EMI, or create a draft lead
- human remains above the relationship manager, product agents, lead creation, exports, messaging, and financial use

## Smart Auto Routing

Smart routing keeps the orchestration manager inside the pipeline instead of exposing it as a visible tool:

- accepts text, mic transcription, uploaded documents, ZIP/PDF/image/spreadsheet evidence, and permitted URL/live evidence
- selects the smallest useful workflow such as Chat, Agent chat, School clerk, Study quiz, Website, Visual maps, Marketing, WhatsApp automation, Compliance, or Live search
- uses the selected LLM as an internal router when a local/approved provider is available, otherwise falls back to deterministic rules
- can show compact routing details only inside the collapsed routing audit
- keeps human review above the orchestrator and all agents

## Metrics, RAG APIs, And MCP Readiness

The `Metrics` node implements the course-session takeaway: metrics first, then RAG/API integration, then MCP server work.

- tracks corpus coverage, source count, retrieval score, numerical/table evidence, API readiness, and vector/storage readiness
- computes BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, and METEOR answer-vs-evidence scores as a lexical evaluation matrix
- logs chat/agent events locally through `monitoring.py`
- supports optional LangSmith, W&B, and Evidently when their packages/keys are configured
- captures thumbs-up/down feedback for future evaluation sets
- produces a RAG + API integration plan and an MCP server tool/resource blueprint
- keeps MCP exposure behind stable typed tools such as `retrieve_evidence`, `answer_grounded`, `create_quiz`, `build_website`, `visual_map`, and `log_feedback`

## School Clerk Automation

The `School clerk` action supports school-office workflows:

- result sheet / marksheet generation from uploaded CSV/XLSX mark tables
- pass/fail, total, percentage, grade, and downloadable CSV result sheet
- attendance, fee reminder, admission register, transfer certificate, bonafide certificate, and parent notice templates
- pro tips for clean school data entry and safer parent communication
- human approval checklist before export, publication, or messaging

## Visual Maps

The `Mindmap` and `Visual maps` actions generate NotebookLM/Google-LM-style study visuals from retrieved evidence:

- groups topics by source
- nests sections and evidence snippets
- creates mindmaps, flowcharts, and concept maps
- previews rendered Mermaid where supported
- renders a downloadable SVG graphic image
- exports `.mmd`, `.svg`, and JSON evidence outlines

## Study Quiz / Question Paper

The `Study quiz` action is inspired by NotebookLM-style source-grounded studying, Physics Wallah-style practice, and textbook-style worked solutions:

- generate exact-style question papers
- generate Physics Wallah-style MCQ practice with feedback
- generate textbook-style step-by-step source-backed solutions
- generate assertion-reason questions
- generate quizzes with answer keys
- run a live exam with vertical options, answer reveal after submission, points, and a final score card
- show a remark for the selected answer and option-wise definitions/notes after submission
- generate flashcards
- choose exam name, topic, difficulty, and number of questions
- use uploaded documents as the only source of truth
- include citations to source file/page/section
- include weak-topic signals and revision tips
- tell students to write “Not found in uploaded documents” when evidence is missing

## ChatGPT / Claude / Microsoft Copilot Policy Profiles

The `AI policy scan` action uses official URLs only:

- OpenAI policies, usage policies, and privacy policy
- Anthropic consumer/commercial/privacy/usage policy resources
- Microsoft Copilot terms, privacy, and Microsoft 365 Copilot enterprise data protection resources

The app does not bypass access controls or scrape arbitrary policy databases. It uses compliant URL fetching with robots.txt checks where applicable, source excerpts, jurisdiction notes, and a legal-review warning. Microsoft Copilot is treated as a policy/institution profile unless you connect an authorized enterprise/custom endpoint.

## Tavily Live Search

Tavily is integrated as a controlled evidence source:

- requires `TAVILY_API_KEY`
- uses `https://api.tavily.com/search`
- does not use Tavily's generated answer as final truth
- converts Tavily result snippets into cited `live_web` evidence chunks
- live search is opt-in from the sidebar
- current-information queries such as latest, current, today, news, rule, guideline, or free model list can trigger it when enabled
- grounded-answer guardrails still apply

## Latest Update Ingestion

The `Ingest latest updates` action updates your evidence store without unrestricted scraping:

- Uses Tavily snippets when `TAVILY_API_KEY` is configured.
- Optionally includes user-provided permitted URLs that pass the compliance fetcher.
- Stores update chunks in PostgreSQL when `DATABASE_URL` is configured.
- Stores vector embeddings in Pinecone when `PINECONE_API_KEY`, `PINECONE_INDEX`, and `OPENAI_API_KEY` are configured.
- Includes jurisdiction policy notes for India, EU/EEA, UK, California, or Global/Unknown.
- Keeps guardrails for robots.txt, site terms, copyright/database rights, institutional policies, privacy, and human review.

## WhatsApp Automation

The `WhatsApp automation` action supports compliant outreach workflows:

- draft WhatsApp-safe outreach copy
- prepare approved-template payload drafts
- prepare Cloud API text payload drafts
- include website/service URLs
- redact common Indian personal identifiers
- require human opt-in/compliance confirmation before optional sending
- optional send through official Meta WhatsApp Cloud API

Required secrets for sending:

```toml
WHATSAPP_TOKEN = ""
WHATSAPP_PHONE_NUMBER_ID = ""
WHATSAPP_BUSINESS_ACCOUNT_ID = ""
```

Policy notes:

- Use only opted-in recipients.
- Business-initiated messages generally require approved templates.
- Free-form replies are for the customer-service window.
- Do not use the tool for unsolicited spam.
- Review Meta WhatsApp Business Platform policies, template rules, pricing, local privacy law, and institutional/government communication rules before sending.

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

Set `OCR_LANG` to a Tesseract language code such as `eng`, `hin`, `urd`, `ara`, `ben`, `tam`, `tel`, `fra`, or `eng+hin+urd`. The server must have the matching Tesseract language data installed.

The advanced processing panel includes presets for English, Hindi, Urdu, Arabic, Sanskrit, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Nepali, Sinhala, Chinese, Japanese, Korean, French, German, Spanish, Russian, and a custom Tesseract code field.

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

Transliteration is hidden from the normal UI and runs automatically. The model keeps the original script, adds roman transliteration on first mention, does not translate meaning unless asked, and marks uncertain OCR/transliteration as approximate. If cloud processing is not approved or no LLM key/local provider is available, the app preserves original script and states that automatic LLM transliteration was not performed. The grounding guard still applies: transliterated text must be treated as OCR-derived evidence and uncertainty must be stated when the source text is unclear.

Advanced transliteration targets include Indic NLP Library, Aksharamukha, Indic transliteration rules, iNLTK target mode, Google Input Tools guidance, Bhashini target mode, and LLM-assisted transliteration. Optional NLP libraries are used when installed; otherwise the app keeps original text and lets the selected approved LLM handle transliteration in the answer.

## Translation / Answer Language

Transliteration is not translation. Transliteration changes script/sound form, for example Arabic or Urdu script into roman letters. Translation changes meaning into another language, for example English evidence into Hindi.

Use the collapsed `Retrieval and input processing` panel and set `Answer language` to `Hindi - Devanagari`, `Urdu`, `English`, or another supported language. You can also ask directly: `answer in Hindi`, `translate summary into Urdu`, or `explain in English`. The LLM translates the final answer while preserving filenames, page numbers, citations, numeric values, names, and source quotes. If no approved LLM provider is active, the app will preserve retrieved evidence and show a translation note.

## Semantic Chunking

Default chunking is section-aware and sentence-based so tables, headings, and document structure are less likely to be broken arbitrarily. Set `CHUNKING_ENGINE=mbert` to enable optional mBERT semantic breakpoints with `MBERT_MODEL=bert-base-multilingual-cased`. If `transformers`/`torch` or the model are unavailable, the app falls back to local section-aware chunking.

## Speech To Text

The chat tab includes a speech input helper for typing queries:

- Browser microphone input: available when Streamlit supports `st.audio_input`.
- Mic recordings transcribe automatically after the user stops recording; the transcript is placed into the prompt on the next rerun.
- Manual transcript paste: free and always available.
- OpenAI Whisper API: paid/cloud, implemented through `OPENAI_API_KEY` and `OPENAI_STT_MODEL`.
- Whisper local/faster-whisper: free/local, selectable integration target.
- Google Speech-to-Text: paid/cloud, selectable integration target.
- Azure Speech: paid/cloud, selectable integration target.
- Bhashini ASR: India-focused, platform dependent, selectable integration target.

Speech transcripts are used only as query text. Answers still come from uploaded document evidence and pass through the grounding guard.

## Text To Speech / Audio Generator

The app can also talk back like an assistant:

- `Talk back` in the processing popover adds a browser Speak/Stop control to RAG answers.
- Browser speech synthesis is free/local and requires no key.
- OpenAI TTS can generate downloadable MP3 audio when `OPENAI_API_KEY` is configured.
- Edge TTS can generate MP3 audio when the optional `edge-tts` package is installed.

The voiceover tab supports free and paid TTS planning and in-app generation where available:

- Browser assistant voice: free/local, instant spoken output.
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
