# 👁️ HybridSight — RAG + Web Search + Vision Agent

> A single [LangGraph](https://langchain-ai.github.io/langgraph/) agent that answers from your uploaded PDFs, the live web, and uploaded images — in one conversation, with full memory and a visible reasoning trace — wrapped in a [Gradio](https://www.gradio.app/) UI.

Built as part of Week 5 of the MSOC-26 Generative AI program.

---

## ✨ Features

- **📄 RAG over your own documents** — upload a PDF, it's chunked and embedded into ChromaDB, and the agent retrieves grounded, page-cited answers from it.
- **🌐 Live web search** — current events and recent news are answered via DuckDuckGo, not stale training data.
- **📚 General knowledge** — well-established facts are routed to Wikipedia.
- **🖼️ Vision** — upload an image and ask about it; a vision-capable LLM describes it before the main agent reasons over the description.
- **🧠 Full conversational memory** — powered by LangGraph's checkpointer, so follow-up questions retain context.
- **🔍 Visible reasoning trace** — every tool call (name + input) is shown in a live sidebar, so you can see exactly which tool answered which part of a question.
- **Graceful failure everywhere** — no raw stack traces reach the user; empty ChromaDB, failed image decodes, and flaky web/Wikipedia calls all degrade to a readable message instead of crashing the app.

---

## 🏗️ Architecture

```
                         ┌─────────────────────┐
                         │   Gradio UI (app.py) │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                   │
          PDF Upload          Text Message         Image Upload
                 │                  │                   │
                 ▼                  ▼                   ▼
          ingest.py         LangGraph Agent      tools_vision.py
        (chunk+embed)         (agent.py)          (description
                 │                  │               generated
                 ▼                  │               up front,
        ┌─────────────────┐         │              fed into the
        │  ChromaDB store   │◄──────┤              message as
        │ (chroma_client.py)│       │              plain text)
        └─────────────────┘         │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Tools available to the agent │
                    │  • search_documents  (RAG)    │
                    │  • duckduckgo_search (live web)│
                    │  • wikipedia         (general)│
                    │  • describe_image    (vision) │
                    └───────────────────────────────┘
```

**Why images bypass the agent's tool-calling:** early versions routed image data through the LLM as a base64 string it had to copy verbatim into a tool call — unreliable and token-expensive. Instead, `app.py` calls the vision model directly on upload, and hands the agent a short text description, keeping routing simple and deterministic.

---

## 🛠️ Tech Stack

| Layer | Choice |
|---|---|
| Agent orchestration | LangGraph (`create_react_agent`) |
| LLM (reasoning) | Groq — `openai/gpt-oss-120b` |
| LLM (vision) | Groq — `meta-llama/llama-4-scout-17b-16e-instruct` |
| Vector store | ChromaDB (persistent, local) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (via `langchain-huggingface`) |
| Web search | DuckDuckGo |
| General knowledge | Wikipedia |
| UI | Gradio 6 |

> **Note on model IDs:** Groq deprecates and renames model IDs frequently. The defaults above are current as of this build; both are overridable via environment variables (`GROQ_AGENT_MODEL`, `GROQ_VISION_MODEL`) without touching code. Check [console.groq.com/docs/models](https://console.groq.com/docs/models) before deploying.

---

## 📂 Project Structure

```
week5-hybridsight/
├── agent.py            # LangGraph agent definition, system prompt, routing rules
├── app.py               # Gradio UI + chat orchestration
├── chroma_client.py      # Single shared ChromaDB client (avoids dual-client conflicts)
├── ingest.py             # PDF chunking + embedding pipeline
├── tools_rag.py          # search_documents tool
├── tools_vision.py        # describe_image tool + direct-call implementation
├── tools_web.py           # duckduckgo_search & wikipedia tools, with graceful failure handling
├── image_utils.py         # Local image → base64 data URI conversion
├── requirements.txt
├── .env.example
├── .gitignore
└── docs/
    └── screenshots/        # Test case screenshots (see below)
```

---

## 🚀 Getting Started

### 1. Clone and set up a virtual environment

```bash
git clone <your-repo-url>
cd week5-hybridsight
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> This project pins dependency versions deliberately — the LangChain/LangGraph/Gradio ecosystem ships breaking changes often. If you hit a build error on `numpy` during install, it usually means your Python version is too new for the pinned `numpy` wheel; **Python 3.11 or 3.12** is recommended over 3.13+.

### 3. Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and add your [Groq API key](https://console.groq.com/keys):

```
GROQ_API_KEY=gsk_your_actual_key_here
```

⚠️ **Never commit your real `.env` file.** It's already in `.gitignore` — keep it that way.

### 4. Run the app

```bash
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## 🧪 Test Cases

All 5 required scenarios were run and verified. Screenshots below.

### 1. Question answerable only from an uploaded PDF → `search_documents`

Uploaded a personal résumé PDF, then asked the agent to summarize and list its main topics. The trace confirms `search_documents` was activated, and the answer is grounded in the retrieved chunks.


<img width="1918" height="1078" alt="test1_rag_query" src="https://github.com/user-attachments/assets/3f7fa53e-21ba-44e6-994e-45dc0effe286" />

<img width="1918" height="1077" alt="test1_rag_answer" src="https://github.com/user-attachments/assets/e610b8a5-39e0-4c69-bc7d-69d3a0b7b446" />

### 2. Current event from this week → `duckduckgo_search`

Asked about the latest news on India's space program. The trace shows `duckduckgo_search` was called with a live query, and the answer is dated and current — not stale training data.

<img width="1918" height="1078" alt="test2_duckduckgo" src="https://github.com/user-attachments/assets/9907a791-fd05-420c-9613-3bce275ad327" />


### 3. Upload an image and ask "what's in this picture?" → `describe_image`

Uploaded a photo of a desk setup. The trace confirms `describe_image` ran on the upload, and the agent's final answer is grounded in that description.

<img width="1918" height="1073" alt="test3_vision" src="https://github.com/user-attachments/assets/eea67f23-0695-42cf-8528-9a3eefa16786" />


### 4. General knowledge question → `wikipedia`

Asked a historical/biographical question. The trace shows the `wikipedia` tool was called with a cleaned-up query, and the agent correctly disambiguated and answered from the retrieved article.

<img width="1918" height="1078" alt="test4_wikipedia" src="https://github.com/user-attachments/assets/1deb7bf0-57ad-43c3-89ac-c3b79ba487e1" />


### 5. Question before any PDF is uploaded → graceful "no documents" message

*Screenshot to be added:* asking a document-related question with an empty ChromaDB collection returns `"No documents uploaded yet. Please ask the user to upload a PDF first."` instead of an error, handled explicitly in `tools_rag.py`.

<img width="1901" height="1078" alt="image" src="https://github.com/user-attachments/assets/0f3a859b-71dd-45ad-8be2-57b76b4946c1" />

---

## ✅ Completion Checklist

- [x] `search_documents` returns grounded chunks with page numbers
- [x] `describe_image` returns coherent descriptions across multiple test images
- [x] Agent routes to the correct tool for all 5 test scenarios
- [x] Reasoning trace shows tool name + input for every step
- [x] Empty ChromaDB and invalid image inputs handled gracefully — no raw stack traces
- [x] README documented with screenshots for each test case

---

## 🔒 Security Notes

- API keys live only in `.env`, which is git-ignored. `.env.example` ships with placeholders only.
- If a key is ever accidentally committed or shared, treat it as compromised — revoke and rotate it immediately at [console.groq.com/keys](https://console.groq.com/keys).
- `chroma_store/` (your local vector DB) is also git-ignored, since it may contain content from personal/uploaded PDFs.

---

## 🗺️ Roadmap

- **Week 6:** package for deployment and ship as a public Hugging Face Space, moving `GROQ_API_KEY` into HF Spaces secrets instead of a local `.env`.

---

## 📄 License

This project was built for educational purposes as part of the MSOC-26 Generative AI curriculum.
