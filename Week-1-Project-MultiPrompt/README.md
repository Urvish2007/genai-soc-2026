# 🔥 PromptForge — Multi-Mode AI Assistant

Week 1 project for GenAI SoC 2026. A Gradio app with 4 selectable AI personas, few-shot prompting, streaming responses, and JSON rendering.

---

## Personas

| Mode | Style | Output |
|---|---|---|
| Technical Explainer | Jargon-free analogies | Text |
| Debate Coach | Both sides + synthesis | Text |
| Code Reviewer | Issues, suggestions, severity | JSON → Markdown |
| Creative Writer | Vivid, sensory prose | Text |

---

## Setup

**1. Clone and enter the folder**
```bash
git clone https://github.com/YOUR_USERNAME/genai-soc-2026.git
cd genai-soc-2026/week1-promptforge
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**
```bash
cp .env.example .env
# Open .env and replace the placeholder with your real key
```
Get a free key at [console.groq.com](https://console.groq.com).

**5. Run the app**
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser.

---

## Screenshots

> Add screenshots here after running the app.
> One per mode: Technical Explainer, Debate Coach, Code Reviewer, Creative Writer.

### Technical Explainer
![Technical Explainer](screenshots/technical_explainer.png)

### Debate Coach
![Debate Coach](screenshots/debate_coach.png)

### Code Reviewer
![Code Reviewer](screenshots/code_reviewer.png)

### Creative Writer
![Creative Writer](screenshots/creative_writer.png)

---

## Project Structure

```
week1-promptforge/
├── app.py
├── requirements.txt
├── .env.example
├── .env               ← not committed (in .gitignore)
└── README.md
```

---

## Features

- **4 AI Personas** — each with a unique system prompt and few-shot examples
- **Streaming** — responses appear token by token
- **Temperature Slider** — 0.0 to 1.5, adjust creativity live
- **System Prompt Viewer** — accordion shows exactly what the model is told
- **JSON Mode** — Code Reviewer parses structured output into formatted Markdown

---

## .gitignore tip

Make sure `.env` is ignored:
```
.env
venv/
__pycache__/
```