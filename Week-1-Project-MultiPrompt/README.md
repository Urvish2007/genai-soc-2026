# 🔥 PromptForge — Multi-Mode AI Assistant

> Week 1 Project — GenAI SoC 2026

PromptForge is a web app that lets you chat with an AI in 4 different modes. Each mode has a different personality and responds in a completely different style. You pick the mode, type your message, and the reply streams in word by word — just like ChatGPT.

Built with Python, Gradio (for the UI), and Groq (for the AI).

---

## What does it actually do?

When you open the app, you see a chat interface. At the top there is a dropdown where you pick one of 4 modes. Once you pick a mode, the AI behaves like that persona for the entire conversation.

Here is what each mode does:

**🧠 Technical Explainer**
You ask it any technical or complex topic — programming, science, math, anything. It explains it in simple language using real-life analogies. No jargon. No complicated words. Like having a patient teacher explain things from scratch.

Example input:
```
What is recursion?
```
What you get: A simple explanation using an analogy like Russian nesting dolls. Not a Wikipedia definition.

---

**⚖️ Debate Coach**
You give it any topic or question. It argues BOTH sides — for and against — with equal strength. Then it gives a balanced conclusion. Useful for understanding all angles of a topic.

Example input:
```
Is social media good or bad for society?
```
What you get: Strong FOR arguments, strong AGAINST arguments, then a fair conclusion.

---

**🔍 Code Reviewer**
You paste any code — Python, JavaScript, anything. It reviews it like a senior developer would. It finds bugs, security problems, and bad practices. It tells you which line has the problem and how to fix it.

This mode is special — the AI responds in JSON format behind the scenes, and the app converts it into clean readable output like this:

```
🔴 Severity: HIGH

Issues:
- Line 3: SQL injection vulnerability — user input is directly in the query

Suggestions:
- Use parameterised queries instead of f-strings
- Add input validation before querying the database
```

Example input:
```python
def get_user(id):
    query = f"SELECT * FROM users WHERE id = {id}"
    return db.execute(query)
```

---

**✍️ Creative Writer**
You give it any scene, feeling, or prompt. It writes vivid, descriptive prose — not generic AI text. Strong verbs, unexpected comparisons, specific details.

Example input:
```
Write a scene: a developer pushing code at midnight
```
What you get: A short, atmospheric piece of writing that actually feels human.

---

## How does the Temperature slider work?

Temperature controls how creative or focused the AI is.

- **0.0** — Very focused and consistent. Ask the same question twice, get almost the same answer. Good for code review.
- **0.7** — Default. Balanced between creativity and accuracy.
- **1.5** — Very creative and unpredictable. Answers vary a lot. Good for creative writing.

---

## What is the "Active System Prompt" accordion?

Every AI mode has a hidden instruction that tells the AI how to behave. This is called the system prompt. You normally never see it in apps like ChatGPT — but in PromptForge, you can click the accordion and read exactly what instructions the AI is following. When you switch modes, it updates automatically.

---

## Project structure

```
Week-1-Project-MultiPrompt/
│
├── app.py              — the entire application (one file)
├── requirements.txt    — list of Python packages needed
├── .env                — your secret API key goes here (you create this)
├── .env.example        — template showing what .env should look like
├── .gitignore          — tells Git to ignore .env and venv folder
└── README.md           — this file
```

---

## Setup — step by step

Follow these steps exactly. If you have never done this before, do not skip any step.

---

### Step 1 — Get the code

If you are cloning from GitHub:
```bash
git clone https://github.com/YOUR_USERNAME/genai-soc-2026.git
cd genai-soc-2026/Week-1-Project-MultiPrompt
```

If you already have the folder on your computer, just open it in VS Code and open the terminal inside it.

---

### Step 2 — Create a virtual environment

A virtual environment is an isolated Python setup just for this project. It keeps packages from this project separate from everything else on your computer.

```bash
python -m venv venv
```

Now activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

You will know it worked when you see `(venv)` at the start of your terminal line.

---

### Step 3 — Install the required packages

```bash
pip install -r requirements.txt
```

This installs three packages:
- `groq` — connects to the Groq AI API
- `gradio` — builds the web UI
- `python-dotenv` — reads your API key from the `.env` file

---

### Step 4 — Get a Groq API key

The app uses Groq to run the AI. You need a free API key.

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Click **API Keys** in the left sidebar
4. Click **Create API Key**
5. Copy the key — you only see it once

---

### Step 5 — Create your `.env` file

In your project folder, create a file called `.env` (just that, no other name).

Inside it, paste this:
```
GROQ_API_KEY1=paste_your_key_here
```

Replace `paste_your_key_here` with the actual key you copied. Save the file.

Your `.env` file should look like this:
```
GROQ_API_KEY1=gsk_abc123yourrealkeyhere
```

> ⚠️ Never share this file or upload it to GitHub. The `.gitignore` file already makes sure Git ignores it.

---

### Step 6 — Run the app

```bash
python app.py
```

You will see something like:
```
Running on local URL: http://127.0.0.1:7860
```

Open that link in your browser. The app is running.

---

## How to use the app

1. Open `http://127.0.0.1:7860` in your browser
2. Pick a mode from the **Mode** dropdown at the top
3. Optionally adjust the **Temperature** slider
4. Type your message in the text box at the bottom
5. Press **Enter** or click **Send**
6. Watch the response stream in word by word
7. To start a new conversation, click **Clear Chat**
8. To switch modes, just pick a different one from the dropdown — the chat clears automatically

---

## Common errors and fixes

**`GROQ_API_KEY1 is not set`**
You forgot to create the `.env` file or the key name is wrong. Make sure the file exists and the variable is named exactly `GROQ_API_KEY1`.

**`Access denied. Please check your network settings`**
Groq is blocking your network. Try switching to mobile hotspot or use a VPN set to a US/EU server.

**`ModuleNotFoundError`**
You forgot to activate the virtual environment or run `pip install -r requirements.txt`. Do both.

**App opens but nothing happens when I send a message**
Check the terminal where you ran `python app.py` — the actual error will be printed there.

---

## Screenshots

### Technical Explainer
![Technical Explainer](screenshots/technical_explainer.png)

### Debate Coach
![Debate Coach](screenshots/debate_coach.png)

### Code Reviewer
![Code Reviewer](screenshots/code_reviewer.png)

### Creative Writer
![Creative Writer](screenshots/creative_writer.png)

---

## Tech used

| Tool | What it does |
|---|---|
| Python | Programming language the app is written in |
| Gradio | Builds the web interface without needing HTML/CSS |
| Groq | Runs the AI model (llama-3.3-70b-versatile) |
| python-dotenv | Loads the API key from the `.env` file safely |
