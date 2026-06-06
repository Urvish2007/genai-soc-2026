import os
import json
import gradio as gr
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY1")
if not API_KEY:
    raise EnvironmentError("GROQ_API_KEY1 is not set. Add it to your .env file.")

client = Groq(api_key=API_KEY)
MODEL = "llama-3.3-70b-versatile"

PERSONAS = {
    "Technical Explainer": {
        "system_prompt": (
            "You are a technical explainer. Explain concepts clearly and avoid unnecessary jargon. "
            "Use simple real-world examples. Be educational and beginner-friendly. "
            "Break down complex ideas into small digestible steps."
        ),
        "few_shot_examples": [
            {
                "role": "user",
                "content": "What is a database index?"
            },
            {
                "role": "assistant",
                "content": (
                    "A database index is like the index at the back of a textbook.\n\n"
                    "Without it: to find every mention of 'Python', you'd read every page.\n"
                    "With it: you jump straight to page 42, 78, 130.\n\n"
                    "In a database, an index stores a sorted copy of a column's values with "
                    "pointers to the actual rows. Lookups go from scanning millions of rows "
                    "to jumping directly to the right ones — much faster."
                )
            },
            {
                "role": "user",
                "content": "What is a race condition?"
            },
            {
                "role": "assistant",
                "content": (
                    "A race condition happens when two processes access shared data at the same time "
                    "and the final result depends on who finishes first.\n\n"
                    "Example: two people book the last concert ticket simultaneously. "
                    "Both see '1 ticket left', both click Buy. Without proper locking, "
                    "both succeed — now the venue is overbooked.\n\n"
                    "Fix: use locks or transactions so only one process can read-then-write at a time."
                )
            },
        ],
        "output_format": "text",
    },
    "Debate Coach": {
        "system_prompt": (
            "You are a debate coach. For any topic, present both sides fairly and with equal strength. "
            "Give the strongest arguments for each position. "
            "Never favour one side. End every response with a balanced conclusion."
        ),
        "few_shot_examples": [
            {
                "role": "user",
                "content": "Should everyone learn to code?"
            },
            {
                "role": "assistant",
                "content": (
                    "**FOR — Yes, everyone should learn to code:**\n"
                    "- Coding teaches logical thinking and structured problem-solving applicable everywhere.\n"
                    "- Digital literacy is increasingly essential; basic coding knowledge prevents exploitation.\n"
                    "- Opens high-paying career paths and lowers barriers to building ideas.\n\n"
                    "**AGAINST — No, it's not necessary for everyone:**\n"
                    "- Coding is a specialist skill, like surgery — most people don't need it daily.\n"
                    "- Time spent coding could be better used deepening domain expertise.\n"
                    "- No-code and AI tools increasingly remove the need for manual code.\n\n"
                    "**Conclusion:** Coding basics (logic, automation thinking) benefit everyone, "
                    "but deep programming skill should remain optional based on interest and career."
                )
            },
            {
                "role": "user",
                "content": "Is remote work better than office work?"
            },
            {
                "role": "assistant",
                "content": (
                    "**FOR — Remote work is better:**\n"
                    "- Eliminates commute time, returning hours to workers each week.\n"
                    "- Enables hiring from anywhere, widening talent pools.\n"
                    "- Studies show many knowledge workers are more productive at home.\n\n"
                    "**AGAINST — Office work is better:**\n"
                    "- Spontaneous collaboration and mentorship happen naturally in person.\n"
                    "- Many people struggle with isolation, blurred work-life boundaries, and distractions at home.\n"
                    "- Company culture and onboarding are harder to build remotely.\n\n"
                    "**Conclusion:** Neither is universally superior. Hybrid models that give workers "
                    "flexibility while preserving intentional in-person time tend to yield the best outcomes."
                )
            },
        ],
        "output_format": "text",
    },
    "Code Reviewer": {
        "system_prompt": (
            "You are a professional code reviewer. Review code thoroughly. "
            "Identify bugs, logic errors, security issues, and style problems. "
            "Suggest concrete improvements. Assign a severity level: low, medium, or high. "
            "You MUST respond with valid JSON only. No prose outside the JSON. "
            'Use this exact schema: {"issues": [{"line": <int>, "description": <str>}], '
            '"suggestions": [<str>], "severity": "low" | "medium" | "high"}'
        ),
        "few_shot_examples": [
            {
                "role": "user",
                "content": (
                    "Review this code:\n"
                    "def get_user(id):\n"
                    "    query = f\"SELECT * FROM users WHERE id = {id}\"\n"
                    "    return db.execute(query)"
                )
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "issues": [
                        {
                            "line": 2,
                            "description": "SQL injection vulnerability: user input is interpolated directly into the query string."
                        }
                    ],
                    "suggestions": [
                        "Use parameterised queries: db.execute('SELECT * FROM users WHERE id = ?', (id,))",
                        "Add input validation to ensure id is an integer before querying.",
                        "Avoid SELECT *; explicitly name the columns you need."
                    ],
                    "severity": "high"
                })
            },
            {
                "role": "user",
                "content": (
                    "Review this code:\n"
                    "def divide(a, b):\n"
                    "    return a / b\n"
                    "result = divide(10, 0)"
                )
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "issues": [
                        {
                            "line": 2,
                            "description": "No zero-division guard: divide(10, 0) raises ZeroDivisionError at runtime."
                        }
                    ],
                    "suggestions": [
                        "Add a guard: if b == 0: raise ValueError('Divisor cannot be zero')",
                        "Add type hints: def divide(a: float, b: float) -> float",
                        "Consider returning None or a default value for invalid inputs depending on context."
                    ],
                    "severity": "medium"
                })
            },
        ],
        "output_format": "json",
    },
    "Creative Writer": {
        "system_prompt": (
            "You are a creative writer. Write vivid, sensory-rich narratives. "
            "Use strong verbs, unexpected metaphors, and specific concrete details. "
            "Prioritise showing over telling. Every sentence should earn its place. "
            "Never be generic or clichéd."
        ),
        "few_shot_examples": [
            {
                "role": "user",
                "content": "Write a short scene: a programmer at 2am."
            },
            {
                "role": "assistant",
                "content": (
                    "The apartment held its breath. Only the mechanical click of keys broke the silence — "
                    "a sound like rain on a tin roof, arrhythmic and relentless.\n\n"
                    "Marcus hadn't moved in four hours. His coffee had grown a skin. On the monitor, "
                    "a single red error blinked like a slow heartbeat: `TypeError: cannot read property of undefined`.\n\n"
                    "He leaned back. The ceiling offered nothing. Outside, a bus groaned past, headlights "
                    "sweeping across the wall like a lighthouse nobody needed. He turned back to the screen, "
                    "cracked his knuckles, and started again from line one."
                )
            },
            {
                "role": "user",
                "content": "Describe a city just before a storm."
            },
            {
                "role": "assistant",
                "content": (
                    "The city went quiet in the wrong way — not the quiet of sleep, but the quiet of held breath.\n\n"
                    "Pigeons vanished from the ledges. A plastic bag cartwheeled across an empty intersection. "
                    "The air smelled of iron and wet stone, and the sky had turned the colour of a bruise, "
                    "that particular yellow-green that meteorologists name and everyone else just fears.\n\n"
                    "A woman on the corner looked up once, folded her newspaper under her arm, "
                    "and walked faster. The first drop hit the pavement like a full stop."
                )
            },
        ],
        "output_format": "text",
    },
}


def build_messages(persona_name: str, user_message: str) -> list[dict]:
    persona = PERSONAS[persona_name]
    msgs = [{"role": "system", "content": persona["system_prompt"]}]
    for shot in persona["few_shot_examples"]:
        msgs.append({"role": shot["role"], "content": shot["content"]})
    final_msg = user_message
    if persona["output_format"] == "json":
        final_msg = (
            user_message
            + "\n\nIMPORTANT: Respond with valid JSON only. "
            "No text before or after the JSON object."
        )
    msgs.append({"role": "user", "content": final_msg})
    return msgs


def stream_response(persona_name: str, user_message: str, temperature: float):
    msgs = build_messages(persona_name, user_message)
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=temperature,
            stream=True,
        )
        accumulated = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            accumulated += delta
            yield accumulated
    except Exception as e:
        yield f"⚠️ API error: {str(e)}"


def format_json_response(raw: str) -> str:
    try:
        data = json.loads(raw)
        md = ""
        issues = data.get("issues", [])
        suggestions = data.get("suggestions", [])
        severity = data.get("severity", "unknown")

        if issues:
            md += "### Issues\n"
            for issue in issues:
                line = issue.get("line", "?")
                desc = issue.get("description", "")
                md += f"- **Line {line}:** {desc}\n"
        else:
            md += "### Issues\n- None found.\n"

        md += "\n### Suggestions\n"
        if suggestions:
            for s in suggestions:
                md += f"- {s}\n"
        else:
            md += "- None.\n"

        sev_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(severity.lower(), "⚪")
        md += f"\n### Severity\n{sev_icon} **{severity.capitalize()}**"
        return md
    except (json.JSONDecodeError, AttributeError):
        return f"⚠️ JSON parsing failed.\n\nRaw response:\n\n```\n{raw}\n```"


def chat(user_message: str, history: list, persona_name: str, temperature: float):
    if not user_message.strip():
        yield history, ""
        return

    history = list(history or [])
    is_json_mode = PERSONAS[persona_name]["output_format"] == "json"
    raw_accumulated = ""

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": ""})

    for partial in stream_response(persona_name, user_message, temperature):
        raw_accumulated = partial
        history[-1]["content"] = partial
        yield history, ""

    if is_json_mode:
        history[-1]["content"] = format_json_response(raw_accumulated)
        yield history, ""


def update_system_prompt(persona_name: str) -> str:
    return PERSONAS[persona_name]["system_prompt"]


def clear_chat():
    return [], ""


with gr.Blocks(title="PromptForge") as demo:
    gr.Markdown("# PromptForge — Multi-Mode AI Assistant")

    with gr.Row():
        mode_dropdown = gr.Dropdown(
            choices=list(PERSONAS.keys()),
            value="Technical Explainer",
            label="Mode",
            scale=3,
        )
        temp_slider = gr.Slider(
            minimum=0.0,
            maximum=1.5,
            value=0.7,
            step=0.1,
            label="Temperature",
            scale=2,
        )

    with gr.Accordion("Active System Prompt", open=False):
        sys_prompt_box = gr.Textbox(
            value=PERSONAS["Technical Explainer"]["system_prompt"],
            lines=4,
            interactive=False,
            label="",
            show_label=False,
        )

    chatbot = gr.Chatbot(height=450)

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Ask anything...",
            show_label=False,
            scale=8,
            lines=1,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    clear_btn = gr.Button("Clear Chat", variant="secondary")

    mode_dropdown.change(
        fn=update_system_prompt,
        inputs=mode_dropdown,
        outputs=sys_prompt_box,
    )
    mode_dropdown.change(
        fn=clear_chat,
        outputs=[chatbot, user_input],
    )

    send_btn.click(
        fn=chat,
        inputs=[user_input, chatbot, mode_dropdown, temp_slider],
        outputs=[chatbot, user_input],
    )
    user_input.submit(
        fn=chat,
        inputs=[user_input, chatbot, mode_dropdown, temp_slider],
        outputs=[chatbot, user_input],
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, user_input],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

    