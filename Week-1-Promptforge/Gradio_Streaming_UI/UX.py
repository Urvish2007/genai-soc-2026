# ============================================================
# Building UI with Gradio
# ============================================================

# gr.Interface()
# The simplest Gradio UI: give it a function, input type,
# and output type, and it generates a complete web app.
#
# Example:
# gr.Interface(
#     fn=my_fn,
#     inputs="text",
#     outputs="text"
# ).launch()
#
# One function → one UI.
# Good for simple demos but limited layout control.
# Best for Week 1 experiments and proofs of concept.


# gr.ChatInterface()
# A full chat UI with message history, streaming support,
# and retry functionality built in.
#
# Your function receives:
# (message, history)
#
# where history is a list of (user, bot) tuples.
#
# Works with Python generators for streaming.
# The quickest path to a production-feeling chat app.


# gr.Blocks()
# Full layout control:
# gr.Row(), gr.Column(), gr.Tab(), gr.Accordion()
#
# Components:
# gr.Dropdown(), gr.Slider(), gr.State()
#
# Event handlers:
# .change(), .click(), .submit()
#
# Use gr.Blocks() when your app needs more than one
# input or output component. This is commonly used
# for larger projects and custom layouts.


# ============================================================
# Running & Sharing
# ============================================================

# .launch()
# Starts a local server at:
# http://localhost:7860
#
# .launch(share=True)
# Generates a public URL (typically valid for ~72 hours)
# that can be shared with anyone.
#
# Use share=True for demos during SoC.
# For permanent hosting, use Hugging Face Spaces.


import gradio as gr
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY1")
)

def chat_stream(message, history):
    """Gradio chat function — yields accumulated text for streaming display."""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
    ]

    # Include conversation history
    for human, bot in history:
        messages.append({
            "role": "user",
            "content": human
        })
        messages.append({
            "role": "assistant",
            "content": bot
        })

    messages.append({
        "role": "user",
        "content": message
    })

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        stream=True,
    )

    accumulated = ""

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        accumulated += delta
        yield accumulated


# One-line chat UI with streaming built in
gr.ChatInterface(
    fn=chat_stream,
    title="My First AI Chat App",
).launch()