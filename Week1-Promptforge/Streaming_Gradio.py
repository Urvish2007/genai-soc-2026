import gradio as gr
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY1"))

def chat_stream(message, history):
    """Gradio chat function — yields accumulated text for streaming display."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    for human, bot in history:
        messages.append({"role": "user",      "content": human})
        messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": message})

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

gr.ChatInterface(
    fn=chat_stream,
    title="My First AI Chat App",
).launch()