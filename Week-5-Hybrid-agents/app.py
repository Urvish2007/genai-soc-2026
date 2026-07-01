import uuid
import gradio as gr
from agent import agent, DEFAULT_CONFIG_EXTRA
from ingest import index_pdf
from image_utils import image_to_data_uri
from tools_vision import _describe_image_impl


def _resolve_filepath(file):
    """gr.File can hand back either a plain path string or an object with
    a .name attribute depending on Gradio version/config. Handle both so
    this doesn't crash on upload."""
    if file is None:
        return None
    if isinstance(file, str):
        return file
    return getattr(file, "name", None)


def handle_pdf_upload(file):
    """Indexes the uploaded PDF and returns a status message."""
    filepath = _resolve_filepath(file)
    if filepath is None:
        return "No file uploaded."
    try:
        n_chunks = index_pdf(filepath)
        return f"✅ Indexed {n_chunks} chunks into ChromaDB."
    except Exception as e:
        return f"❌ Failed to index PDF: {e}"


def chat(message, image, history, session_id):
    """Processes user input, handles images, and streams agent response."""
    original_message = message or ""
    augmented_message = original_message
    trace_log = []

    if image is not None:
        try:
            data_uri = image_to_data_uri(image)
            # Call the vision model directly instead of relying on the
            # agent's LLM to copy a huge base64 string into a tool-call
            # argument. This is faster, cheaper, and can't get corrupted
            # by the model truncating a long token sequence.
            description = _describe_image_impl(data_uri)
            trace_log.append(f"🛠️ **Tool Activated:** `describe_image`\n   *Input:* (uploaded image)")
            augmented_message = (
                f"{original_message}\n\n[Image description: {description}]"
                if original_message
                else f"[Image description: {description}]\n\nWhat's in this picture?"
            )
        except Exception as e:
            history.append({"role": "user", "content": original_message})
            history.append({"role": "assistant", "content": f"❌ Image Error: {e}"})
            return history, session_id, "Error processing image."

    if not augmented_message.strip():
        history.append({"role": "user", "content": original_message})
        history.append({"role": "assistant", "content": "Please type a question or upload an image."})
        return history, session_id, "🧠 No tools were called."

    config = {
        "configurable": {"thread_id": session_id},
        **DEFAULT_CONFIG_EXTRA,
    }

    final_answer = ""

    try:
        for event in agent.stream(
            {"messages": [{"role": "user", "content": augmented_message}]},
            config=config,
            stream_mode="values",
        ):
            last = event["messages"][-1]

            if getattr(last, "tool_calls", None):
                for tc in last.tool_calls:
                    trace_log.append(f"🛠️ **Tool Activated:** `{tc['name']}`\n   *Input:* {tc['args']}")

            elif getattr(last, "type", None) == "ai" and not getattr(last, "tool_calls", None):
                if last.content:
                    final_answer = last.content
    except Exception as e:
        final_answer = f"⚠️ System limit or error encountered: {str(e)}"

    if not final_answer:
        final_answer = "I wasn't able to produce an answer for that — please try rephrasing."

    history.append({"role": "user", "content": original_message or "(image uploaded)"})
    history.append({"role": "assistant", "content": final_answer})

    trace_output = "\n\n".join(trace_log) if trace_log else "🧠 No tools were called. Answered from internal knowledge."

    return history, session_id, trace_output


# GRADIO UI SETUP
custom_theme = gr.themes.Soft(primary_hue="indigo")

with gr.Blocks(title="HybridSight") as demo:
    session_id = gr.State(value=lambda: str(uuid.uuid4()))

    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px; padding-top: 10px;">
        <h1 style="font-weight: 800; font-size: 2.5rem; color: #1e1b4b;">👁️ HybridSight</h1>
        <p style="font-size: 1.1rem; color: #4b5563;">RAG + Web + Vision Agent</p>
    </div>
    """)

    with gr.Row():
        pdf_upload = gr.File(label="Upload a PDF", file_types=[".pdf"], scale=1)
        image_upload = gr.Image(label="Upload an image", type="filepath", scale=1)

    index_status = gr.Textbox(label="Indexing status", interactive=False)
    pdf_upload.change(handle_pdf_upload, inputs=pdf_upload, outputs=index_status)

    with gr.Row():
        with gr.Column(scale=7):
            # type="messages" is REQUIRED in Gradio 6 — the tuples format
            # was removed, and history is built as {"role", "content"} dicts.
            # Gradio 6.0 stable: "messages" (dict) format is the only format,
            # so `type=` was removed from the constructor entirely.
            chatbot = gr.Chatbot(height=500, label="Conversation")
            with gr.Row():
                msg_box = gr.Textbox(placeholder="Ask anything...", show_label=False, scale=5, container=False)
                submit_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=3):
            with gr.Accordion("🔍 Agent Reasoning Trace", open=True):
                trace_box = gr.Markdown(value="*Tools called will appear here...*", label="Active Thought Process")

    # WIRING EVENTS
    msg_box.submit(
        chat,
        inputs=[msg_box, image_upload, chatbot, session_id],
        outputs=[chatbot, session_id, trace_box],
    ).then(lambda: "", outputs=msg_box)

    submit_btn.click(
        chat,
        inputs=[msg_box, image_upload, chatbot, session_id],
        outputs=[chatbot, session_id, trace_box],
    ).then(lambda: "", outputs=msg_box)

if __name__ == "__main__":
    print("🚀 Launching HybridSight Agent...")
    demo.launch(theme=custom_theme)