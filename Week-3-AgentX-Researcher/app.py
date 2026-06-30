import uuid
import gradio as gr
from agent import run_agent_with_trace

# ── CORE CHAT LOGIC ──
def process_chat(user_message, chat_history, current_session_id):
    """Handles the user input, calls the agent, and updates the Gradio UI components."""
    if not user_message.strip():
        return chat_history, current_session_id, "No input provided."

    # Call the AgentX backend
    final_answer, reasoning_trace = run_agent_with_trace(user_message, current_session_id)
    
    # Update the visual chat history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": final_answer})
    
    return chat_history, current_session_id, reasoning_trace

# ── GRADIO UI DESIGN ──
custom_theme = gr.themes.Soft(
    primary_hue="indigo", 
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
)

# FIXED: Removed theme from Blocks constructor for Gradio 6.0
with gr.Blocks(title="AgentX — Autonomous Researcher") as demo:
    session_id = gr.State(value=lambda: str(uuid.uuid4()))

    gr.HTML("""
    <div style="text-align: center; margin-bottom: 25px; padding-top: 20px;">
        <h1 style="font-weight: 800; font-size: 2.8rem; margin-bottom: 5px; color: #1e1b4b;">🤖 AgentX</h1>
        <p style="font-size: 1.1rem; color: #4b5563;">Autonomous Research Agent • Live Web Search • Long-term Memory</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=7):
            # FIXED: Removed 'type' and 'show_copy_button' for Gradio 6.0
            chatbot = gr.Chatbot(
                height=600, 
                label="Conversation History"
            )
            
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask AgentX a complex research question...", 
                    show_label=False,
                    scale=5,
                    container=False
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=3):
            with gr.Accordion("🔍 Agent Reasoning Trace", open=True):
                gr.Markdown(
                    "*See exactly how the ReAct loop functions. Watch AgentX decide which tools to use and what data to search for.*"
                )
                trace_box = gr.Markdown(
                    value="*Awaiting first query...*",
                    label="Active Thought Process"
                )

    # ── EVENT WIRING ──
    msg_box.submit(
        fn=process_chat,
        inputs=[msg_box, chatbot, session_id],
        outputs=[chatbot, session_id, trace_box]
    ).then(
        fn=lambda: "", 
        outputs=[msg_box]
    )

    submit_btn.click(
        fn=process_chat,
        inputs=[msg_box, chatbot, session_id],
        outputs=[chatbot, session_id, trace_box]
    ).then(
        fn=lambda: "", 
        outputs=[msg_box]
    )

# ── LAUNCH ──
if __name__ == "__main__":
    print("🚀 Launching AgentX Researcher...")
    # FIXED: Added theme here for Gradio 6.0
    demo.launch(theme=custom_theme)