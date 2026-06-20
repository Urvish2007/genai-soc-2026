import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY1"))

# Reload the persisted vector store (No re-indexing here!)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_store",
    embedding_function=embedding_model,
    collection_name="pharmachain_docs", # Must match indexer.py
)

def ask_document(question: str) -> str:
    """Core RAG retrieval and generation function"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    raw_chunks = retriever.invoke(question)

    # Assemble context block
    context = "\n\n".join([
        f"[Source {i}: {doc.metadata.get('source','?')}, page {doc.metadata.get('page','?')}]\n{doc.page_content}"
        for i, doc in enumerate(raw_chunks, 1)
    ])

    messages = [
        {"role": "system", "content": (
            "You are a helpful and precise document assistant. "
            "If the user says a basic greeting (like 'Hello', 'Hi', or 'Good morning'), respond politely and ask how you can help them with the uploaded PharmaChain documents. Do NOT use sources for greetings. "
            "For all other questions, answer using ONLY the context below. "
            "If the factual answer isn't there, say exactly: 'I don't have that information in the uploaded documents.' "
            "After your factual answer, add a 'Sources:' line citing the [Source N] labels."
        )},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0,   
    )
    return response.choices[0].message.content

def respond(message, history):
    """Bridge function for Gradio UI"""
    # 1. Get LLM Answer
    result = ask_document(message)
    
    # 2. Get same chunks to display in UI for debugging
    chunks = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(message)
    context_display = "\n\n".join([
        f"**[{doc.metadata.get('source','?')} · Page {doc.metadata.get('page','?')}]**\n{doc.page_content}"
        for doc in chunks
    ])
    
    # Gradio chat format requires appending to history
    # Gradio navu format: Dict objects
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result})
    
    return "", history, context_display

# --- Web UI (Gradio) ---
with gr.Blocks(title="PharmaChain RAG") as demo:
    gr.Markdown("## PharmaChain Database — Ask Your Documents")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=450)
            msg_box = gr.Textbox(placeholder="Ask a question about PharmaChain...", show_label=False)

        with gr.Column(scale=2):
            context_box = gr.Accordion("🔍 Retrieved Context (Developer Tool)", open=False)
            with context_box:
                context_display = gr.Markdown()

    # Trigger action when user presses Enter
    msg_box.submit(
        respond, 
        inputs=[msg_box, chatbot], 
        outputs=[msg_box, chatbot, context_display]
    )

if __name__ == "__main__":
    demo.launch()