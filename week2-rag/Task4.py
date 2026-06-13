import os
import gradio as gr
from dotenv import load_dotenv

# Load environment variables securely
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

# ---------------------------------------------------------
# 1. API & Setup
# ---------------------------------------------------------
if not os.environ.get("GROQ_API_KEY1"):
    print("⚠️ WARNING: GROQ_API_KEY1 not found in environment variables.")

client = Groq(api_key=os.environ.get("GROQ_API_KEY1"))

PERSIST_DIR = "./chroma_task4"
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Load existing database if it exists
if os.path.exists(PERSIST_DIR):
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
        collection_name="rag_collection"
    )
    db_initialized = True
else:
    vectorstore = None
    db_initialized = False

# ---------------------------------------------------------
# 2. Backend Functions
# ---------------------------------------------------------
def process_documents(file_paths):
    """Takes uploaded PDFs, chunks them, and stores them in ChromaDB."""
    global vectorstore, db_initialized
    
    if not file_paths:
        return "No files uploaded.", db_initialized

    all_chunks = []
    total_docs = len(file_paths)
    
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        
        filename = os.path.basename(file_path)
        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1
            
        all_chunks.extend(chunks)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR,
        collection_name="rag_collection"
    )
    db_initialized = True
    
    status_msg = f"✅ Indexed {total_docs} document(s) – {len(all_chunks)} total chunks."
    return status_msg, db_initialized

def handle_query(user_question, chat_history):
    """Retrieves context, calls Groq LLM, and formats the response."""
    global vectorstore
    
    # Ensure chat_history is a valid list
    if chat_history is None:
        chat_history = []
    
    if not vectorstore:
        error_msg = "⚠️ Please upload and index a document first!"
        chat_history.append({"role": "user", "content": user_question})
        chat_history.append({"role": "assistant", "content": error_msg})
        return "", chat_history, "No context available."

    # Retrieve context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(user_question)

    context_text = ""
    display_context = ""
    
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        text = doc.page_content.replace('\n', ' ')
        
        context_text += f"[Source: {source}, page {page}]\n{text}\n\n"
        display_context += f"**Chunk {i} ({source}, Page {page})**\n{text}\n\n---\n\n"

    # Build prompt
    system_prompt = (
        "You are a precise, data-driven assistant.\n"
        "Answer the user's question using ONLY the provided context.\n"
        "If the answer is not in the context, you MUST respond exactly with: "
        "'I don't have that information in the uploaded documents.'\n"
        "If you find the answer, you must cite the source at the end of each factual statement "
        "using the exact format provided in the context blocks (e.g., [Source: filename.pdf, page X])."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_question}"}
    ]

    # Generate Response
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0,
            max_tokens=1024,
        )
        # CRITICAL FIX: The index is strictly required here.
        print(response)
        #ai_response = response.choices.message.content
        ai_response = response.choices[0].message.content
    except Exception as e:
        ai_response = f"Groq API Error: {str(e)}"

    # CRITICAL FIX: Gradio 5+ strictly requires the Dictionary format
    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "assistant", "content": ai_response})

    return "", chat_history, display_context

# ---------------------------------------------------------
# 3. Gradio UI
# ---------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 Grounded RAG Document Q&A")
    gr.Markdown("Upload PDFs, index them into ChromaDB, and ask questions with strict anti-hallucination checks.")
    
    is_ready = gr.State(value=db_initialized)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload & Index")
            file_upload = gr.File(label="Upload PDFs", file_count="multiple", type="filepath")
            index_btn = gr.Button("⚙️ Index Documents", variant="primary")
            status_label = gr.Label(
                value="Database Ready" if db_initialized else "Waiting for documents...", 
                label="Status"
            )
            
            with gr.Accordion("🔍 Retrieved Context (Developer Tool)", open=False):
                context_display = gr.Markdown(value="Context will appear here after asking a question.")
                
        with gr.Column(scale=2):
            gr.Markdown("### 2. Query the Documents")
            chatbot = gr.Chatbot(height=500)
            
            with gr.Row():
                msg_input = gr.Textbox(
                    scale=8,
                    show_label=False,
                    placeholder="Ask a question about your documents...",
                )
                submit_btn = gr.Button("Send", scale=1)

    # UI Event Wiring
    index_btn.click(
        fn=process_documents,
        inputs=[file_upload],
        outputs=[status_label, is_ready]
    )
    
    msg_input.submit(
        fn=handle_query,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, context_display]
    )
    submit_btn.click(
        fn=handle_query,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, context_display]
    )

if __name__ == "__main__":
    demo.launch()