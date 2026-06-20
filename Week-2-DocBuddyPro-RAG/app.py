import os
import shutil
import gradio as gr
import gc      
import time    
from dotenv import load_dotenv

# Load environment variables securely
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # UPDATED IMPORT
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Initialize LangChain's Groq wrapper instead
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY1")
)

PERSIST_DIR = "./chroma_store"

# UPDATED EMBEDDING CALL
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Initialize global state variables
vectorstore = None
db_initialized = False
existing_sources = ["All Documents"]

# Persistence Check: Load existing DB if it exists
if os.path.exists(PERSIST_DIR):
    try:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model,
            collection_name="docbuddy_collection"
        )
        db_initialized = True
        
        # Populate Dropdown with existing documents
        db_data = vectorstore.get()
        if db_data and "metadatas" in db_data and db_data["metadatas"]:
            sources = set(meta.get("source") for meta in db_data["metadatas"] if meta and "source" in meta)
            existing_sources.extend(list(sources))
        print("✅ Successfully loaded existing Chroma database.")
    except Exception as e:
        print(f"⚠️ Error loading existing database: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. CORE BACKEND FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def index_documents(file_paths):
    """Processes PDFs, chunks them, stores in ChromaDB, and calculates analytics."""
    global vectorstore, db_initialized, existing_sources
    
    if not file_paths:
        return "⚠️ No files uploaded.", gr.update(), gr.update(value=[])

    all_chunks = []
    chunk_analytics = [] 
    
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        
        filename = os.path.basename(file_path)
        
        # Tag metadata
        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1
            
        all_chunks.extend(chunks)
        
        chunk_analytics.append([filename, len(chunks)])
        
        if filename not in existing_sources:
            existing_sources.append(filename)

    # Store in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR,
        collection_name="docbuddy_collection"
    )
    db_initialized = True
    
    total_docs = len(file_paths)
    total_chunks = len(all_chunks)
    print(f"Index Complete: {total_docs} docs, {total_chunks} chunks.")
    
    status_msg = f"✅ {total_docs} document(s) indexed — {total_chunks} total chunks."
    
    return status_msg, gr.update(choices=existing_sources, value="All Documents"), chunk_analytics


def ask(user_question, chat_history, selected_doc):
    """Retrieves context and streams the LLM response token-by-token."""
    global vectorstore
    
    if chat_history is None:
        chat_history = []
        
    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "assistant", "content": ""})

    # 🌟 NEW ADDITION: Smart Greeting Bypass
    greetings = ["hi", "hello", "hey", "greetings", "who are you", "how are you"]
    if user_question.strip().lower() in greetings:
        reply = "👋 Hello there! I am **DocBuddy Pro**, your AI research assistant. Make sure your documents are indexed on the left, then ask me anything about them!"
        chat_history[-1]["content"] = reply
        yield gr.update(value=""), chat_history, "No context needed for greetings."
        return
    # 🌟 END OF NEW ADDITION

    if not vectorstore:
        chat_history[-1]["content"] = "⚠️ Please upload and index a document first!"
        yield gr.update(value=""), chat_history, "No context available."
        return
        

    search_kwargs = {"k": 5}
    if selected_doc and selected_doc != "All Documents":
        search_kwargs["filter"] = {"source": selected_doc}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    retrieved_docs = retriever.invoke(user_question)

    context_text = ""
    display_context = ""
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        text = doc.page_content.replace('\n', ' ')
        
        context_text += f"[Source: {source}, page {page}]\n{text}\n\n"
        display_context += f"### 📄 Chunk {i} ({source}, Page {page})\n> {text}\n\n---\n"

    system_prompt = (
        "You are DocBuddy Pro, an elite, highly precise research assistant.\n"
        "Your PRIMARY DIRECTIVE is to answer the user's question using ONLY the provided context.\n"
        "If the answer is NOT explicitly contained in the context, you MUST reply exactly with:\n"
        "'I don't have that information.'\n"
        "DO NOT guess, DO NOT use outside knowledge, and DO NOT hallucinate.\n"
        "When you provide factual statements, you MUST append the exact citation format found "
        "in the context (e.g., [Source: document.pdf, page X])."
    )

    messages = [{"role": "system", "content": system_prompt}]
    
    recent_history = chat_history[-5:-1] 
    for msg in recent_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_question}"})

    try:
        # Using LangChain's clean streaming method
        response_stream = llm.stream(messages)
        
        for chunk in response_stream:
            # LangChain standardizes the output, so we just call .content
            if chunk.content:
                chat_history[-1]["content"] += chunk.content
                yield gr.update(value=""), chat_history, display_context
                
    except Exception as e:
        chat_history[-1]["content"] = f"❌ AI Error: {str(e)}"
        yield gr.update(value=""), chat_history, display_context

def delete_database():
    """Wipes the ChromaDB directory and resets global variables."""
    global vectorstore, db_initialized, existing_sources
    
    # 1. The Database Way: Tell Chroma to empty itself first
    if vectorstore is not None:
        try:
            vectorstore.delete_collection()
        except Exception:
            pass # Ignore if collection already doesn't exist
            
    # 2. Clear memory references
    vectorstore = None
    db_initialized = False
    existing_sources = ["All Documents"]
    
    # 3. THE WINDOWS FIX: Force Python to close lingering file handles
    gc.collect()
    time.sleep(0.5) # Give Windows half a second to release the locks
    
    # 4. Physically delete the directory if it exists
    if os.path.exists(PERSIST_DIR):
        try:
            shutil.rmtree(PERSIST_DIR) 
            print("🗑️ Database physically deleted from disk.")
            status_msg = "🗑️ Database wiped clean. Please upload new documents."
        except Exception as e:
            print(f"⚠️ Error deleting database folder: {e}")
            status_msg = f"⚠️ Error clearing database: {str(e)}"
    else:
        status_msg = "ℹ️ Database is already empty."

    return status_msg, gr.update(choices=["All Documents"], value="All Documents"), gr.update(value=None)
# ═══════════════════════════════════════════════════════════════════════════
# 3. GRADIO USER INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

custom_theme = gr.themes.Monochrome(
    primary_hue="indigo", 
    secondary_hue="blue",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
)

# FIXED: Removed 'theme' from gr.Blocks() constructor for Gradio 6.0
with gr.Blocks(title="DocBuddy Pro") as demo:
    gr.HTML("""
    <div style="text-align: center; max-width: 800px; margin: 0 auto; padding-bottom: 20px;">
        <h1 style="font-weight: 800; font-size: 2.5rem; margin-bottom: 0;">📚 DocBuddy Pro</h1>
        <p style="font-size: 1.1rem; color: #555;">Multi-Document Intelligence & Hallucination-Free Retrieval</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### ⚙️ Engine Bay")
            file_upload = gr.File(label="Upload Knowledge Base (PDFs)", file_count="multiple", file_types=[".pdf"])
            
            with gr.Row():
                index_btn = gr.Button("⚡ Index Documents", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear Database", variant="stop", scale=1) # ADD THIS BUTTON

            
            status_label = gr.Label(
                value="Database Ready" if db_initialized else "No documents indexed yet.", 
                label="System Status"
            )
            
            analytics_table = gr.Dataframe(
                headers=["Filename", "Total Chunks"],
                datatype=["str", "number"],
                label="📊 Data Analytics",
                interactive=False
            )
            
            with gr.Accordion("🔍 Retrieved Context (Engine View)", open=False):
                gr.Markdown("*This panel shows the exact data blocks the AI is reading to answer your question.*")
                context_display = gr.Markdown(value="Awaiting query...")
                
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Interrogation Room")
            
            doc_filter = gr.Dropdown(
                choices=existing_sources, 
                value="All Documents", 
                label="🎯 Focus Retrieval (Optional)",
                info="Force the AI to only read from a specific document."
            )
            
            # FIXED: Removed 'type="messages"' as it is not allowed/is default in Gradio 6.0
            chatbot = gr.Chatbot(height=550)
            
            with gr.Row():
                msg_input = gr.Textbox(
                    scale=8,
                    show_label=False,
                    placeholder="Ask your documents a question...",
                    container=False
                )
                submit_btn = gr.Button("Send", scale=1, variant="secondary")

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. EVENT WIRING
    # ═══════════════════════════════════════════════════════════════════════════
    index_btn.click(
        fn=index_documents,
        inputs=[file_upload],
        outputs=[status_label, doc_filter, analytics_table]
    )
    
    clear_btn.click(
        fn=delete_database,
        inputs=None,
        outputs=[status_label, doc_filter, analytics_table]
    )
    
    msg_input.submit(
        fn=ask,
        inputs=[msg_input, chatbot, doc_filter],
        outputs=[msg_input, chatbot, context_display]
    )
    submit_btn.click(
        fn=ask,
        inputs=[msg_input, chatbot, doc_filter],
        outputs=[msg_input, chatbot, context_display]
    )

if __name__ == "__main__":
    print("🚀 Launching DocBuddy Pro...")

    demo.launch(theme=custom_theme)