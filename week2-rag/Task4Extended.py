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

# Load existing database and extract existing document filenames for the dropdown
existing_sources = ["All Documents"]
if os.path.exists(PERSIST_DIR):
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
        collection_name="rag_collection"
    )
    db_initialized = True
    
    # Extract unique filenames currently stored in the DB
    try:
        db_data = vectorstore.get()
        if db_data and "metadatas" in db_data and db_data["metadatas"]:
            sources = set(meta.get("source") for meta in db_data["metadatas"] if meta and "source" in meta)
            existing_sources.extend(list(sources))
    except Exception as e:
        print(f"Could not load existing sources: {e}")
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
        return "No files uploaded.", db_initialized, gr.update()

    all_chunks = []
    total_docs = len(file_paths)
    new_sources = set()
    
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        
        filename = os.path.basename(file_path)
        new_sources.add(filename)
        
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
    
    # Combine old sources with new sources for the dropdown
    global existing_sources
    for source in new_sources:
        if source not in existing_sources:
            existing_sources.append(source)
            
    status_msg = f"✅ Indexed {total_docs} document(s) – {len(all_chunks)} total chunks."
    
    # Return status, state, and an updated Dropdown UI component
    return status_msg, db_initialized, gr.update(choices=existing_sources, value="All Documents")

def handle_query(user_question, chat_history, selected_doc):
    """Retrieves context, calls Groq LLM, and formats the response."""
    global vectorstore
    
    if chat_history is None:
        chat_history = []
    
    if not vectorstore:
        error_msg = "⚠️ Please upload and index a document first!"
        chat_history.append({"role": "user", "content": user_question})
        chat_history.append({"role": "assistant", "content": error_msg})
        return "", chat_history, "No context available."

    # --- THE BONUS CHALLENGE LOGIC ---
    # Configure the search based on the dropdown selection
    search_kwargs = {"k": 4}
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
        display_context += f"**Chunk {i} ({source}, Page {page})**\n{text}\n\n---\n\n"

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

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0,
            max_tokens=1024,
        )
        ai_response = response.choices[0].message.content
    except Exception as e:
        ai_response = f"Groq API Error: {str(e)}"

    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "assistant", "content": ai_response})

    return "", chat_history, display_context

# ---------------------------------------------------------
# 3. Gradio UI
# ---------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 Grounded RAG Document Q&A (With Filtering)")
    gr.Markdown("Upload PDFs, filter by specific document, and ask questions with strict anti-hallucination checks.")
    
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
            
            # --- THE NEW DROPDOWN COMPONENT ---
            doc_filter = gr.Dropdown(
                choices=existing_sources, 
                value="All Documents", 
                label="🎯 Filter Search to Specific Document"
            )
            
            chatbot = gr.Chatbot(height=450)
            
            with gr.Row():
                msg_input = gr.Textbox(
                    scale=8,
                    show_label=False,
                    placeholder="Ask a question...",
                )
                submit_btn = gr.Button("Send", scale=1)

    # UI Event Wiring
    index_btn.click(
        fn=process_documents,
        inputs=[file_upload],
        # Outputs update the Dropdown now too!
        outputs=[status_label, is_ready, doc_filter]
    )
    
    msg_input.submit(
        fn=handle_query,
        # Inputs now include the Dropdown state!
        inputs=[msg_input, chatbot, doc_filter],
        outputs=[msg_input, chatbot, context_display]
    )
    submit_btn.click(
        fn=handle_query,
        # Inputs now include the Dropdown state!
        inputs=[msg_input, chatbot, doc_filter],
        outputs=[msg_input, chatbot, context_display]
    )

if __name__ == "__main__":
    demo.launch()