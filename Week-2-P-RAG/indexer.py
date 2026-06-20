from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # <-- Updated Line
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# ... baki no aagal no code e no e j ...

print("Starting Indexing Pipeline...")

# 1. Load the PDF
pdf_name = "PharmaChain_Latex_DB.pdf"
loader = PyPDFLoader(f"./{pdf_name}")
pages  = loader.load()
print(f"Loaded {len(pages)} pages from {pdf_name}")

# 2. Split into overlapping chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")

# 3. Tag every chunk with source + page metadata
for chunk in chunks:
    chunk.metadata["source"] = pdf_name
    chunk.metadata["page"]   = chunk.metadata.get("page", 0) + 1

# 4. Embed and store — locally
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}, 
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_store",  # Saved to disk
    collection_name="pharmachain_docs",  # Updated collection name
)

print(f"✅ Success! Indexed {vectorstore._collection.count()} chunks into ChromaDB.")