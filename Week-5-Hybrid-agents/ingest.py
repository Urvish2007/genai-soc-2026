from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from chroma_client import client, CHROMA_DIR, COLLECTION_NAME

# One embedding model instance, reused for both indexing and retrieval.
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def index_pdf(filepath: str) -> int:
    """Reads a PDF, chunks it, and adds it to the shared ChromaDB collection.

    Uses the shared PersistentClient (see chroma_client.py) so this stays
    in sync with what tools_rag.py reads from — no duplicate/competing
    client instances on the same directory.
    """
    loader = PyPDFLoader(filepath)
    pages = loader.load()

    if not pages:
        raise ValueError("No extractable text found in this PDF (is it scanned/image-only?).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(pages)

    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    vectorstore.add_documents(chunks)

    return len(chunks)