from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

def run_indexing_pipeline():
   
    pdf_name = "PharmaChain_Latex_DB.pdf"
    
    if not os.path.exists(pdf_name):
        print(f"❌ Error: '{pdf_name}' not found in the current directory.")
        print("Please copy your PDF file into this folder before running.")
        return

    loader = PyPDFLoader(pdf_name)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    persist_dir = "./chroma_store"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name="task3_documents"
    )

    print(f"\n✅ Indexed {len(chunks)} chunks into ChromaDB (persistent directory: {persist_dir})\n")

    for i in range(min(3, len(chunks))):
        page_num = chunks[i].metadata.get("page", 0) + 1
        
        content_preview = chunks[i].page_content.strip().replace("\n", " ")
        if len(content_preview) > 150:
            content_preview = content_preview[:150] + "..."

        print(f"Sample chunk {i+1} (page {page_num}):")
        print(f"[{content_preview}]\n")

if __name__ == "__main__":
    run_indexing_pipeline()