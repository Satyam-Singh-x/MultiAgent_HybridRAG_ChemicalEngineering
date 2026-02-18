import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -------- CONFIG --------
DATA_DIR = r"D:\AdvancedML\MultiAgent_HybridRAG_ChemicalEngineering\data"
VECTOR_DB_DIR = r"D:\AdvancedML\MultiAgent_HybridRAG_ChemicalEngineering\embeddings\vectorstore"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
# ------------------------


def load_documents():
    """
    Loads PDF documents from topic-wise folders.
    Adds metadata: topic, source file, page number.
    """
    documents = []

    for topic in os.listdir(DATA_DIR):
        topic_path = os.path.join(DATA_DIR, topic)

        if not os.path.isdir(topic_path):
            continue

        for file in os.listdir(topic_path):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(topic_path, file)
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()

                for page in pages:
                    page.metadata["topic"] = topic
                    page.metadata["source_file"] = file
                    page.metadata["page_number"] = page.metadata.get("page", None)

                documents.extend(pages)

    return documents


def chunk_documents(documents):
    """
    Splits documents into overlapping chunks.
    Adds unique chunk_id to each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


def create_vectorstore(chunks):
    """
    Creates FAISS vector store from chunks and saves locally.
    """
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_DIR)


if __name__ == "__main__":
    print("📄 Loading documents...")
    docs = load_documents()
    print(f"✅ Loaded {len(docs)} pages")

    print("✂️ Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")

    print("🧠 Creating vector store...")
    create_vectorstore(chunks)

    print("🚀 Ingestion complete. Vector store saved locally.")
