"""
Hybrid Retrieval Agent
----------------------

Responsibilities:
1. Perform dense semantic retrieval (FAISS)
2. Perform keyword-based retrieval (BM25)
3. Merge and deduplicate results
4. Format structured context
5. Return retrieval metadata for evaluation

This agent does NOT generate answers.
"""

import os
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================================
# ---------------------- CONFIG ----------------------------
# ==========================================================

DATA_DIR = r"D:\AdvancedML\MultiAgent_HybridRAG_ChemicalEngineering\data"
VECTOR_DB_DIR = r"D:\AdvancedML\MultiAgent_HybridRAG_ChemicalEngineering\embeddings\vectorstore"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

TOP_K_DENSE = 4
TOP_K_BM25 = 4


# ==========================================================
# ---------------- DENSE RETRIEVER -------------------------
# ==========================================================

def load_dense_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_DENSE})


# ==========================================================
# ---------------- BM25 SETUP ------------------------------
# ==========================================================

def load_and_chunk_documents():
    documents = []

    for topic in os.listdir(DATA_DIR):
        topic_path = os.path.join(DATA_DIR, topic)

        if not os.path.isdir(topic_path):
            continue

        for file in os.listdir(topic_path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(topic_path, file))
                pages = loader.load()

                for page in pages:
                    page.metadata["topic"] = topic
                    page.metadata["source_file"] = file
                    page.metadata["page_number"] = page.metadata.get("page", None)

                documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(documents)


def build_bm25_index(chunks):
    corpus = [chunk.page_content for chunk in chunks]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks


def retrieve_bm25(query, bm25, chunks):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:TOP_K_BM25]

    return [chunks[i] for i in top_indices]


# ==========================================================
# ---------------- MERGE & FORMAT --------------------------
# ==========================================================

def merge_results(dense_docs, bm25_docs):
    combined = dense_docs + bm25_docs

    unique = {}
    for doc in combined:
        key = (
            doc.metadata.get("source_file"),
            doc.metadata.get("page_number"),
            doc.page_content[:50]
        )
        unique[key] = doc

    return list(unique.values())


def format_context(docs):
    formatted = []

    for doc in docs:
        block = (
            f"[Source: {doc.metadata.get('source_file')} | "
            f"Page: {doc.metadata.get('page_number')} | "
            f"Topic: {doc.metadata.get('topic')}]\n"
            f"{doc.page_content}"
        )
        formatted.append(block)

    return "\n\n".join(formatted)


# ==========================================================
# ---------------- PUBLIC HYBRID FUNCTION ------------------
# ==========================================================

class HybridRetrievalAgent:

    def __init__(self):
        print("Initializing Hybrid Retrieval Agent...")

        self.dense_retriever = load_dense_retriever()
        self.bm25_chunks = load_and_chunk_documents()
        self.bm25, self.bm25_chunks = build_bm25_index(self.bm25_chunks)

    def retrieve(self, structured_query):
        """
        structured_query: output from Query Analyzer Agent
        """

        refined_query = structured_query["refined_query"]

        dense_docs = self.dense_retriever.invoke(refined_query)
        bm25_docs = retrieve_bm25(refined_query, self.bm25, self.bm25_chunks)

        merged_docs = merge_results(dense_docs, bm25_docs)
        formatted_context = format_context(merged_docs)

        return {
            "original_query": structured_query["original_query"],
            "refined_query": refined_query,
            "domain": structured_query["domain"],
            "query_type": structured_query["query_type"],
            "retrieved_documents": merged_docs,
            "formatted_context": formatted_context
        }


# ==========================================================
# ---------------- TESTING MODE ----------------------------
# ==========================================================

if __name__ == "__main__":
    from query_analyzer_agent import analyze_query

    agent = HybridRetrievalAgent()

    while True:
        user_query = input("\nEnter query (or type 'exit'): ")
        if user_query.lower() == "exit":
            break

        structured_query = analyze_query(user_query)
        result = agent.retrieve(structured_query)

        print("\n--- Formatted Context ---\n")
        print(result["formatted_context"])
