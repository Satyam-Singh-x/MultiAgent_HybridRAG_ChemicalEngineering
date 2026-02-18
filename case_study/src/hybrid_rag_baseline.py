"""
Hybrid RAG System
-----------------
This module implements a Hybrid Retrieval-Augmented Generation (RAG) system
by combining:

1. Dense semantic retrieval (FAISS + embeddings)
2. Keyword-based retrieval (BM25)

The retrieved results from both methods are merged, deduplicated,
and passed to the LLM for grounded answer generation.
"""

import os
import time
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================================
# ---------------------- CONFIGURATION ---------------------
# ==========================================================

"""
Central configuration block.
Change values here without modifying core logic.
"""

DATA_DIR = r"D:\AdvancedML\MultiAgent_HybridRAG_ChemicalEngineering\data"
VECTOR_DB_DIR = r"D:\AdvancedML\MultiAgent_HybridRAG_ChemicalEngineering\embeddings\vectorstore"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

TOP_K_DENSE = 4
TOP_K_BM25 = 4

OLLAMA_MODEL = "qwen2.5:latest"
DEBUG_PRINT = True


# ==========================================================
# --------------------- LOAD LLM ---------------------------
# ==========================================================

"""
Loads the local Ollama LLM.
Low temperature ensures more deterministic and grounded responses.
"""

def load_llm():
    return OllamaLLM(
        model=OLLAMA_MODEL,
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
        num_predict=512,
    )


# ==========================================================
# ----------------- DENSE RETRIEVAL (FAISS) ----------------
# ==========================================================

"""
Loads pre-built FAISS vector database and returns
a semantic retriever based on embeddings.
"""

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
# ---------------- BM25 KEYWORD RETRIEVAL ------------------
# ==========================================================

"""
Loads documents again and builds a BM25 index.
BM25 works purely on lexical overlap (keyword matching).
"""

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
# ---------------- MERGING RETRIEVAL RESULTS ---------------
# ==========================================================

"""
Combines dense and BM25 results and removes duplicates.
This ensures diverse yet non-redundant context.
"""

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


# ==========================================================
# ----------------- CONTEXT FORMATTING ---------------------
# ==========================================================

"""
Formats retrieved chunks into structured context
with citation metadata for transparency.
"""

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source_file')} | "
        f"Page: {doc.metadata.get('page_number', 'N/A')} | "
        f"Topic: {doc.metadata.get('topic')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


# ==========================================================
# ----------------- MAIN QA PIPELINE -----------------------
# ==========================================================

"""
Full Hybrid RAG pipeline:
1. Dense retrieval
2. BM25 retrieval
3. Merge + deduplicate
4. Generate grounded response
5. Measure latency
"""

def ask_question(query, dense_retriever, bm25, bm25_chunks, llm):
    start_time = time.time()

    dense_docs = dense_retriever.invoke(query)
    bm25_docs = retrieve_bm25(query, bm25, bm25_chunks)

    merged_docs = merge_results(dense_docs, bm25_docs)

    if DEBUG_PRINT:
        print("\n--- Dense Retrieved ---")
        for doc in dense_docs:
            print(doc.metadata)

        print("\n--- BM25 Retrieved ---")
        for doc in bm25_docs:
            print(doc.metadata)

        print("\n--- Final Merged Context ---")
        for doc in merged_docs:
            print(doc.metadata)

    context = format_docs(merged_docs)

    prompt = PromptTemplate.from_template(
        """
You are a Chemical Engineering Knowledge Generator Agent operating 
inside a strictly grounded multi-agent RAG system.

Your task is to generate a fully grounded explanation using ONLY 
the provided validated context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT GROUNDING REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. You MUST use ONLY information explicitly present in the provided context.
2. You MUST NOT introduce external knowledge.
3. You MUST NOT infer missing equations, steps, or assumptions.
4. If information is incomplete, explicitly state:
   "The provided context does not contain sufficient detail on this aspect."
5. Every major paragraph MUST end with a citation.
6. The citation MUST exactly follow this format:

   (Source: filename, Page: page_number)

   Example:
   (Source: chemical process equipment.pdf, Page: 425)

7. Do NOT combine multiple sources inside one citation.
8. Do NOT invent filenames or page numbers.
9. Do NOT place all citations at the end — citations must appear 
   immediately after the paragraph they support.
10. If no valid citation can be attached to a paragraph, 
    DO NOT generate that paragraph.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Start with a concise conceptual overview paragraph.
• Follow with structured explanation using clear sections or bullet points.
• Each paragraph must end with one citation in the exact format:

  (Source: filename, Page: page_number)

• Keep explanation suitable for undergraduate chemical engineering students.
• Prefer clarity over excessive complexity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query:
{query}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATED CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate a fully grounded, citation-backed explanation now.

"""
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": query
    })

    latency = time.time() - start_time

    return answer, latency

# ==========================================================
# --------------- PUBLIC EVALUATION FUNCTION ---------------
# ==========================================================

def run_hybrid_baseline(query: str):
    """
    Public evaluation wrapper for Hybrid RAG baseline.

    Returns:
    --------
    dict containing:
        - final_answer
        - retrieval_count
        - formatted_context
        - validation_result (None for baseline systems)
    """

    # Load components
    dense_retriever = load_dense_retriever()
    bm25_chunks = load_and_chunk_documents()
    bm25, bm25_chunks = build_bm25_index(bm25_chunks)
    llm = load_llm()

    # Hybrid Retrieval
    dense_docs = dense_retriever.invoke(query)
    bm25_docs = retrieve_bm25(query, bm25, bm25_chunks)

    merged_docs = merge_results(dense_docs, bm25_docs)
    retrieval_count = len(merged_docs)

    # Format context (IMPORTANT for similarity computation)
    context = format_docs(merged_docs)

    # Prompt
    prompt = PromptTemplate.from_template(
        """
You are a Chemical Engineering Knowledge Generator Agent.

STRICT RULES:
1. Use ONLY the provided context.
2. Do NOT introduce external knowledge.
3. Every major paragraph MUST end with a citation:
   (Source: filename, Page: page_number)

Query:
{query}

Context:
{context}
"""
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "query": query,
        "context": context
    })

    return {
        "final_answer": answer,
        "retrieval_count": retrieval_count,
        "formatted_context": context,
        "validation_result": None
    }

# ==========================================================
# ----------------------- MAIN LOOP ------------------------
# ==========================================================

if __name__ == "__main__":
    print("Loading Dense Retriever...")
    dense_retriever = load_dense_retriever()

    print("Building BM25 Index...")
    bm25_chunks = load_and_chunk_documents()
    bm25, bm25_chunks = build_bm25_index(bm25_chunks)

    llm = load_llm()

    while True:
        query = input("\nEnter your question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer, latency = ask_question(
            query,
            dense_retriever,
            bm25,
            bm25_chunks,
            llm
        )

        print("\nAnswer:\n")
        print(answer)
        print(f"\nLatency: {latency:.2f} seconds")
