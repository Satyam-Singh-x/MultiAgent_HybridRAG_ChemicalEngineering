import os
import time
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -------- CONFIG --------
DATA_DIR = r"D:\AdvancedML\MultiAgent_HybridRAG_ChemicalEngineering\data"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 6
OLLAMA_MODEL = "qwen2.5:latest"
DEBUG_PRINT = True  # Set to False to disable retrieved chunk printing
# ------------------------


# -------- Load LLM --------
def load_llm():
    return OllamaLLM(
        model=OLLAMA_MODEL,
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
        num_predict=512,
    )


# -------- Load and Chunk Documents --------
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

    chunks = splitter.split_documents(documents)

    return chunks


# -------- Build BM25 Index --------
def build_bm25_index(chunks):
    corpus = [chunk.page_content for chunk in chunks]

    # Lowercase tokenization for stability
    tokenized_corpus = [doc.lower().split() for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, chunks


# -------- Retrieve Using BM25 --------
def retrieve_bm25(query, bm25, chunks):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:TOP_K]

    top_chunks = [chunks[i] for i in top_indices]

    return top_chunks


# -------- Format Context for LLM --------
def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source_file')} | "
        f"Page: {doc.metadata.get('page_number', 'N/A')} | "
        f"Topic: {doc.metadata.get('topic')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


# -------- Ask Question --------
def ask_question(query, bm25, chunks, llm):
    start_time = time.time()

    retrieved_docs = retrieve_bm25(query, bm25, chunks)

    # 🔍 DEBUG PRINT RETRIEVED CHUNKS
    if DEBUG_PRINT:
        print("\n--- Retrieved Context (BM25) ---")
        for doc in retrieved_docs:
            print(doc.metadata)
            print(doc.page_content[:300])
            print("--------------------------------------------------")

    context = format_docs(retrieved_docs)

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
# PUBLIC EVALUATION FUNCTION
# ==========================================================

# ==========================================================
# PUBLIC EVALUATION FUNCTION
# ==========================================================

def run_keyword_baseline(query: str):
    """
    Executes BM25 keyword-based RAG baseline.

    Returns:
    --------
    dict containing:
        - final_answer
        - retrieval_count
        - formatted_context
        - validation_result (None for baseline)
    """

    # Load documents and build BM25
    chunks = load_and_chunk_documents()
    bm25, chunks = build_bm25_index(chunks)

    # Retrieve
    retrieved_docs = retrieve_bm25(query, bm25, chunks)
    retrieval_count = len(retrieved_docs)

    # Format context (IMPORTANT for evaluation similarity)
    context = format_docs(retrieved_docs)

    # Load LLM
    llm = load_llm()

    prompt = PromptTemplate.from_template(
        """
You are a Chemical Engineering Knowledge Generator Agent operating 
inside a strictly grounded multi-agent RAG system.

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

# -------- Main --------
if __name__ == "__main__":
    print("📄 Loading and chunking documents...")
    chunks = load_and_chunk_documents()

    print("📚 Building BM25 index...")
    bm25, chunks = build_bm25_index(chunks)

    llm = load_llm()

    while True:
        query = input("\nEnter your question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer, latency = ask_question(query, bm25, chunks, llm)

        print("\nAnswer:\n")
        print(answer)
        print(f"\nLatency: {latency:.2f} seconds")
