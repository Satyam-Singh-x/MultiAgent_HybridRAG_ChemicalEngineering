"""
Dense Baseline RAG System
--------------------------

Pure dense embedding retrieval using FAISS.
No validation layer.
Used for baseline comparison in evaluation.
"""

import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================================
# CONFIGURATION
# ==========================================================

VECTOR_DB_DIR = r"D:\AdvancedML\MultiAgent_HybridRAG_ChemicalEngineering\embeddings\vectorstore"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:latest"
TOP_K = 6


# ==========================================================
# LOAD LLM
# ==========================================================

def load_llm():
    return OllamaLLM(
        model=OLLAMA_MODEL,
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
        num_predict=512,
    )


# ==========================================================
# LOAD RETRIEVER
# ==========================================================

def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )


# ==========================================================
# FORMAT DOCUMENTS
# ==========================================================

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source_file')} | "
        f"Page: {doc.metadata.get('page_number', 'N/A')} | "
        f"Topic: {doc.metadata.get('topic')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


# ==========================================================
# GENERATOR PROMPT
# ==========================================================

def build_prompt_chain(llm):
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

    return prompt | llm | StrOutputParser()


# ==========================================================
# PUBLIC EVALUATION FUNCTION
# ==========================================================

def run_dense_baseline(query: str):
    """
    Executes dense embedding baseline RAG.

    Returns:
    --------
    dict containing:
        - final_answer
        - retrieval_count
        - formatted_context
        - validation_result (None for baseline)
    """

    retriever = load_retriever()
    llm = load_llm()

    # Retrieve documents
    docs = retriever.invoke(query)
    retrieval_count = len(docs)

    # Format context
    context = format_docs(docs)

    # Generate answer
    chain = build_prompt_chain(llm)

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
# INTERACTIVE MODE
# ==========================================================

if __name__ == "__main__":

    while True:
        user_query = input("\nEnter your question (or type 'exit'): ")
        if user_query.lower() == "exit":
            break

        result = run_dense_baseline(user_query)

        print("\nAnswer:\n")
        print(result["final_answer"])
        print(f"\nRetrieved Chunks: {result['retrieval_count']}")
