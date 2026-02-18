"""
Validator Agent
---------------

Responsibilities:
1. Evaluate retrieval relevance
2. Check contextual sufficiency
3. Detect potential grounding gaps
4. Decide whether generation should proceed

Outputs structured validation result.
"""

import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================================
# ---------------- CONFIG ----------------------------------
# ==========================================================

OLLAMA_MODEL = "qwen2.5:latest"


# ==========================================================
# ---------------- LOAD LLM --------------------------------
# ==========================================================

def load_llm():
    return OllamaLLM(
        model=OLLAMA_MODEL,
        base_url="http://127.0.0.1:11434",
        temperature=0.0,   # Deterministic validation
        num_predict=512,
    )


# ==========================================================
# ---------------- VALIDATION CHAIN -------------------------
# ==========================================================

def build_validator_chain():
    llm = load_llm()

    prompt = PromptTemplate.from_template(
        """
You are a Retrieval Validation Agent for a Chemical Engineering RAG system.

Your job is to evaluate whether the retrieved context is:

1. Relevant to the user query
2. Sufficient to generate a grounded answer
3. Explicitly supportive of the key concepts in the query

You must NOT generate the final answer.
You must only evaluate retrieval quality.

Evaluation Inputs:

Original Query:
{original_query}

Refined Query:
{refined_query}

Domain:
{domain}

Query Type:
{query_type}

Retrieved Context:
{context}

Evaluate carefully and respond ONLY in JSON format:

{{
  "relevance_score": 0-10,
  "sufficiency_score": 0-10,
  "contains_explicit_method_reference": true/false,
  "validation_decision": "APPROVED" or "INSUFFICIENT_CONTEXT" or "IRRELEVANT_RETRIEVAL",
  "reasoning": "brief explanation"
}}
"""
    )

    return prompt | llm | StrOutputParser()


# ==========================================================
# ---------------- PUBLIC FUNCTION --------------------------
# ==========================================================

def validate_retrieval(retrieval_package):
    chain = build_validator_chain()

    response = chain.invoke({
        "original_query": retrieval_package["original_query"],
        "refined_query": retrieval_package["refined_query"],
        "domain": retrieval_package["domain"],
        "query_type": retrieval_package["query_type"],
        "context": retrieval_package["formatted_context"]
    })

    try:
        structured = json.loads(response)
    except:
        structured = {
            "relevance_score": 0,
            "sufficiency_score": 0,
            "contains_explicit_method_reference": False,
            "validation_decision": "INSUFFICIENT_CONTEXT",
            "reasoning": "Validation parsing failed."
        }

    return structured


# ==========================================================
# ---------------- TESTING MODE -----------------------------
# ==========================================================

if __name__ == "__main__":
    from query_analyzer_agent import analyze_query
    from hybrid_rag_agent import HybridRetrievalAgent

    analyzer = analyze_query
    retriever = HybridRetrievalAgent()

    while True:
        user_query = input("\nEnter query (or type 'exit'): ")
        if user_query.lower() == "exit":
            break

        structured_query = analyzer(user_query)
        retrieval_package = retriever.retrieve(structured_query)

        validation_result = validate_retrieval(retrieval_package)

        print("\n--- Validation Result ---")
        print(json.dumps(validation_result, indent=4))
