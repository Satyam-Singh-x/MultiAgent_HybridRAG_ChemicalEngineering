"""
Query Analyzer Agent
--------------------

This agent performs:

1. Domain classification
2. Keyword extraction
3. Query refinement
4. Query type detection

Output: Structured dictionary used by downstream retrieval system.
"""

import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -------- CONFIG --------
OLLAMA_MODEL = "qwen2.5:latest"
# ------------------------


# ==========================================================
# ---------------- LOAD LLM -------------------------------
# ==========================================================

def load_llm():
    return OllamaLLM(
        model=OLLAMA_MODEL,
        base_url="http://127.0.0.1:11434",
        temperature=0.1,   # Lower temperature for structured output
        num_predict=512,
    )


# ==========================================================
# ---------------- QUERY ANALYSIS --------------------------
# ==========================================================

def build_query_analyzer():
    llm = load_llm()
    prompt = PromptTemplate.from_template(
        """
    You are a Chemical Engineering Query Analysis Agent.

    Analyze the user query and provide structured output.

    Tasks:
    1. Classify the domain:
       - chemical_reactions
       - equipment_basics
       - msds
       - process_safety
       - unit_operations

    2. Identify query type:
       - conceptual
       - method_procedure
       - equation_based
       - safety_related

    3. Extract key technical keywords.

    4. Rewrite the query to optimize retrieval clarity.

    Respond ONLY in valid JSON format:

    {{
      "original_query": "...",
      "domain": "...",
      "query_type": "...",
      "keywords": ["...", "..."],
      "refined_query": "..."
    }}

    User Query:
    {query}
    """
    )

    chain = prompt | llm | StrOutputParser()

    return chain


# ==========================================================
# ---------------- PUBLIC FUNCTION -------------------------
# ==========================================================

def analyze_query(query):
    chain = build_query_analyzer()
    response = chain.invoke({"query": query})

    try:
        structured_output = json.loads(response)
    except:
        structured_output = {
            "original_query": query,
            "domain": "unknown",
            "query_type": "conceptual",
            "keywords": [],
            "refined_query": query
        }

    return structured_output


# ==========================================================
# ---------------- TESTING MODE ----------------------------
# ==========================================================

if __name__ == "__main__":
    while True:
        user_query = input("\nEnter query (or type 'exit'): ")
        if user_query.lower() == "exit":
            break

        result = analyze_query(user_query)

        print("\n--- Query Analysis Output ---")
        print(json.dumps(result, indent=4))
