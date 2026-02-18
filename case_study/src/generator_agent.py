"""
Generator Agent
--------------------------------------------

Responsibilities:
1. Generate strictly grounded answers
2. Enforce inline citation usage
3. Respect validator decision
4. Prevent hallucination by restricting to retrieved context
5. Produce structured, citation-backed explanations
"""

import re
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
        temperature=0.1,   # Lower temperature for stricter grounding
        num_predict=900,
    )


# ==========================================================
# ---------------- GENERATOR PROMPT -------------------------
# ==========================================================

def build_generator_chain():
    llm = load_llm()

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

    return prompt | llm | StrOutputParser()


# ==========================================================
# ---------------- CITATION CHECK ---------------------------
# ==========================================================

def check_citation_presence(text):
    """
    Ensures at least one citation is present.
    Citation format example:
    (Source: chemical process equipment.pdf, Page: 425)
    """
    pattern = r"\(Source: .*?, Page: .*?\)"
    matches = re.findall(pattern, text)
    return len(matches) > 0


# ==========================================================
# ---------------- PUBLIC FUNCTION --------------------------
# ==========================================================

def generate_answer(retrieval_package, validation_result):

    # Step 1: Respect Validator
    if validation_result["validation_decision"] != "APPROVED":
        return (
            "Generation aborted due to insufficient contextual grounding.\n\n"
            f"Validator Reason: {validation_result['reasoning']}"
        )

    chain = build_generator_chain()

    response = chain.invoke({
        "query": retrieval_package["refined_query"],
        "context": retrieval_package["formatted_context"]
    })

    # Step 2: Citation Enforcement Check
    if not check_citation_presence(response):
        return (
            "Generation rejected due to missing required citations.\n\n"
            "The generated answer did not include proper grounding references."
        )

    return response


# ==========================================================
# ---------------- FULL PIPELINE TEST -----------------------
# ==========================================================

if __name__ == "__main__":
    from query_analyzer_agent import analyze_query
    from hybrid_rag_agent import HybridRetrievalAgent
    from validator_agent import validate_retrieval

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
        print(validation_result)

        answer = generate_answer(retrieval_package, validation_result)

        print("\n--- Final Generated Answer ---\n")
        print(answer)
