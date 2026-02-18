"""
Evaluation Dataset Builder
====================================================

Purpose:
--------
Compare performance of:

1. dense_baseline
2. keyword_baseline
3. hybrid_baseline
4. multi_agent_system (agentic RAG)

Evaluation Dimensions:
----------------------
• Retrieval statistics
• Validator metrics
• Citation coverage ratio
• Citation hallucination severity
• Semantic grounding similarity
• Semantic hallucination severity
• Composite hallucination score
• Latency
• Answer length

Output:
-------
evaluation_results.csv
"""

import time
import re
import csv
import numpy as np
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# IMPORT SYSTEMS
# ================================

from dense_baseline import run_dense_baseline
from keyword_search_baseline import run_keyword_baseline
from hybrid_rag_baseline import run_hybrid_baseline
from final_multi_agent_rag_system import run_multi_agent_system


# ==========================================================
# ---------------- CONFIG ----------------------------------
# ==========================================================

OUTPUT_FILE = "evaluation_results.csv"

SYSTEM_TYPES = [
    "dense_baseline",
    "keyword_baseline",
    "hybrid_baseline",
    "multi_agent_system"
]

# Load embedding model ONCE globally
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================================
# ---------------- QUERY DATASET ----------------------------
# ==========================================================

EVALUATION_QUERIES = [
    "Explain McCabe–Thiele graphical method for binary distillation.",
    "What is constant molar overflow?",
    "Explain reflux ratio in distillation.",
    "Compare simple and fractional distillation.",
    "What is HETP?",
    "Explain minimum reflux condition.",
    "Derive operating line equation in distillation.",
    "Explain q-line in distillation.",
    "Compare packed and plate columns.",
    "Explain tray efficiency.",
    "Explain design steps of a distillation column.",
    "Discuss assumptions behind graphical methods in distillation.",
    "Explain distillation with heat balance considerations.",
    "Explain polymerase chain reaction.",
    "Describe photosynthesis.",
    "Explain neural networks in AI.",
    "What is blockchain?",
    "Describe quantum computing."
]


# ==========================================================
# ---------------- UTILITY FUNCTIONS ------------------------
# ==========================================================

def measure_latency(func, query: str):
    start = time.time()
    output = func(query)
    end = time.time()
    return output, round(end - start, 4)


# ==========================================================
# CITATION METRICS
# ==========================================================

def compute_citation_coverage(answer: str):
    citation_pattern = r"\(Source: .*?, Page: .*?\)"

    paragraphs = [
        p.strip()
        for p in answer.split("\n\n")
        if p.strip() and not p.strip().startswith("###")
    ]

    if not paragraphs:
        return 0.0

    cited_count = sum(
        1 for p in paragraphs if re.search(citation_pattern, p)
    )

    return round(cited_count / len(paragraphs), 3)


def classify_citation_severity(coverage: float):
    if coverage >= 0.7:
        return "Very Low"
    elif 0.4 <= coverage < 0.7:
        return "Medium"
    else:
        return "High"


# ==========================================================
# SEMANTIC GROUNDING METRICS
# ==========================================================

def compute_semantic_similarity(context: str, answer: str):
    if not context.strip() or not answer.strip():
        return 0.0

    context_embedding = embedding_model.encode([context])
    answer_embedding = embedding_model.encode([answer])

    similarity = cosine_similarity(
        context_embedding,
        answer_embedding
    )[0][0]

    return round(float(similarity), 3)


def classify_semantic_severity(similarity: float):
    if similarity >= 0.80:
        return "Very Low"
    elif 0.60 <= similarity < 0.80:
        return "Low"
    elif 0.40 <= similarity < 0.60:
        return "Medium"
    else:
        return "High"


# ==========================================================
# COMPOSITE HALLUCINATION SCORE
# ==========================================================

def compute_composite_score(citation_coverage: float, semantic_similarity: float):
    """
    Weighted hallucination score.

    Lower is better.

    Composite =
        0.4 * (1 - citation_coverage)
      + 0.6 * (1 - semantic_similarity)
    """

    score = (
        0.4 * (1 - citation_coverage)
        + 0.6 * (1 - semantic_similarity)
    )

    return round(score, 3)


# ==========================================================
# VALIDATION EXTRACTION
# ==========================================================

def extract_validation_info(system_output: Dict):
    validation = system_output.get("validation_result", None)

    if not validation:
        return None, None, None

    return (
        validation.get("relevance_score"),
        validation.get("sufficiency_score"),
        validation.get("validation_decision")
    )


# ==========================================================
# SYSTEM RUNNER
# ==========================================================

def run_system(system_type: str, query: str):

    if system_type == "dense_baseline":
        return run_dense_baseline(query)

    elif system_type == "keyword_baseline":
        return run_keyword_baseline(query)

    elif system_type == "hybrid_baseline":
        return run_hybrid_baseline(query)

    elif system_type == "multi_agent_system":
        return run_multi_agent_system(query)

    else:
        raise ValueError("Invalid system type")


# ==========================================================
# MAIN DATASET BUILDER
# ==========================================================

def build_evaluation_dataset():

    rows = []

    for query in EVALUATION_QUERIES:
        print(f"\nEvaluating Query: {query}")

        for system in SYSTEM_TYPES:
            print(f"  Running System: {system}")

            output, latency = measure_latency(
                lambda q: run_system(system, q),
                query
            )

            answer = output.get("final_answer", "")
            retrieval_count = output.get("retrieval_count", None)
            context = output.get("formatted_context", "")

            relevance_score, sufficiency_score, decision = extract_validation_info(output)

            # ---- Citation Metrics ----
            citation_coverage = compute_citation_coverage(answer)
            citation_severity = classify_citation_severity(citation_coverage)

            # ---- Semantic Metrics ----
            semantic_similarity = compute_semantic_similarity(context, answer)
            semantic_severity = classify_semantic_severity(semantic_similarity)

            # ---- Composite Score ----
            composite_score = compute_composite_score(
                citation_coverage,
                semantic_similarity
            )

            row = {
                "query": query,
                "system_type": system,
                "retrieval_count": retrieval_count,
                "relevance_score": relevance_score,
                "sufficiency_score": sufficiency_score,
                "validator_decision": decision,
                "citation_coverage_ratio": citation_coverage,
                "citation_severity": citation_severity,
                "semantic_similarity": semantic_similarity,
                "semantic_severity": semantic_severity,
                "composite_hallucination_score": composite_score,
                "latency_seconds": latency,
                "answer_length": len(answer),
                "final_answer": answer
            }

            rows.append(row)

    save_to_csv(rows)


# ==========================================================
# SAVE CSV
# ==========================================================

def save_to_csv(rows: List[Dict]):

    if not rows:
        return

    fieldnames = rows[0].keys()

    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nEvaluation dataset saved to {OUTPUT_FILE}")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    build_evaluation_dataset()
