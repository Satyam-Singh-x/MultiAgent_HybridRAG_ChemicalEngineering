"""
Multi-Agent Agentic RAG System (LangGraph Orchestrated)
--------------------------------------------------------

Agents:
1. Query Analyzer
2. Hybrid Retrieval
3. Validator
4. Generator

Includes conditional branching between Validator and Generator.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from query_analyzer_agent import analyze_query
from hybrid_rag_agent import HybridRetrievalAgent
from validator_agent import validate_retrieval
from generator_agent import generate_answer


# ==========================================================
# ---------------- STATE DEFINITION ------------------------
# ==========================================================

class AgentState(TypedDict):
    user_query: str
    structured_query: Optional[dict]
    retrieval_package: Optional[dict]
    retrieval_count: Optional[int]
    validation_result: Optional[dict]
    final_answer: Optional[str]


# ==========================================================
# ---------------- NODE FUNCTIONS --------------------------
# ==========================================================

# Node 1: Query Analyzer
def query_analyzer_node(state: AgentState):
    structured = analyze_query(state["user_query"])
    return {"structured_query": structured}


# Node 2: Hybrid Retrieval
retriever = HybridRetrievalAgent()

def retrieval_node(state: AgentState):
    retrieval_package = retriever.retrieve(state["structured_query"])

    retrieval_count = len(
        retrieval_package.get("retrieved_documents", [])
    )

    return {
        "retrieval_package": retrieval_package,
        "retrieval_count": retrieval_count
    }



# Node 3: Validator
def validator_node(state: AgentState):
    validation = validate_retrieval(state["retrieval_package"])
    return {"validation_result": validation}


# Node 4: Generator
def generator_node(state: AgentState):
    answer = generate_answer(
        state["retrieval_package"],
        state["validation_result"]
    )
    return {"final_answer": answer}


# Node 5: Refusal Handler
def refusal_node(state: AgentState):
    reason = state["validation_result"]["reasoning"]
    return {
        "final_answer": (
            "The system cannot generate a grounded response.\n\n"
            f"Validator Reason: {reason}"
        )
    }


# ==========================================================
# ---------------- CONDITIONAL LOGIC -----------------------
# ==========================================================

def validation_decision(state: AgentState):
    if state["validation_result"]["validation_decision"] == "APPROVED":
        return "generator"
    else:
        return "refusal"


# ==========================================================
# ---------------- GRAPH CONSTRUCTION ----------------------
# ==========================================================

def build_graph():

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("query_analyzer", query_analyzer_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("validator", validator_node)
    graph.add_node("generator", generator_node)
    graph.add_node("refusal", refusal_node)

    # Define flow
    graph.set_entry_point("query_analyzer")

    graph.add_edge("query_analyzer", "retrieval")
    graph.add_edge("retrieval", "validator")

    graph.add_conditional_edges(
        "validator",
        validation_decision,
        {
            "generator": "generator",
            "refusal": "refusal"
        }
    )

    graph.add_edge("generator", END)
    graph.add_edge("refusal", END)

    return graph.compile()

# ==========================================================
# --------------- PUBLIC EVALUATION FUNCTION ---------------
# ==========================================================

def run_multi_agent_system(query: str):
    """
    Public evaluation wrapper for full multi-agent Agentic RAG system.

    Returns:
    --------
    dict containing:
        - final_answer
        - retrieval_count
        - formatted_context
        - validation_result
    """

    app = build_graph()

    result = app.invoke({
        "user_query": query
    })

    retrieval_package = result.get("retrieval_package", {})

    return {
        "final_answer": result.get("final_answer", ""),
        "retrieval_count": result.get("retrieval_count", None),
        "formatted_context": retrieval_package.get("formatted_context", ""),
        "validation_result": result.get("validation_result", None)
    }


# ==========================================================
# ---------------- MAIN EXECUTION --------------------------
# ==========================================================

if __name__ == "__main__":

    app = build_graph()

    while True:
        user_input = input("\nEnter query (or type 'exit'): ")
        if user_input.lower() == "exit":
            break

        result = app.invoke({
            "user_query": user_input
        })

        print("\n--- FINAL OUTPUT ---\n")
        print(result["final_answer"])
