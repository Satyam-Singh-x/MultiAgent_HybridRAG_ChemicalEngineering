import streamlit as st
from final_multi_agent_rag_system import run_multi_agent_system

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="VAG-RAG | Reliability-Aware RAG",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Validation-Augmented Hybrid RAG (VAG-RAG)")
st.markdown("Reliability-aware Multi-Agent Retrieval-Augmented Generation")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "logs" not in st.session_state:
    st.session_state.logs = []

# -------------------------------------------------
# CHAT DISPLAY
# -------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------
# USER INPUT
# -------------------------------------------------

if prompt := st.chat_input("Ask a Chemical Engineering question..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run system
    with st.spinner("Running Multi-Agent Pipeline..."):
        result = run_multi_agent_system(prompt)

    answer = result["final_answer"]
    retrieval_count = result["retrieval_count"]
    validation = result["validation_result"]
    context = result["formatted_context"]

    # Determine approval status
    decision = validation.get("validation_decision", "UNKNOWN")
    reasoning = validation.get("reasoning", "No reasoning provided.")

    # -------------------------------------------------
    # ASSISTANT RESPONSE
    # -------------------------------------------------

    with st.chat_message("assistant"):

        # Validator Status Banner
        if decision == "APPROVED":
            st.success("✅ Context Approved by Validator")
        else:
            st.error("❌ Context Rejected by Validator")

        # Show Answer
        st.markdown(answer)

        # Show Rejection Reason (if rejected)
        if decision != "APPROVED":
            st.warning(f"**Validator Reason:** {reasoning}")

        # Expandable Logs
        with st.expander("📊 Retrieval & Validation Logs"):
            st.write(f"**Retrieved Chunks:** {retrieval_count}")
            st.write(f"**Validation Decision:** {decision}")
            st.write(f"**Relevance Score:** {validation.get('relevance_score')}")
            st.write(f"**Sufficiency Score:** {validation.get('sufficiency_score')}")

        # Expandable Context
        with st.expander("📚 Retrieved Context"):
            st.text(context)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

