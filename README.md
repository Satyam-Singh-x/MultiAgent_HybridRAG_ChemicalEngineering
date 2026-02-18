Validation-Augmented Hybrid RAG (VAG-RAG)

A Reliability-Aware Multi-Agent Retrieval-Augmented Generation System

Domain: Chemical Engineering Knowledge Systems

Detailed Youtube Video Explaination: https://youtu.be/AMX2DU09rok



📌 Overview

This repository contains the complete implementation, evaluation framework, and research documentation for a Multi-Agent Validation-Augmented Hybrid Retrieval-Augmented Generation (VAG-RAG) system.

The project investigates a core research question:

How can hallucination in domain-specific RAG systems be reduced structurally using validation and agent orchestration rather than relying solely on improved retrieval?

Unlike traditional linear RAG pipelines (Retrieval → Generation), this work introduces a validation-controlled multi-agent architecture that enables:

Context sufficiency assessment

Conditional generation

Safe abstention

Citation-enforced grounded responses

The system is evaluated across multiple baseline architectures using a structured hallucination-aware evaluation framework.



🏗️ System Architectures Implemented

Four distinct architectures were designed and compared:

Dense Retrieval Baseline

FAISS + SentenceTransformer embeddings

Semantic similarity search

Keyword Baseline (BM25)

Lexical token-based retrieval

Inverse document frequency ranking

Hybrid Baseline

Dense + BM25 merged retrieval

Improved recall, no validation

Multi-Agent VAG-RAG (Proposed System)

Query Analyzer Agent

Hybrid Retrieval Agent

Validator Agent

Generator Agent

Conditional branching with abstention



🧠 Core Innovation

The proposed Validation-Augmented Hybrid RAG (VAG-RAG) introduces:

A Validator Agent between retrieval and generation

Relevance scoring

Context sufficiency scoring

Conditional execution logic

Refusal behavior for weak grounding

Mandatory citation enforcement

This transforms the objective from:

Always Generate



to

Generate Only When Grounded

📂 Repository Structure

.

├── case_study/

│   ├── src/

│   │   ├── app.py

│   │   ├── final_multi_agent_rag_system.py

│   │   ├── query_analyzer_agent.py

│   │   ├── hybrid_rag_agent.py

│   │   ├── validator_agent.py

│   │   ├── generator_agent.py

│   │   ├── dense_baseline.py

│   │   ├── keyword_search_baseline.py

│   │   ├── hybrid_rag_baseline.py

│   │   ├── ingest.py

│   │   ├── evaluation_dataset_builder.py

│   │   └── evaluation_results.csv

│   └── evaluation_notebook.ipynb

│
├── Technical_document.docx

├── Final_draft_paper.docx

└── README.md



📊 Evaluation Framework

A structured evaluation dataset was created consisting of:


Fully Covered Queries

Partially Covered Queries

Ambiguous Queries

Out-of-Domain Queries

Each query was tested across all four architectures.

Evaluation Metrics

Retrieval Count

Citation Coverage Ratio

Context–Answer Semantic Similarity

Composite Hallucination Score

Latency

Abstention Behavior

Composite Hallucination Score:

0.4 × (1 - Citation Coverage)

+ 0.6 × (1 - Semantic Similarity)


Filtered evaluation separates abstention from hallucination.



📈 Key Findings

Hybrid retrieval improves recall but does not prevent hallucination.

Baseline systems always generate answers.

Multi-agent architecture selectively generates.

Validator-driven abstention reduces unsafe outputs.

On approved queries, VAG-RAG achieves:

Lowest hallucination score

Highest citation coverage

Strong semantic grounding

This demonstrates that structural validation outperforms retrieval-only improvements in domain-critical settings.



🧪 Dataset

The corpus consists of Chemical Engineering documents including:

Unit Operations textbooks

Distillation design references

Process safety documents

Equipment fundamentals

Reaction engineering materials

All documents are:

Page-level segmented

Metadata enriched

Chunked with overlap

Indexed via FAISS and BM25

⚙️ Installation

git clone https://github.com/Satyam-Singh-x/MultiAgent_HybridRAG_ChemicalEngineering/

cd case_study/src
pip install -r requirements.txt


Ensure:

Python 3.9+

FAISS installed

SentenceTransformers available

LangGraph installed

▶️ Running the Multi-Agent System
python final_multi_agent_rag_system.py


Or if using Streamlit UI:

streamlit run app.py



🧩 Multi-Agent Flow

User Query
    ↓
Query Analyzer
    ↓
Hybrid Retrieval
    ↓
Validator Agent

    ↓ (Conditional)
    
Generator Agent OR Refusal


If validation fails, the system outputs:

Explicit abstention

Validator reasoning

No fabricated content



📘 Research Documentation

Included in this repository:

📄 Technical Document (Implementation Details)

📄 Research Draft Paper (Journal-ready format)

📓 Evaluation Notebook

📊 Structured Results CSV



🎯 Target Applications

This architecture is suitable for:

Chemical Engineering QA systems

Healthcare knowledge assistants

Legal advisory systems

Industrial safety AI tools

Domain-specific educational agents



🔬 Research Contributions

Multi-agent validation-controlled RAG architecture

Dual hallucination evaluation framework

Abstention-aware evaluation methodology

Domain-specific grounding enforcement

Reliability-aware design paradigm



🚀 Future Directions

Cross-encoder re-ranking

Learned validation model

Confidence-aware generation

Adaptive retrieval depth

Domain-specific embedding fine-tuning


📜 License

This repository is developed for academic and research purposes.

👨‍💻 Author

Satyam Singh

Chemical Engineering & AI Research

Multi-Agent Reliability-Aware RAG Systems
