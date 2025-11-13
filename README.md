# RAG LangGraph Agent (Groq + HuggingFace Embeddings, Improved Retrieval + UI)

This project implements a Retrieval-Augmented Generation (RAG) agent using:

- **LangGraph** for the workflow (plan → retrieve → answer → reflect)
- **LangChain** + **Chroma** as the vector store
- **Groq** (`ChatGroq` with Llama 3.1) as the LLM
- **HuggingFace Sentence-Transformer** embeddings (`all-mpnet-base-v2`) for better retrieval quality
- A **Streamlit UI** that shows relevance scores for retrieved documents

Fixes / improvements vs earlier version:
- Uses `similarity_search_with_relevance_scores` + a relevance threshold
- Stronger planner prompt (`RETRIEVE: yes/no` + explanation)
- Answer prompt explicitly tells the model to ignore irrelevant context
- Updated imports (`langchain-huggingface`) and `numpy<2` for compatibility

## Project Structure

```bash
rag_agent_groq_rag_fixed_ui/
├── data/
│   ├── renewable_energy.txt
│   └── ai_intro.txt
├── agent.py
├── streamlit_app.py
├── eval.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a free Groq API key at https://console.groq.com/keys and add a `.env` file:

```bash
GROQ_API_KEY=gsk_your_key_here
```

## Run CLI Agent

```bash
python agent.py
```

## Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

The UI shows:
- Plan
- Retrieved documents + relevance scores
- Final answer
- Reflection
