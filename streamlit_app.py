import os
from typing import List, TypedDict

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# ---------------- Load secrets ----------------
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not API_KEY:
    st.error(" Missing GROQ_API_KEY in .env or Streamlit Secrets.")
    st.stop()

# ---------------- LLM + Embeddings ----------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.0,
    api_key=API_KEY,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ---------------- Vector DB ----------------
@st.cache_resource
def load_vector_db(data_path: str = "data"):
    loaders = [
        TextLoader(os.path.join(data_path, fname))
        for fname in os.listdir(data_path)
        if fname.endswith(".txt")
    ]

    docs_raw = []
    for loader in loaders:
        docs_raw.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(docs_raw)

    vectordb = Chroma.from_documents(
        docs,
        embeddings,
        collection_name="groq_rag_fixed",
    )
    return vectordb

VECTORDb = load_vector_db()

# ---------------- Agent State ----------------
class AgentState(TypedDict):
    question: str
    plan: str
    retrieved_docs: List[str]
    answer: str
    reflection: str

# ---------------- Nodes ----------------
def plan_step(state: AgentState) -> AgentState:
    q = state["question"]
    prompt = f"""You are a planner in a RAG system.

Question: {q}

Respond ONLY:

RETRIEVE: yes/no
EXPLANATION: <short reason>
"""
    result = llm.invoke([HumanMessage(content=prompt)]).content
    state["plan"] = result
    return state

def retrieve_step(state: AgentState) -> AgentState:
    plan_text = state.get("plan", "").lower()
    docs = []

    if "retrieve:" in plan_text and "yes" in plan_text:
        query = state["question"]
        results = VECTORDb.similarity_search_with_relevance_scores(query, k=5)
        for doc, score in results:
            if score and score >= 0.3:
                docs.append(doc.page_content)

    state["retrieved_docs"] = docs
    return state

def answer_step(state: AgentState) -> AgentState:
    question = state["question"]
    docs = state.get("retrieved_docs", [])
    context = "\n\n".join(docs) if docs else "(no relevant context found)"

    prompt = f"""You are a careful RAG assistant.

Question: {question}
Context:
{context}

Write a concise answer.
"""
    result = llm.invoke([HumanMessage(content=prompt)]).content
    state["answer"] = result
    return state

def reflect_step(state: AgentState) -> AgentState:
    question = state["question"]
    answer = state["answer"]

    prompt = f"""Evaluate this RAG answer.

Question: {question}
Answer: {answer}

Respond:
VERDICT: PASS or FAIL
JUSTIFICATION: <short reason>
"""
    result = llm.invoke([HumanMessage(content=prompt)]).content
    state["reflection"] = result
    return state

# ---------------- Build Graph ----------------
graph = StateGraph(AgentState)
graph.add_node("plan", plan_step)
graph.add_node("retrieve", retrieve_step)
graph.add_node("answer", answer_step)
graph.add_node("reflect", reflect_step)

graph.set_entry_point("plan")
graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", "reflect")
graph.add_edge("reflect", END)

APP = graph.compile()

# ---------------- Streamlit UI ----------------
def main():
    st.title("🤖 RAG LangGraph Agent (Groq + Chroma + HF Embeddings)")
    st.write("Ask any question and the RAG agent will plan, retrieve, answer, and reflect.")

    q = st.text_input("Your question:", value="What is artificial intelligence?")

    if st.button("Run RAG Agent"):
        if not q.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Running RAG Agent..."):
            result = APP.invoke({
                "question": q,
                "plan": "",
                "retrieved_docs": [],
                "answer": "",
                "reflection": "",
            })

        st.subheader("📌 Plan")
        st.code(result["plan"])

        st.subheader(" Retrieved Documents")
        if result["retrieved_docs"]:
            for i, d in enumerate(result["retrieved_docs"], start=1):
                st.markdown(f"**Doc {i}:**")
                st.code(d)
        else:
            st.write("No relevant documents retrieved.")

        st.subheader(" Final Answer")
        st.write(result["answer"])

        st.subheader("✔ Reflection")
        st.code(result["reflection"])

if __name__ == "__main__":
    main()
