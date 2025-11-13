import os
from typing import List, TypedDict, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, END


load_dotenv()


llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.0,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


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
        collection_name="groq_rag_fixed_ui",
    )
    return vectordb


VECTORDb = load_vector_db()


class AgentState(TypedDict):
    question: str
    plan: str
    retrieved_docs: List[str]
    answer: str
    reflection: str


def plan_step(state: AgentState) -> AgentState:
    q = state["question"]

    prompt = f"""You are a planner in a RAG (Retrieval-Augmented Generation) system.

Question: {q}

Decide if we should retrieve documents from a knowledge base.
Respond ONLY in this exact format:

RETRIEVE: yes/no
EXPLANATION: <short reason>
"""

    result = llm.invoke([HumanMessage(content=prompt)]).content
    state["plan"] = result
    return state


def retrieve_step(state: AgentState) -> AgentState:
    plan_text = state.get("plan", "").lower()

    docs: List[str] = []

    if "retrieve:" in plan_text and "yes" in plan_text:
        query = state["question"]
        results = VECTORDb.similarity_search_with_relevance_scores(query, k=5)
        for doc, score in results:
            if score is None or score < 0.3:
                continue
            docs.append(f"[score={score:.3f}]\n{doc.page_content}")
    else:
        docs = []

    state["retrieved_docs"] = docs
    return state


def answer_step(state: AgentState) -> AgentState:
    question = state["question"]
    # Strip the score prefix from each doc for context
    raw_docs: List[str] = []
    for d in state.get("retrieved_docs", []):
        if d.startswith("[score="):
            # split on first newline
            parts = d.split("\n", 1)
            if len(parts) == 2:
                raw_docs.append(parts[1])
        else:
            raw_docs.append(d)

    context = "\n\n".join(raw_docs) if raw_docs else "(no relevant context found)"

    prompt = f"""You are a careful assistant in a RAG system.

User Question:
{question}

Retrieved Context (may contain multiple chunks):
{context}

Instructions:
- Use ONLY the parts of the context that are clearly relevant to the question.
- If some chunks are unrelated, IGNORE them completely.
- If the context is empty or not useful, say so briefly and then answer from general knowledge.

Now write a clear, concise answer.
"""

    result = llm.invoke([HumanMessage(content=prompt)]).content
    state["answer"] = result
    return state


def reflect_step(state: AgentState) -> AgentState:
    question = state["question"]
    answer = state["answer"]

    prompt = f"""You are a strict evaluator of RAG answers.

Question: {question}
Answer: {answer}

Evaluate the answer's relevance and completeness.
Respond ONLY in this format:

VERDICT: PASS or FAIL
JUSTIFICATION: <one sentence>
"""

    result = llm.invoke([HumanMessage(content=prompt)]).content
    state["reflection"] = result
    return state


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


def run_agent(question: str) -> AgentState:
    init: AgentState = {
        "question": question,
        "plan": "",
        "retrieved_docs": [],
        "answer": "",
        "reflection": "",
    }
    return APP.invoke(init)


def main():
    st.title("RAG LangGraph Agent (Groq + Better Retrieval)")

    st.write("Ask a question. See planner decision, retrieved docs with relevance, final answer, and reflection.")

    q = st.text_input("Your question:", value="What is artificial intelligence?")

    if st.button("Run RAG Agent") and q.strip():
        with st.spinner("Thinking..."):
            result = run_agent(q.strip())

        st.subheader("Plan")
        st.code(result["plan"])

        st.subheader("Retrieved Documents (with relevance scores)")
        if result["retrieved_docs"]:
            for i, d in enumerate(result["retrieved_docs"], start=1):
                st.markdown(f"**Doc {i}:**")
                st.code(d)
        else:
            st.write("No documents retrieved (planner said no, or no relevant docs).")

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Reflection")
        st.code(result["reflection"])


if __name__ == "__main__":
    main()
