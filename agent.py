import os
from typing import List, TypedDict

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


load_dotenv()


# LLM + Embeddings 

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.0,
)

# Better quality embeddings than MiniLM
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


#  Vector DB 

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


#  Agent State 

class AgentState(TypedDict):
    question: str
    plan: str
    retrieved_docs: List[str]
    answer: str
    reflection: str


#  Nodes 

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

    print("\n--- PLAN STEP ---")
    print(result)

    state["plan"] = result
    return state


def retrieve_step(state: AgentState) -> AgentState:
    plan_text = state.get("plan", "").lower()

    docs: List[str] = []

    if "retrieve:" in plan_text and "yes" in plan_text:
        query = state["question"]
        # Use relevance scores + threshold
        results = VECTORDb.similarity_search_with_relevance_scores(query, k=5)
        for doc, score in results:
            if score is None or score < 0.3:
                continue
            docs.append(doc.page_content)

    print("\n--- RETRIEVE STEP ---")
    if docs:
        for i, d in enumerate(docs, start=1):
            print(f"Doc {i}: {d[:200]}...\n")
    else:
        print("No relevant documents retrieved (or planner said no).")

    state["retrieved_docs"] = docs
    return state


def answer_step(state: AgentState) -> AgentState:
    question = state["question"]
    docs = state.get("retrieved_docs", [])

    context = "\n\n".join(docs) if docs else "(no relevant context found)"

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

    print("\n--- ANSWER STEP ---")
    print(result)

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

    print("\n--- REFLECTION STEP ---")
    print(result)

    state["reflection"] = result
    return state


#  Build Graph 

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


if __name__ == "__main__":
    q = input("Ask: ")
    init: AgentState = {
        "question": q,
        "plan": "",
        "retrieved_docs": [],
        "answer": "",
        "reflection": "",
    }
    out = APP.invoke(init)
    print("\n=== FINAL STATE ===")
    for k, v in out.items():
        print(f"\n[{k.upper()}]\n{v}\n")
