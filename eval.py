from typing import List, Dict

from dotenv import load_dotenv
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from agent import APP, AgentState

load_dotenv()

TEST_CASES: List[Dict[str, str]] = [
    {
        "question": "What is artificial intelligence?",
        "reference": "Artificial intelligence is the field of building computer systems that can perform tasks "
                     "that normally require human intelligence, such as understanding language, recognizing patterns, "
                     "and making decisions."
    },
    {
        "question": "What are the benefits of renewable energy?",
        "reference": "Renewable energy reduces greenhouse gas emissions, improves air quality, "
                     "provides domestic energy security, creates jobs, and offers long-term cost stability."
    },
]


def run_agent(question: str) -> str:
    init: AgentState = {
        "question": question,
        "plan": "",
        "retrieved_docs": [],
        "answer": "",
        "reflection": "",
    }
    final = APP.invoke(init)
    return final["answer"]


def main():
    print("Running evaluation on", len(TEST_CASES), "test cases...\n")

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    rouge_scores = []
    cosine_scores = []

    for case in TEST_CASES:
        q = case["question"]
        ref = case["reference"]

        print("QUESTION:", q)
        pred = run_agent(q)
        print("PREDICTED:", pred)
        print("REFERENCE:", ref)

        r = rouge.score(ref, pred)["rougeL"].fmeasure
        rouge_scores.append(r)

        emb = model.encode([ref, pred])
        cos = cosine_similarity([emb[0]], [emb[1]])[0][0]
        cosine_scores.append(cos)

        print(f"ROUGE-L F1: {r:.4f}")
        print(f"Cosine similarity: {cos:.4f}\n{'-'*60}\n")

    print("=== AGGREGATE SCORES ===")
    if rouge_scores:
        print("Average ROUGE-L F1:", sum(rouge_scores) / len(rouge_scores))
    if cosine_scores:
        print("Average cosine similarity:", sum(cosine_scores) / len(cosine_scores))


if __name__ == "__main__":
    main()
