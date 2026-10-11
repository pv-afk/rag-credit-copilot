"""
Retrieval evaluation: Recall@k over the hand-labeled eval set.

Usage:
    python -m eval.evaluate [--k 5]
"""
import argparse
import json
from pathlib import Path

from src.config import MIN_RELEVANCE_SCORE
from src.hybrid import HybridRetriever
from src.llm import top_vector_score

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def evaluate(k: int = 5) -> None:
    retriever = HybridRetriever()
    retriever.load()

    eval_set = load_eval_set()
    in_domain = [q for q in eval_set if q["expected_source_contains"]]
    out_of_domain = [q for q in eval_set if not q["expected_source_contains"]]

    hits = 0
    print(f"\n=== Recall@{k} on {len(in_domain)} in-domain questions ===\n")
    for item in in_domain:
        retrieved = retriever.retrieve(item["question"], top_k=k)
        expected = item["expected_source_contains"]
        hit = any(expected in rc.chunk.source_path for rc in retrieved)
        hits += hit
        status = "HIT " if hit else "MISS"
        print(f"[{status}] {item['id']}: {item['question'][:70]}")

    recall_at_k = hits / len(in_domain) if in_domain else 0.0
    print(f"\nRecall@{k}: {hits}/{len(in_domain)} = {recall_at_k:.2%}\n")

    if out_of_domain:
        print(f"=== Out-of-domain check ({len(out_of_domain)} questions) ===\n")
        for item in out_of_domain:
            retrieved = retriever.retrieve(item["question"], top_k=k)
            top_score = top_vector_score(retrieved)
            would_answer = top_score >= MIN_RELEVANCE_SCORE
            status = "WOULD ANSWER (unexpected)" if would_answer else "would say 'I don't know' (expected)"
            print(f"[{status}] {item['id']}: {item['question'][:70]} (top vector score={top_score:.4f})")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    evaluate(k=args.k)
