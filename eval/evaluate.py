"""Оценка retrieval: document-level Recall@k на размеченном наборе eval_set.json.

Попадание засчитывается, если среди top-k фрагментов есть хотя бы один из ожидаемого
документа; содержит ли этот фрагмент сам ответ, метрика не проверяет. Для вопросов вне
домена выводится решение порога отказа (MIN_RELEVANCE_SCORE). LLM не вызывается.

Запуск:
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
    print(f"\n=== Recall@{k} (document-level), вопросов по теме: {len(in_domain)} ===\n")
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
        print(f"=== Вопросы вне домена: {len(out_of_domain)} ===\n")
        for item in out_of_domain:
            retrieved = retriever.retrieve(item["question"], top_k=k)
            top_score = top_vector_score(retrieved)
            would_answer = top_score >= MIN_RELEVANCE_SCORE
            status = "ОТВЕТИТ (ошибка)" if would_answer else "откажет (ожидаемо)"
            print(f"[{status}] {item['id']}: {item['question'][:70]} (max cosine={top_score:.4f})")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    evaluate(k=args.k)
