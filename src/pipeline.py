from src.hybrid import HybridRetriever
from src.llm import generate_answer
from src.schemas import QueryResponse


class RAGPipeline:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self._loaded = False

    def load(self) -> None:
        self.retriever.load()
        self._loaded = True

    def answer(self, question: str, top_k: int | None = None) -> QueryResponse:
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded. Call .load() first (or run scripts/build_index.py).")

        retrieved = self.retriever.retrieve(question, top_k=top_k) if top_k else self.retriever.retrieve(question)
        result = generate_answer(question, retrieved)

        return QueryResponse(
            question=question,
            answer=result.answer,
            citations=result.citations,
            confidence=result.confidence,
            is_answerable=result.is_answerable,
            retrieved_chunk_ids=[rc.chunk.chunk_id for rc in retrieved],
        )
