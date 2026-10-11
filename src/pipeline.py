from src.config import TOP_K_FINAL
from src.hybrid import HybridRetriever
from src.llm import generate_answer
from src.schemas import QueryResponse


class RAGPipeline:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self.retriever.load()
        self._loaded = True

    def answer(self, question: str, top_k: int | None = None) -> QueryResponse:
        if not self._loaded:
            raise RuntimeError("Индекс не загружен: вызовите load() после python -m src.build_index")

        retrieved = self.retriever.retrieve(question, top_k=top_k or TOP_K_FINAL)
        result = generate_answer(question, retrieved)

        return QueryResponse(
            question=question,
            answer=result.answer,
            citations=result.citations,
            confidence=result.confidence,
            is_answerable=result.is_answerable,
            retrieved_chunk_ids=[rc.chunk.chunk_id for rc in retrieved],
        )
