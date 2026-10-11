from src.bm25_index import BM25Index
from src.chunk_store import save_chunks
from src.chunking import chunk_documents
from src.config import PROCESSED_DIR, RAW_DIR
from src.embeddings import embed_texts
from src.ingestion import load_documents
from src.vectorstore import VectorStore


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    documents = load_documents(RAW_DIR)
    chunks = chunk_documents(documents)
    print(f"Документов: {len(documents)}, фрагментов: {len(chunks)}")
    save_chunks(chunks)

    print("Считаю эмбеддинги и строю FAISS-индекс ...")
    embeddings = embed_texts([c.text for c in chunks])
    vector_store = VectorStore()
    vector_store.build(chunks, embeddings)
    vector_store.save()

    print("Строю BM25-индекс ...")
    bm25_index = BM25Index()
    bm25_index.build(chunks)
    bm25_index.save()

    print(f"Готово: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
