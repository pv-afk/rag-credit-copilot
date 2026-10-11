from src.bm25_index import BM25Index
from src.chunk_store import save_chunks
from src.chunking import chunk_documents
from src.config import RAW_DIR
from src.embeddings import embed_texts
from src.ingestion import load_documents
from src.vectorstore import VectorStore


def main() -> None:
    print(f"Loading documents from {RAW_DIR} ...")
    documents = load_documents(RAW_DIR)
    print(f"Loaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    print(f"Produced {len(chunks)} chunks")
    save_chunks(chunks)

    print("Computing embeddings ...")
    embeddings = embed_texts([c.text for c in chunks])

    print("Building FAISS index ...")
    vector_store = VectorStore()
    vector_store.build(chunks, embeddings)
    vector_store.save()

    print("Building BM25 index ...")
    bm25_index = BM25Index()
    bm25_index.build(chunks)
    bm25_index.save()

    print("Index build complete.")


if __name__ == "__main__":
    main()
