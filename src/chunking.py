from src.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS
from src.schemas import Chunk, Document


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_document(
    document: Document,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    paragraphs = _split_paragraphs(document.text)
    words: list[str] = []
    for p in paragraphs:
        words.extend(p.split(" "))
        words.append("\n")

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = max(chunk_size_words - overlap_words, 1)
    while start < len(words):
        window = words[start : start + chunk_size_words]
        text = " ".join(window).replace(" \n ", "\n").strip()
        if text:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}_{index}",
                    doc_id=document.doc_id,
                    title=document.title,
                    source_path=document.source_path,
                    chunk_index=index,
                    text=text,
                )
            )
            index += 1
        if start + chunk_size_words >= len(words):
            break
        start += step
    return chunks


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
