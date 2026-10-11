from src.chunking import chunk_document
from src.schemas import Document


def make_doc(word_count: int) -> Document:
    text = " ".join(f"word{i}" for i in range(word_count))
    return Document(doc_id="doc1", source_path="doc1.pdf", source_format="pdf", title="Test Doc", text=text)


def test_chunk_document_produces_multiple_chunks_for_long_text():
    doc = make_doc(1000)
    chunks = chunk_document(doc, chunk_size_words=200, overlap_words=40)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.doc_id == "doc1"
        assert chunk.text.strip() != ""
        assert chunk.title == "Test Doc"


def test_chunk_ids_are_unique_and_sequential():
    doc = make_doc(500)
    chunks = chunk_document(doc, chunk_size_words=100, overlap_words=20)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_overlap():
    doc = make_doc(300)
    chunks = chunk_document(doc, chunk_size_words=100, overlap_words=30)

    assert len(chunks) >= 2
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    overlap = set(first_words[-30:]) & set(second_words[:30])
    assert len(overlap) > 0


def test_short_document_produces_single_chunk():
    doc = make_doc(50)
    chunks = chunk_document(doc, chunk_size_words=200, overlap_words=40)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
