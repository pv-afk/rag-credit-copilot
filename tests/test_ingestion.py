from src.config import RAW_DIR
from src.ingestion import load_documents


def test_load_documents_parses_pdf_and_html_with_relative_paths():
    documents = load_documents(RAW_DIR)

    assert documents, "data/raw should contain the bundled sample documents"
    assert {d.source_format for d in documents} == {"pdf", "html"}
    for doc in documents:
        assert doc.text.strip()
        assert doc.title.strip()
        # Paths are stored relative to the repo root, never as absolute local paths.
        assert doc.source_path.startswith("data/raw/")
    assert len({d.doc_id for d in documents}) == len(documents)
