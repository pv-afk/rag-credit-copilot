from src.config import RAW_DIR
from src.ingestion import load_documents


def test_load_documents_parses_pdf_and_html_with_relative_paths():
    documents = load_documents(RAW_DIR)

    assert documents, "data/raw should contain the bundled sample documents"
    assert {d.source_format for d in documents} == {"pdf", "html"}
    for doc in documents:
        assert doc.text.strip()
        assert doc.title.strip()
        # Пути хранятся относительно корня репозитория, без локальных абсолютных путей.
        assert doc.source_path.startswith("data/raw/")
    assert len({d.doc_id for d in documents}) == len(documents)


def test_load_documents_skips_unsupported_files(tmp_path):
    html = "<html><title>Заголовок</title><body><p>Текст</p></body></html>"
    (tmp_path / "doc.html").write_text(html, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("не документ", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [d.title for d in documents] == ["Заголовок"]
