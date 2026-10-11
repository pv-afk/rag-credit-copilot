import hashlib
import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.config import BASE_DIR
from src.schemas import Document


def _doc_id_for(path: Path) -> str:
    return hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:12]


def _relative_source_path(path: Path) -> str:
    # Store paths relative to the repo root so indexes are portable and don't
    # embed machine-specific absolute paths.
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return _normalize_whitespace("\n\n".join(pages))


def _load_html(path: Path) -> tuple[str, str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    title_tag = soup.find("title") or soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else path.stem
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return title, _normalize_whitespace(text)


def _title_from_pdf_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:150]
    return fallback


def load_documents(raw_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            text = _load_pdf(path)
            title = _title_from_pdf_text(text, path.stem)
            documents.append(
                Document(
                    doc_id=_doc_id_for(path),
                    source_path=_relative_source_path(path),
                    source_format="pdf",
                    title=title,
                    text=text,
                )
            )
        elif path.suffix.lower() in (".html", ".htm"):
            title, text = _load_html(path)
            documents.append(
                Document(
                    doc_id=_doc_id_for(path),
                    source_path=_relative_source_path(path),
                    source_format="html",
                    title=title,
                    text=text,
                )
            )
    return documents
