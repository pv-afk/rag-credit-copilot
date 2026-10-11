from pathlib import Path

from src.config import CHUNKS_PATH
from src.schemas import Chunk


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def load_chunks(path: Path = CHUNKS_PATH) -> dict[str, Chunk]:
    chunks: dict[str, Chunk] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = Chunk.model_validate_json(line)
            chunks[chunk.chunk_id] = chunk
    return chunks
