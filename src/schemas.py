from typing import Literal, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    source_path: str
    source_format: Literal["pdf", "html"]
    title: str
    text: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    chunk_index: int
    text: str


class RetrievedChunk(BaseModel):
    chunk: Chunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0


class Citation(BaseModel):
    chunk_id: str
    doc_title: str
    quote: str = Field(description="Short verbatim quote from the cited chunk supporting the claim")


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low", "insufficient_context"]
    is_answerable: bool = Field(
        description="False if the retrieved context does not contain enough information to answer"
    )


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    confidence: str
    is_answerable: bool
    retrieved_chunk_ids: list[str]
