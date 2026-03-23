import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI
from psycopg2.extras import Json, RealDictCursor, execute_values
from rank_bm25 import BM25Okapi

load_dotenv()


DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_RERANK_MODEL = os.getenv("RERANK_MODEL", DEFAULT_CHAT_MODEL)
DEFAULT_EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
DEFAULT_TABLE_NAME = os.getenv("PGVECTOR_TABLE_NAME", "knowledge_base_documents")


class KnowledgeBaseError(Exception):
    """Raised when the knowledge-base workflow fails."""


@dataclass
class KnowledgeChunk:
    source_id: str
    title: str
    category: str
    content: str
    metadata: Dict[str, Any]
    chunk_index: int


@dataclass
class RetrievedChunk:
    id: str
    source_id: str
    title: str
    category: str
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    vector_distance: Optional[float] = None
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise KnowledgeBaseError("OPENAI_API_KEY is not configured.")

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base:
        client_kwargs["base_url"] = api_base
    return OpenAI(**client_kwargs)


def get_postgresql_connection():
    required_keys = [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        raise KnowledgeBaseError(
            f"PostgreSQL environment variables are missing: {', '.join(missing_keys)}"
        )

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        sslmode=os.getenv("POSTGRES_SSLMODE", "prefer"),
        connect_timeout=int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10")),
    )


def preprocess_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[str]:
    """Normalize and split text into overlapping chunks."""
    if not text:
        return []

    normalized = re.sub(r"\r\n?", "\n", text)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if not normalized:
        return []

    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            overlap = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = f"{overlap}\n\n{paragraph}".strip()
        else:
            for start in range(0, len(paragraph), max(chunk_size - chunk_overlap, 1)):
                chunk = paragraph[start : start + chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
            current = ""

    if current:
        chunks.append(current)

    return chunks


def build_knowledge_chunks(
    source_id: str,
    title: str,
    category: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[KnowledgeChunk]:
    base_metadata = metadata or {}
    chunks = preprocess_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [
        KnowledgeChunk(
            source_id=source_id,
            title=title,
            category=category,
            content=chunk,
            metadata=base_metadata,
            chunk_index=index,
        )
        for index, chunk in enumerate(chunks)
    ]


def generate_embeddings(
    texts: Sequence[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: Optional[int] = DEFAULT_EMBEDDING_DIMENSIONS,
) -> List[List[float]]:
    if not texts:
        return []

    client = get_openai_client()
    request_kwargs: Dict[str, Any] = {"model": model, "input": list(texts)}
    if dimensions and model.startswith("text-embedding-3"):
        request_kwargs["dimensions"] = dimensions

    response = client.embeddings.create(**request_kwargs)
    return [item.embedding for item in response.data]


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def batch_insert_knowledge(
    chunks: Sequence[KnowledgeChunk],
    table_name: str = DEFAULT_TABLE_NAME,
    batch_size: int = 50,
) -> int:
    if not chunks:
        return 0

    texts = [chunk.content for chunk in chunks]
    embeddings = generate_embeddings(texts)
    rows = [
        (
            chunk.source_id,
            chunk.title,
            chunk.category,
            chunk.content,
            Json(chunk.metadata),
            chunk.chunk_index,
            _vector_literal(embedding),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    insert_sql = f"""
        INSERT INTO {table_name} (
            source_id,
            title,
            category,
            content,
            metadata,
            chunk_index,
            embedding
        )
        VALUES %s
    """

    with get_postgresql_connection() as conn:
        with conn.cursor() as cursor:
            for index in range(0, len(rows), batch_size):
                batch = rows[index : index + batch_size]
                execute_values(
                    cursor,
                    insert_sql,
                    batch,
                    template="(%s, %s, %s, %s, %s, %s, %s::vector)",
                )
        conn.commit()

    return len(rows)


def execute_sql_file(sql_file_path: str) -> None:
    """Execute a SQL file against the configured PostgreSQL database."""
    with open(sql_file_path, "r", encoding="utf-8") as file:
        sql = file.read()

    with get_postgresql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
        conn.commit()


def _tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _normalize_scores(items: Sequence[float], reverse: bool = False) -> List[float]:
    if not items:
        return []
    if reverse:
        items = [(-1.0) * item for item in items]

    min_score = min(items)
    max_score = max(items)
    if max_score == min_score:
        return [1.0 for _ in items]
    return [(item - min_score) / (max_score - min_score) for item in items]


def _fetch_vector_candidates(
    query_embedding: Sequence[float],
    category: Optional[str],
    metadata_filter: Optional[Dict[str, Any]],
    limit: int,
    table_name: str,
) -> List[RetrievedChunk]:
    conditions: List[str] = []
    parameters: List[Any] = [_vector_literal(query_embedding)]

    if category:
        conditions.append("category = %s")
        parameters.append(category)
    if metadata_filter:
        conditions.append("metadata @> %s::jsonb")
        parameters.append(json.dumps(metadata_filter))

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"""
        SELECT
            id::text,
            source_id,
            title,
            category,
            content,
            metadata,
            chunk_index,
            embedding <=> %s::vector AS vector_distance
        FROM {table_name}
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    parameters.append(_vector_literal(query_embedding))
    parameters.append(limit)

    with get_postgresql_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql, parameters)
            records = cursor.fetchall()

    return [
        RetrievedChunk(
            id=record["id"],
            source_id=record["source_id"],
            title=record["title"],
            category=record["category"],
            content=record["content"],
            metadata=record["metadata"] or {},
            chunk_index=record["chunk_index"],
            vector_distance=record["vector_distance"],
        )
        for record in records
    ]


def _fetch_text_candidates(
    query: str,
    category: Optional[str],
    metadata_filter: Optional[Dict[str, Any]],
    limit: int,
    table_name: str,
) -> List[RetrievedChunk]:
    conditions: List[str] = []
    parameters: List[Any] = [query]

    if category:
        conditions.append("category = %s")
        parameters.append(category)
    if metadata_filter:
        conditions.append("metadata @> %s::jsonb")
        parameters.append(json.dumps(metadata_filter))

    where_clause = f"AND {' AND '.join(conditions)}" if conditions else ""
    sql = f"""
        SELECT
            id::text,
            source_id,
            title,
            category,
            content,
            metadata,
            chunk_index
        FROM {table_name}
        WHERE content_tsv @@ websearch_to_tsquery('simple', %s)
        {where_clause}
        ORDER BY ts_rank_cd(content_tsv, websearch_to_tsquery('simple', %s)) DESC
        LIMIT %s
    """
    parameters.append(query)
    parameters.append(limit)

    with get_postgresql_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql, parameters)
            records = cursor.fetchall()

    return [
        RetrievedChunk(
            id=record["id"],
            source_id=record["source_id"],
            title=record["title"],
            category=record["category"],
            content=record["content"],
            metadata=record["metadata"] or {},
            chunk_index=record["chunk_index"],
        )
        for record in records
    ]


def hybrid_search(
    query: str,
    category: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    vector_limit: int = 20,
    keyword_limit: int = 20,
    table_name: str = DEFAULT_TABLE_NAME,
) -> List[RetrievedChunk]:
    query_embedding = generate_embeddings([query])[0]
    vector_candidates = _fetch_vector_candidates(
        query_embedding=query_embedding,
        category=category,
        metadata_filter=metadata_filter,
        limit=vector_limit,
        table_name=table_name,
    )
    text_candidates = _fetch_text_candidates(
        query=query,
        category=category,
        metadata_filter=metadata_filter,
        limit=keyword_limit,
        table_name=table_name,
    )

    merged: Dict[str, RetrievedChunk] = {}
    for item in vector_candidates + text_candidates:
        existing = merged.get(item.id)
        if existing is None:
            merged[item.id] = item
            continue
        if item.vector_distance is not None:
            existing.vector_distance = item.vector_distance

    candidates = list(merged.values())
    if not candidates:
        return []

    corpus = [_tokenize_for_bm25(chunk.content) for chunk in candidates]
    tokenized_query = _tokenize_for_bm25(query)
    if any(tokens for tokens in corpus) and tokenized_query:
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(tokenized_query).tolist()
    else:
        bm25_scores = [0.0 for _ in candidates]

    vector_distances = [
        item.vector_distance if item.vector_distance is not None else 999.0
        for item in candidates
    ]
    normalized_vector = _normalize_scores(vector_distances, reverse=True)
    normalized_bm25 = _normalize_scores(bm25_scores)

    for index, item in enumerate(candidates):
        item.vector_score = normalized_vector[index]
        item.bm25_score = normalized_bm25[index]
        item.final_score = (0.6 * item.vector_score) + (0.4 * item.bm25_score)

    candidates.sort(key=lambda item: item.final_score, reverse=True)
    return candidates[:top_k]


def rerank_documents(
    query: str,
    documents: Sequence[RetrievedChunk],
    model: str = DEFAULT_RERANK_MODEL,
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    if not documents:
        return []

    client = get_openai_client()
    doc_payload = [
        {
            "id": document.id,
            "title": document.title,
            "category": document.category,
            "content": document.content[:1000],
            "metadata": document.metadata,
        }
        for document in documents
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rerank retrieved knowledge chunks for a RAG pipeline. "
                        "Return JSON with key 'results' containing items with "
                        "'id' and 'score' between 0 and 1."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "documents": doc_payload,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        payload = json.loads(response.choices[0].message.content)
        score_map = {
            item["id"]: float(item.get("score", 0.0))
            for item in payload.get("results", [])
            if "id" in item
        }
        reranked = []
        for document in documents:
            document.rerank_score = score_map.get(document.id, document.final_score)
            document.final_score = (0.7 * document.rerank_score) + (0.3 * document.final_score)
            reranked.append(document)
        reranked.sort(key=lambda item: item.final_score, reverse=True)
    except Exception:
        reranked = list(documents)
        reranked.sort(key=lambda item: item.final_score, reverse=True)

    if top_k is not None:
        return reranked[:top_k]
    return reranked


def answer_question_with_rag(
    question: str,
    category: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    retrieval_top_k: int = 8,
    final_top_k: int = 4,
    model: str = DEFAULT_CHAT_MODEL,
    table_name: str = DEFAULT_TABLE_NAME,
) -> Dict[str, Any]:
    retrieved = hybrid_search(
        query=question,
        category=category,
        metadata_filter=metadata_filter,
        top_k=retrieval_top_k,
        vector_limit=max(retrieval_top_k * 3, 12),
        keyword_limit=max(retrieval_top_k * 3, 12),
        table_name=table_name,
    )
    reranked = rerank_documents(question, retrieved, top_k=final_top_k)

    if not reranked:
        return {
            "answer": "I could not find relevant knowledge base content for this question.",
            "sources": [],
            "documents": [],
        }

    context_blocks = []
    for document in reranked:
        context_blocks.append(
            "\n".join(
                [
                    f"[{document.id}]",
                    f"Title: {document.title}",
                    f"Category: {document.category}",
                    f"Metadata: {json.dumps(document.metadata, ensure_ascii=False)}",
                    f"Content: {document.content}",
                ]
            )
        )

    prompt = "\n\n".join(context_blocks)
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided knowledge base context. "
                    "Cite chunk ids in square brackets. "
                    "If the context is insufficient, say so explicitly and do not invent facts. "
                    "Prefer concise, operational answers."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Knowledge base context:\n{prompt}\n\n"
                    "Write the best possible answer grounded in the context."
                ),
            },
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {
                "id": document.id,
                "source_id": document.source_id,
                "title": document.title,
                "category": document.category,
                "chunk_index": document.chunk_index,
                "metadata": document.metadata,
                "score": document.final_score,
            }
            for document in reranked
        ],
        "documents": reranked,
    }
