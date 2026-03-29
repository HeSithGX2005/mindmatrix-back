from typing import Optional
import re

from dependencies import collection
from services.document_service import embed_text


def get_session_documents(session_id: str) -> list[str]:
    results = collection.get(
        where={"session_id": session_id},
        include=["documents", "metadatas"]
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    if not documents:
        return []

    ordered = sorted(
        zip(documents, metadatas),
        key=lambda item: (
            item[1].get("filename", ""),
            item[1].get("chunk_id", 0)
        )
    )

    return [document for document, _ in ordered]


def get_session_documents_with_metadata(session_id: str) -> list[dict]:
    """Return ordered session chunks with chunk index metadata."""
    results = collection.get(
        where={"session_id": session_id},
        include=["documents", "metadatas"]
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    if not documents:
        return []

    ordered = sorted(
        zip(documents, metadatas),
        key=lambda item: (
            item[1].get("filename", ""),
            item[1].get("chunk_id", 0)
        )
    )

    normalized = []
    for ordered_index, (document, metadata) in enumerate(ordered):
        normalized.append({
            "text": document,
            "filename": metadata.get("filename", ""),
            "chunk_index": ordered_index,
            "source_chunk_id": metadata.get("chunk_id", 0),
        })

    return normalized


def _normalize_chunk(text: str) -> str:
    return " ".join((text or "").split())


def _is_low_signal_chunk(text: str) -> bool:
    normalized = _normalize_chunk(text)
    if not normalized:
        return True

    if len(normalized) < 40:
        return True

    alpha_count = sum(1 for char in normalized if char.isalpha())
    if alpha_count < 20:
        return True

    url_like = len(re.findall(r"https?://\\S+|www\\.\\S+", normalized))
    if url_like >= 3:
        return True

    return False


def _dedupe_and_filter_chunks(chunks: list[str], k: int) -> list[str]:
    if not chunks:
        return []

    seen: set[str] = set()
    clean_chunks: list[str] = []

    for chunk in chunks:
        normalized = _normalize_chunk(chunk)
        if not normalized:
            continue

        key = normalized[:220]
        if key in seen:
            continue
        seen.add(key)

        if _is_low_signal_chunk(normalized):
            continue

        clean_chunks.append(normalized)
        if len(clean_chunks) >= max(1, k):
            return clean_chunks

    if clean_chunks:
        return clean_chunks[:max(1, k)]

    # If all chunks are noisy, still return some content instead of empty.
    fallback = []
    for chunk in chunks:
        normalized = _normalize_chunk(chunk)
        if normalized:
            fallback.append(normalized)
        if len(fallback) >= max(1, k):
            break

    return fallback


def retrieve_relevant_chunks(
    question: str,
    session_id: Optional[str] = None,
    k: int = 3
) -> list[str]:
    """
    Retrieve top-k relevant chunks from ChromaDB using vector similarity.
    """
    def _fallback_chunks() -> list[str]:
        if session_id:
            session_docs = get_session_documents(session_id)
            if session_docs:
                return _dedupe_and_filter_chunks(session_docs, k)
        return []

    question_embedding = embed_text(question)

    if not question_embedding:
        return _fallback_chunks()

    query_kwargs = {
        "query_embeddings": [question_embedding],
        "n_results": k
    }

    if session_id:
        query_kwargs["where"] = {"session_id": session_id}

    try:
        results = collection.query(**query_kwargs)
        docs = results.get("documents")

        if docs and docs[0]:
            return _dedupe_and_filter_chunks(docs[0], k)
    except Exception as e:
        print(f"Retrieval query error: {e}")

    return _fallback_chunks()
