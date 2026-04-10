from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            
            client = chromadb.Client(chromadb.config.Settings(allow_reset=True))
            # We wrap the provided embedding_fn to match Chroma's expected interface if needed,
            # but for this implementation, we'll pass embeddings manually.
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        """Creates a record containing content, embedding, and metadata."""
        return {
            "id": doc.id or f"id_{self._next_index}",
            "content": doc.content,
            "embedding": self._embedding_fn(doc.content),
            "metadata": doc.metadata or {}
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        """Helper to perform similarity search on a list of records."""
        # Calculate similarity for each record
        scored_records = []
        for rec in records:
            # Using dot product as the similarity metric
            score = _dot(query, rec["embedding"])
            scored_records.append({**rec, "score": score})
        
        # Sort by score descending and take top_k
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        """Embed each document's content and store it."""
        for doc in docs:
            record = self._make_record(doc)
            self._next_index += 1
            
            if self._use_chroma:
                self._collection.add(
                    ids=[record["id"]],
                    documents=[record["content"]],
                    embeddings=[record["embedding"]],
                    metadatas=[record["metadata"]]
                )
            else:
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        """Find the top_k most similar documents to query."""
        query_vec = self._embedding_fn(query)
        
        if self._use_chroma:
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k
            )
            # Flatten Chroma's nested response format
            formatted = []
            for i in range(len(results["ids"][0])):
                formatted.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": results["distances"][0][i]  # Note: Chroma uses distance, not dot prod
                })
            return formatted
        else:
            return self._search_records(query_vec, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)
    
    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        """Search with optional metadata pre-filtering."""
        query_vec = self._embedding_fn(query)
        
        if self._use_chroma:
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                where=metadata_filter
            )
            # Format results similarly to the search method...
            return [{"id": results["ids"][0][i], "content": results["documents"][0][i]} for i in range(len(results["ids"][0]))]
        else:
            # Manual filtering for in-memory
            filtered_records = self._store
            if metadata_filter:
                filtered_records = [
                    r for r in self._store 
                    if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
                ]
            return self._search_records(query_vec, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        """Remove all chunks belonging to a document."""
        initial_count = self.get_collection_size()
        
        if self._use_chroma:
            # Chroma filters by metadata for deletion
            self._collection.delete(where={"doc_id": doc_id})
            return self._collection.count() < initial_count
        else:
            self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_count
