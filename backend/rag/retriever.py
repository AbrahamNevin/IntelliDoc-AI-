from typing import List, Dict
import numpy as np

from backend.embeddings.embedder import Embedder
from backend.vectorstore.faiss_store import FAISSVectorStore


class Retriever:
    """
    Retrieves relevant document chunks for a given query using embeddings + FAISS.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: FAISSVectorStore,
        top_k: int = 5,
        min_score: float = 0.3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieves top relevant chunks for a query.
        """
        # 1. Embed the query
        query_vector = self.embedder.embed_query(query)

        # 2. Search vector store
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=self.top_k
        )

        # 3. Filter by similarity score
        filtered = [
            r for r in results if r.get("score", 0) >= self.min_score
        ]

        # 4. Sort by score (descending)
        filtered.sort(key=lambda x: x["score"], reverse=True)

        return filtered

    @staticmethod
    def build_context(chunks: List[Dict]) -> str:
        """
        Builds a context string for LLM input.
        """
        context_blocks = []

        for chunk in chunks:
            block = (
                f"[Document: {chunk.get('doc_name')} | "
                f"Page: {chunk.get('page')} | "
                f"Score: {chunk.get('score'):.2f}]\n"
                f"{chunk.get('chunk_text')}"
            )
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)
