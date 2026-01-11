from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np


class Embedder:
    """
    Generates vector embeddings for text chunks using a sentence transformer.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generates embeddings for document chunks.

        Returns:
            List of chunks with embeddings added.
        """
        texts = [chunk["chunk_text"] for chunk in chunks]

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        enriched_chunks = []

        for chunk, embedding in zip(chunks, embeddings):
            enriched_chunks.append({
                **chunk,
                "embedding": embedding
            })

        return enriched_chunks

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates embedding for a search query.
        """
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
