import faiss
import numpy as np
import pickle
from typing import List, Dict


class FAISSVectorStore:
    """
    FAISS-based vector store for similarity search.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata = []

    def add(self, embedded_chunks: List[Dict]):
        """
        Adds embeddings and metadata to the index.
        """
        if not embedded_chunks:
            return

        vectors = np.array(
            [chunk["embedding"] for chunk in embedded_chunks],
            dtype="float32"
        )

        self.index.add(vectors)
        self.metadata.extend(embedded_chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Searches the index for top-k similar vectors.
        """
        query_vector = np.array([query_vector], dtype="float32")
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = self.metadata[idx].copy()
            result["score"] = float(score)
            results.append(result)

        return results

    def save(self, index_path: str, metadata_path: str):
        """
        Saves FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path: str, metadata_path: str):
        """
        Loads FAISS index and metadata from disk.
        """
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
