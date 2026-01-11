from backend.ingestion.pdf_loader import PDFLoader
from backend.ingestion.chunker import SemanticChunker
from backend.embeddings.embedder import Embedder
from backend.vectorstore.faiss_store import FAISSVectorStore

# Load and process document
loader = PDFLoader("data/raw_pdfs/sample.pdf")
pages = loader.load()

chunker = SemanticChunker()
chunks = chunker.chunk(pages)

embedder = Embedder()
embedded_chunks = embedder.embed_chunks(chunks)

# Create FAISS store
store = FAISSVectorStore(embedding_dim=384)
store.add(embedded_chunks)

# Query
query = "What is the ending balance?"
query_vector = embedder.embed_query(query)

results = store.search(query_vector, top_k=3)

print("Top results:")
for r in results:
    print(f"Score: {r['score']:.3f} | Page: {r['page']} | Section: {r.get('section')}")
    print(r["chunk_text"][:200])
    print("-" * 50)
