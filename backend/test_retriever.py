from backend.ingestion.pdf_loader import PDFLoader
from backend.ingestion.chunker import SemanticChunker
from backend.embeddings.embedder import Embedder
from backend.vectorstore.faiss_store import FAISSVectorStore
from backend.rag.retriever import Retriever

# Load document
loader = PDFLoader("data/raw_pdfs/sample.pdf")
pages = loader.load()

# Chunk
chunker = SemanticChunker()
chunks = chunker.chunk(pages)

# Embed
embedder = Embedder()
embedded_chunks = embedder.embed_chunks(chunks)

# Vector store
store = FAISSVectorStore(embedding_dim=384)
store.add(embedded_chunks)

# Retriever
retriever = Retriever(
    embedder=embedder,
    vector_store=store,
    top_k=3,
    min_score=0.25
)

query = "What is the beginning balance?"
retrieved_chunks = retriever.retrieve(query)

print(f"Retrieved {len(retrieved_chunks)} chunks\n")

context = retriever.build_context(retrieved_chunks)
print("Context sent to LLM:\n")
print(context[:1000])
