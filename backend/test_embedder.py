from backend.ingestion.pdf_loader import PDFLoader
from backend.ingestion.chunker import SemanticChunker
from backend.embeddings.embedder import Embedder

loader = PDFLoader("data/raw_pdfs/sample.pdf")
pages = loader.load()

chunker = SemanticChunker()
chunks = chunker.chunk(pages)

embedder = Embedder()
embedded_chunks = embedder.embed_chunks(chunks)

print(f"Total embedded chunks: {len(embedded_chunks)}")
print("Embedding shape:", embedded_chunks[0]["embedding"].shape)
