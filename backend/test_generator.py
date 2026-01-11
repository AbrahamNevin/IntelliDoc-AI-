from backend.ingestion.pdf_loader import PDFLoader
from backend.ingestion.chunker import SemanticChunker
from backend.embeddings.embedder import Embedder
from backend.vectorstore.faiss_store import FAISSVectorStore
from backend.rag.retriever import Retriever
from backend.rag.prompt import build_prompt
from backend.rag.generator import LLMGenerator

# Load document
loader = PDFLoader("data/raw_pdfs/sample.pdf")
pages = loader.load()

# Chunk
chunker = SemanticChunker()
chunks = chunker.chunk(pages)

# Embed
embedder = Embedder()
embedded_chunks = embedder.embed_chunks(chunks)

# Store
store = FAISSVectorStore(embedding_dim=384)
store.add(embedded_chunks)

# Retrieve
retriever = Retriever(embedder, store)
query = "What is the beginning balance?"
retrieved = retriever.retrieve(query)
context = retriever.build_context(retrieved)

# Prompt
prompt = build_prompt(context, query)

# Generate
generator = LLMGenerator(mock=True)
response = generator.generate(prompt)

print("LLM Response:")
print(response)
