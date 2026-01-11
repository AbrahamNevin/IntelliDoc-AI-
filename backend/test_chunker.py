from backend.ingestion.pdf_loader import PDFLoader
from backend.ingestion.chunker import SemanticChunker

loader = PDFLoader("data/raw_pdfs/sample.pdf")
pages = loader.load()

print(f"Pages extracted: {len(pages)}")

if pages:
    print("Sample page text (first 300 chars):")
    print(pages[0]["text"][:300])

chunker = SemanticChunker(max_words=400, overlap=50)
chunks = chunker.chunk(pages)

print(f"Total chunks created: {len(chunks)}")
