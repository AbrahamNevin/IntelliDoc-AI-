from backend.ingestion.pdf_loader import PDFLoader
from backend.ingestion.metadata import MetadataExtractor

loader = PDFLoader("data/raw_pdfs/sample.pdf")
pages = loader.load()

extractor = MetadataExtractor()
result = extractor.extract(pages)

print("Document Metadata:")
print(result["document_metadata"])

print("\nPage Sections:")
for p in result["pages"]:
    print(f"Page {p['page']} → Section: {p['section']}")
