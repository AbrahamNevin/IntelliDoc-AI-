import fitz  # PyMuPDF
import os
from typing import List, Dict


class PDFLoader:
    """
    Loads PDF documents and extracts page-wise text with metadata.
    """

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.lower().endswith(".pdf"):
            raise ValueError("Provided file is not a PDF")

        self.file_path = file_path
        self.doc_name = os.path.basename(file_path)

    def load(self) -> List[Dict]:
        pages = []

        try:
            pdf = fitz.open(self.file_path)

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if text:
                    pages.append({
                        "text": text,
                        "page": page_number,
                        "doc_name": self.doc_name
                    })

            pdf.close()

        except Exception as e:
            raise RuntimeError(f"Failed to load PDF: {str(e)}")

        return pages
