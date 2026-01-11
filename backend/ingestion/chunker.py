import nltk
from typing import List, Dict

nltk.download("punkt", quiet=True)


class SemanticChunker:
    """
    Splits text into semantic chunks using sentence boundaries.
    """

    def __init__(self, max_words: int = 400, overlap: int = 50):
        self.max_words = max_words
        self.overlap = overlap

    def chunk(self, pages: List[Dict]) -> List[Dict]:
        """
        Takes page-wise text and returns semantic chunks with metadata.
        """
        chunks = []

        for page_data in pages:
            try:
                 sentences = nltk.sent_tokenize(page_data["text"])
            except LookupError:
                 sentences = page_data["text"].split(". ")

            current_chunk = []
            current_word_count = 0
            chunk_index = 0

            for sentence in sentences:
                words = sentence.split()
                word_count = len(words)

                if current_word_count + word_count > self.max_words:
                    chunk_text = " ".join(current_chunk)

                    chunks.append({
                        "chunk_text": chunk_text,
                        "page": page_data["page"],
                        "doc_name": page_data["doc_name"],
                        "chunk_id": f"{page_data['doc_name']}_{page_data['page']}_{chunk_index}"
                    })

                    # overlap handling
                    overlap_words = chunk_text.split()[-self.overlap:]
                    current_chunk = overlap_words.copy()
                    current_word_count = len(current_chunk)
                    chunk_index += 1

                current_chunk.append(sentence)
                current_word_count += word_count

            # add remaining text
            if current_chunk:
                chunks.append({
                    "chunk_text": " ".join(current_chunk),
                    "page": page_data["page"],
                    "doc_name": page_data["doc_name"],
                    "chunk_id": f"{page_data['doc_name']}_{page_data['page']}_{chunk_index}"
                })

        return chunks
