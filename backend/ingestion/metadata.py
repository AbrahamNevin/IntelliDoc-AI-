import re
from typing import List, Dict


class MetadataExtractor:
    """
    Extracts document-level and page-level metadata from text.
    """

    SECTION_KEYWORDS = {
        "summary": ["summary", "overview", "account summary"],
        "transactions": ["transaction", "activity", "account transactions"],
        "deposits": ["deposit", "credit"],
        "withdrawals": ["withdrawal", "debit", "atm"],
        "charges": ["fee", "charge", "service charge"],
        "checks": ["check", "cheque"],
    }

    def extract(self, pages: List[Dict]) -> Dict:
        """
        Main entry point for metadata extraction.
        """
        doc_metadata = {
            "account_id": None,
            "date_range": None,
            "document_type": "bank_statement",
        }

        enriched_pages = []

        for page in pages:
            text = page["text"].lower()

            # --- account / company ID ---
            if not doc_metadata["account_id"]:
                acc_match = re.search(r"account\s*#?\s*(\d{6,})", text)
                if acc_match:
                    doc_metadata["account_id"] = acc_match.group(1)

            # --- date range ---
            if not doc_metadata["date_range"]:
                date_match = re.search(
                    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}",
                    text
                )
                if date_match:
                    doc_metadata["date_range"] = date_match.group(0)

            # --- section detection ---
            section = self._detect_section(text)

            enriched_pages.append({
                **page,
                "section": section
            })

        return {
            "document_metadata": doc_metadata,
            "pages": enriched_pages
        }

    def _detect_section(self, text: str) -> str:
        """
        Detects document section based on keywords.
        """
        for section, keywords in self.SECTION_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return section
        return "unknown"
