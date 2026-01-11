import json
import os
from typing import Dict


class LLMGenerator:
    """
    Handles LLM calls and response parsing for RAG.
    Supports mock mode when API quota is unavailable.
    """

    def __init__(self, model: str = "gpt-4o-mini", mock: bool = False):
        self.model = model
        self.mock = mock

        if not mock:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not set in environment")
            self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> Dict:
        """
        Sends prompt to LLM or returns mock response.
        """
        if self.mock:
            return self._mock_response(prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            raw_output = response.choices[0].message.content.strip()
            return self._parse_response(raw_output)

        except Exception as e:
            return {
                "answer": "Error generating response.",
                "citations": [],
                "confidence": 0,
                "error": str(e)
            }

    def _parse_response(self, text: str) -> Dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "answer": text,
                "citations": [],
                "confidence": 50
            }

    def _mock_response(self, prompt: str) -> Dict:
        """
        Deterministic mock response for development/testing.
        """
        return {
            "answer": "The beginning balance is mentioned in the provided document context.",
            "citations": ["Page 1"],
            "confidence": 85
        }
