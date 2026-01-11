def build_prompt(context: str, question: str) -> str:
    """
    Builds a strict RAG prompt that prevents hallucinations
    and enforces citation-based answers.
    """

    system_instructions = (
        "You are a financial document analysis assistant.\n"
        "You MUST answer using ONLY the information provided in the context.\n"
        "If the answer is not explicitly present in the context, say:\n"
        "\"The information is not available in the provided document.\"\n\n"
        "Rules:\n"
        "- Do NOT use outside knowledge.\n"
        "- Do NOT guess or assume.\n"
        "- Cite page numbers for every factual claim.\n"
        "- Be concise and factual.\n\n"
        "Output Format:\n"
        "{\n"
        '  "answer": "<answer here>",\n'
        '  "citations": ["Page X", "Page Y"],\n'
        '  "confidence": <number between 0 and 100>\n'
        "}\n"
    )

    prompt = (
        system_instructions
        + "\n\nContext:\n"
        + context
        + "\n\nQuestion:\n"
        + question
    )

    return prompt
