from backend.rag.prompt import build_prompt

context = """
[Document: sample.pdf | Page: 1 | Score: 0.78]
Beginning balance on Oct 10: $5,000
"""

question = "What is the beginning balance?"

prompt = build_prompt(context, question)

print(prompt)
