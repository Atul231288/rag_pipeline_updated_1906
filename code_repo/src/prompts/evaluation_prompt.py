RAG_EVALUATION_PROMPT = """
You are an expert RAG evaluator.
Question:
{question}
Retrieved Context:
{contexts}
Generated Answer:
{answer}

Evaluate the answer.
Return:

1. total_chunks
   Total retrieved chunks.

2. relevant_chunks
   Retrieved chunks that are relevant to answering the question.

3. total_context_facts
   Important facts present in the retrieved context.

4. used_context_facts
   Context facts actually used in the generated answer.

5. total_relevant_facts
   Relevant facts required for a complete answer.

6. covered_relevant_facts
   Relevant facts covered in the answer.

7. total_answer_statements
   Number of factual statements in the answer.

8. supported_answer_statements
   Statements grounded in retrieved context.

Provide concise reasoning.
"""