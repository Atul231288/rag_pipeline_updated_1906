from pydantic import BaseModel, Field
class RAGEvaluationCounts(BaseModel):
    # Context Relevance
    total_chunks: int = Field(description="Total retrieved chunks")
    relevant_chunks: int = Field(description="Retrieved chunks relevant to the question")
    # Context Utilization
    total_context_facts: int = Field(description="Total important facts present in retrieved context")
    used_context_facts: int = Field(description="Facts from context used in the answer")
    # Completeness
    total_relevant_facts: int = Field(description="Relevant facts needed to answer the question")
    covered_relevant_facts: int = Field(description="Relevant facts covered in the answer")
    # Adherence
    total_answer_statements: int = Field(description="Total factual statements in the answer")
    supported_answer_statements: int = Field(description="Answer statements supported by retrieved context")
    # Reasoning of the evaluation done
    reasoning: str = Field(description="Short explanation of the evaluation")