# generating Context relevance, context utilization, completeness and adherence scores for evaluation of RAG pipeline
from typing import Literal
from pydantic import BaseModel, Field
from prompts.evaluation_prompt import RAG_EVALUATION_PROMPT
from evaluation.evaluation_schema import RAGEvaluationCounts
from utils.logger import get_logger
from llm.model import llm
logger = get_logger(__name__)

def calculate_metrics(result):
    return {
        "context_relevance":result.relevant_chunks / max(result.total_chunks, 1),
        "context_utilization": result.used_context_facts / max(result.total_context_facts, 1),
        "completeness": result.covered_relevant_facts / max(result.total_relevant_facts, 1),
        "adherence": result.supported_answer_statements / max(result.total_answer_statements, 1)
    }

def evaluate_rag_response(question: str,contexts: list[str],answer: str):
    prompt = RAG_EVALUATION_PROMPT.format(
        question=question,
        contexts="\n\n".join(contexts),
        answer=answer
    )
    judge_llm = llm.with_structured_output(RAGEvaluationCounts)
    result = judge_llm.invoke(prompt)
    return result