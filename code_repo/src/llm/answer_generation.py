from llm.model import llm
from prompts.answer_prompt import ANSWER_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

def generate_answer(query, context):
    logger.info("Generating answer using LLM...")
    prompt = ANSWER_PROMPT.format(query=query, context=context)
    response = llm.invoke(prompt)
    logger.info("Answer generated successfully.")
    logger.info("-" * 50)
    return response.content.strip()
