# rewriting query for dense retrieval
from llm.model import llm
from prompts.rewrite_prompt import REWRITE_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

def rewrite_query(query: str) -> str:
    """
    Rewrites the query for dense retrieval using a language model.
    
    Args:
        query (str): The original query to be rewritten.
    
    Returns:
        str: The rewritten query.
    """
    logger.info("-"*50)
    logger.info("Rewriting query for dense retrieval...")
    logger.info(f"Original query: {query}")

    prompt = REWRITE_PROMPT.format(query=query)

    response = llm.invoke(prompt)
    rewitten_query = response.content.strip()
    logger.info(f"Rewritten query: {rewitten_query}")
    logger.info("-"*50)

    return rewitten_query
