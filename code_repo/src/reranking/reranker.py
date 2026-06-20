# reranking semantic search results using a cross-encoder model
from sentence_transformers import CrossEncoder
from config.settings import RERANK_MODEL, TOP_K_RERANK
from utils.logger import get_logger

logger = get_logger(__name__)

reranker = CrossEncoder(RERANK_MODEL)

def rerank(query, retrieved_docs_dict):
    logger.info("-"*50)
    logger.info("Reranking retrieved documents using cross-encoder model...")
    logger.info(f"Number of retrieved documents: {len(retrieved_docs_dict['documents'][0])}. Returning top {TOP_K_RERANK} reranked documents.")
    docs = retrieved_docs_dict['documents'][0]
    ids = retrieved_docs_dict['ids'][0]
    # create pairs of query and retrieved documents for reranking
    pairs = [(query, doc) for doc in docs]
    # compute relevance scores for each pair using the cross-encoder model
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, ids, docs), reverse=True)
    # sort the retrieved documents based on the relevance scores in descending order
    reranked_docs = [doc for score,id, doc in ranked[:TOP_K_RERANK]]
    for score, doc_id, doc in ranked[:TOP_K_RERANK]:
        logger.info(f"Score: {score:.4f}, Doc ID: {doc_id}, Doc: {doc[:50]}...")
    logger.info("-"*50)
    return reranked_docs


