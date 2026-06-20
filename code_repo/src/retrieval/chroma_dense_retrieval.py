# Performing semantic search across the chunks of documents
import chromadb
from utils.logger import get_logger

logger = get_logger(__name__)

def chroma_dense_retriever(collection_name: str, query: str, embedding_function, top_k: int = 5, path: str = None):
    # Initialize the ChromaDB client
    client = chromadb.PersistentClient(path=path)

    # Get the collection
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)

    logger.info(f"Performing semantic search for query: '{query}' in collection: '{collection_name}' with top_k: {top_k}")

    # Perform semantic search
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    docs = []
    for idx in range(len(results['documents'][0])):
        docs.append(
            {
            "Id": results['ids'][0][idx],
            "text": results['documents'][0][idx],
            "metadata": results['metadata'][0][idx],
            "distance": results['distances'][0][idx]
        })
    logger.info(f"Retrieved {len(docs)} documents for the query. Dense Retrival Ends")
    logger.info("-"*50)
    return docs