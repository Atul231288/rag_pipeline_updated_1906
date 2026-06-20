from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from config.settings import EMBED_MODEL


def get_embedding_function():
    """ChromaDB-compatible embedding function used when building/querying the vector DB."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def get_embedder():
    """Raw SentenceTransformer model — use this when you want to inspect embedding vectors directly."""
    return SentenceTransformer(EMBED_MODEL)
