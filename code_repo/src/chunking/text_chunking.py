import re
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Loaded once per session — avoids reloading on every document
_semantic_model = None

def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        from config.settings import EMBED_MODEL
        _semantic_model = SentenceTransformer(EMBED_MODEL)
    return _semantic_model

def _cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def _split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+|\n', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def _semantic_chunk(text, breakpoint_threshold=0.4, max_chunk_chars=1000):
    """
    Splits text where cosine similarity between consecutive sentences drops
    below the threshold — keeps semantically related content in one chunk.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [text.strip()]

    model = _get_semantic_model()
    embeddings = model.encode(sentences, show_progress_bar=False)

    chunks = []
    current = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        too_large = (current_len + len(sentences[i])) > max_chunk_chars
        semantic_break = sim < breakpoint_threshold

        if semantic_break or too_large:
            chunk_text = ' '.join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current = [sentences[i]]
            current_len = len(sentences[i])
        else:
            current.append(sentences[i])
            current_len += len(sentences[i])

    if current:
        chunk_text = ' '.join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def chunk_documents(preprocessed_docs, chunk_size=1000, chunk_overlap=100, strategy="semantic"):
    """
    strategy:
        "recursive" — character-based splitting (fast, no model needed)
        "semantic"  — meaning-based splitting (slower, better quality)
    """
    if not preprocessed_docs or not preprocessed_docs.strip():
        return []

    if strategy == "semantic":
        return _semantic_chunk(preprocessed_docs, max_chunk_chars=chunk_size)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(preprocessed_docs)
