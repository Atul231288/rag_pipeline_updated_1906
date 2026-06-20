# Chunking the preprocessed text data into smaller chunks to add to the vector database. This is done to ensure that the chunks are of manageable size for the vector database and to improve the retrieval performance.
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(preprocessed_docs, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(preprocessed_docs)
    return chunks
