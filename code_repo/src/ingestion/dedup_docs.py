# Deduping the ragbench documents for each subset of data
def dedup_docs(subset_data):
    raw_documents = []
    seen_docs = set()
    # for i in range(min(5, len(legal_data))):  # Indexing a small subset for demonstration
    for doc_text in subset_data['documents']:
        for docs in doc_text:
            if docs not in seen_docs:
                seen_docs.add(docs)
                # Wrap standard strings into LangChain Document objects
                raw_documents.append(docs)
    return raw_documents