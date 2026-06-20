# preprocessing the text data before adding to the vector database. This includes removing the new line characters greater than '\n' from the documents data.
# def preprocess_documents(documents):
#     preprocessed_docs = []
#     for doc in documents:
#         # Remove new line characters
#         # Replace multiple new lines with a single one
#         cleaned_doc = doc.replace('\n\n', '\n') 
#         preprocessed_docs.append(cleaned_doc)

#     return '\n'.join(preprocessed_docs)
def preprocess_documents(document):
    return document.replace('\n\n', '\n') 