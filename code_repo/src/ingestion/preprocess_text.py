# preprocessing the text data before adding to the vector database. This includes removing the new line characters greater than '\n' from the documents data.
# def preprocess_documents(documents):
#     preprocessed_docs = []
#     for doc in documents:
#         # Remove new line characters
#         # Replace multiple new lines with a single one
#         cleaned_doc = doc.replace('\n\n', '\n') 
#         preprocessed_docs.append(cleaned_doc)

#     return '\n'.join(preprocessed_docs)
import json

def _table_to_text(table: list) -> str:
    """Convert a JSON list-of-lists table into readable sentences."""
    if not table or not isinstance(table[0], list):
        return str(table)
    headers = table[0]
    rows = table[1:]
    lines = []
    for row in rows:
        label = row[0] if row else ""
        parts = []
        for col_idx, value in enumerate(row[1:], start=1):
            if col_idx < len(headers) and value not in ("", None):
                parts.append(f"{headers[col_idx]}: {value}")
        if parts:
            lines.append(f"{label} — " + ", ".join(parts))
    return "\n".join(lines)


def _convert_tables(text: str) -> str:
    """Find JSON tables embedded in text and replace them with readable prose."""
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list):
            return _table_to_text(parsed)
    except (json.JSONDecodeError, ValueError):
        pass
    return text


def preprocess_documents(document: str) -> str:
    document = document.replace('\n\n', '\n')
    document = _convert_tables(document)
    return document