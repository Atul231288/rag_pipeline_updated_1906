def determine_strategy(classification):
    retrieval_required = classification["retrieval_required"]
    if not retrieval_required:
        return {
            "retrieve": False,
            "strategy": "direct_llm"
        }

    return {
        "retrieve": True,
        "strategy": "rag"
    }