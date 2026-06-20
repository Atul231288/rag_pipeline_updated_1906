"""
Reranking models for result refinement
"""


class BaseReranker:
    """Base class for reranking models"""
    
    def rerank(self, query, documents, top_k=5):
        """Rerank documents based on query"""
        raise NotImplementedError


class MonoT5(BaseReranker):
    """MonoT5 reranker"""
    
    def rerank(self, query, documents, top_k=5):
        """Rerank using MonoT5"""
        pass


class MonoBERT(BaseReranker):
    """MonoBERT reranker"""
    
    def rerank(self, query, documents, top_k=5):
        """Rerank using MonoBERT"""
        pass


class RankLLaMA(BaseReranker):
    """RankLLaMA reranker"""
    
    def rerank(self, query, documents, top_k=5):
        """Rerank using RankLLaMA"""
        pass
