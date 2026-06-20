"""
Embedding models for document representation
"""


class BaseEmbedding:
    """Base class for embedding models"""
    
    def embed(self, texts):
        """
        Embed texts into vector space
        
        Args:
            texts: List of texts to embed
            
        Returns:
            list: List of embedding vectors
        """
        raise NotImplementedError


class MPNetEmbedding(BaseEmbedding):
    """MPNet embedding model"""
    
    def embed(self, texts):
        """Embed using MPNet model"""
        pass


class BGEEmbedding(BaseEmbedding):
    """BGE embedding model"""
    
    def embed(self, texts):
        """Embed using BGE model"""
        pass


class E5Embedding(BaseEmbedding):
    """E5 embedding model"""
    
    def embed(self, texts):
        """Embed using E5 model"""
        pass


class JinaEmbedding(BaseEmbedding):
    """Jina embedding model"""
    
    def embed(self, texts):
        """Embed using Jina model"""
        pass
