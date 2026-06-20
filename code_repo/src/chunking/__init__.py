"""
Text chunking strategies for document processing
"""


class BaseChunker:
    """Base class for text chunking strategies"""
    
    def chunk(self, text, chunk_size=512, overlap=50):
        """
        Chunk text into smaller pieces
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            list: List of text chunks
        """
        raise NotImplementedError


class FixedChunker(BaseChunker):
    """Fixed-size chunking strategy"""
    
    def chunk(self, text, chunk_size=512, overlap=50):
        """Implement fixed-size chunking"""
        pass


class SemanticChunker(BaseChunker):
    """Semantic-aware chunking strategy"""
    
    def chunk(self, text, chunk_size=512, overlap=50):
        """Implement semantic chunking"""
        pass


class SlidingWindow(BaseChunker):
    """Sliding window chunking strategy"""
    
    def chunk(self, text, chunk_size=512, overlap=50):
        """Implement sliding window chunking"""
        pass


class Small2Big(BaseChunker):
    """Small-to-big chunking strategy"""
    
    def chunk(self, text, chunk_size=512, overlap=50):
        """Implement small-to-big chunking"""
        pass
