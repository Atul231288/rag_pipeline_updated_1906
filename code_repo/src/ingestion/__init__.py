"""
Document ingestion module for PDF, DOCX, and TXT files
"""

class DocumentIngestionError(Exception):
    """Raised when document ingestion fails"""
    pass


class PDFLoader:
    """Load and parse PDF documents"""
    
    def __init__(self):
        pass
    
    def load(self, file_path):
        """Load PDF from file path"""
        raise NotImplementedError


class DocxLoader:
    """Load and parse DOCX documents"""
    
    def __init__(self):
        pass
    
    def load(self, file_path):
        """Load DOCX from file path"""
        raise NotImplementedError


class TextLoader:
    """Load and parse TXT documents"""
    
    def __init__(self):
        pass
    
    def load(self, file_path):
        """Load TXT from file path"""
        raise NotImplementedError
