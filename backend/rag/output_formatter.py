from .text_processor import TextProcessor
from typing import List, Dict
from datetime import datetime
import os
import uuid

class OutputFormatter(TextProcessor):
    """Formats chunks for storage or downstream use."""
    
    def __init__(self, id_prefix: str = "chunk"):
        """
        Initialize OutputFormatter.
        
        Args:
            id_prefix: Prefix for chunk IDs (default: "chunk")
        """
        self.id_prefix = id_prefix
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using simple word-based tokenization.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens (words)
        """
        # Simple word-based tokenization for Vietnamese text
        # Split by whitespace and count non-empty tokens
        tokens = text.split()
        return len([t for t in tokens if t.strip()])
    
    
    def process(self, chunks: List[str], source_file: str = None) -> List[Dict]:
        """
        Format chunks with metadata for storage or downstream use.
        
        Args:
            chunks: List of text chunks
            source_file: Optional source file path
            
        Returns:
            List of formatted chunk dictionaries with metadata
        """
        formatted_chunks = []
        
        for idx, chunk in enumerate(chunks):
            metadata = {}
            
            # Add source file metadata if provided
            if source_file:
                metadata['source_file'] = source_file
                metadata['file_name'] = os.path.basename(source_file)
            
            # Add chunk metrics
            metadata['length'] = len(chunk)
            metadata['token'] = self._count_tokens(chunk)
            metadata['created_at'] = datetime.now().isoformat()
            
            # Create formatted chunk
            formatted_chunks.append({
                'id': str(uuid.uuid4()),
                'content': chunk,
                'metadata': metadata
            })
            
        return formatted_chunks
