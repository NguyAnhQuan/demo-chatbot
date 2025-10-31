from .text_processor import TextProcessor
from typing import List, Dict, Any
import chromadb

class EmbeddingStorage(TextProcessor):
    """Stores embeddings in ChromaDB."""
    def __init__(self, db_path: str, collection_name: str):
        # Sử dụng PersistentClient thay vì Client với Settings cũ
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def process(self, formatted_chunks: List[Dict], embeddings: List[Any]) -> bool:
        try:
            ids = [chunk['id'] for chunk in formatted_chunks]
            metadatas = [chunk['metadata'] for chunk in formatted_chunks]
            documents = [chunk['content'] for chunk in formatted_chunks]
            embeddings = embeddings
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            return True
        except Exception as e:
            print(f"Error storing embeddings: {e}")
            return False
    