from .text_processor import TextProcessor
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Any

class EmbeddingGenerator(TextProcessor):
    """Generates embeddings for text segments."""
    def __init__(self, model_name: str = "keepitreal/vietnamese-sbert", device: str = None):
        # Nếu không chỉ định device, tự chọn GPU nếu có, ngược lại dùng CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)

    def process(self, segments: List[str]) -> List[Any]:
        # Gọi encode, đảm bảo chạy trên GPU nếu có
        return self.model.encode(segments, device=self.device, show_progress_bar=False)
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()