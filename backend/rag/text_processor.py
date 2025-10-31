from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class TextProcessor(ABC):
    """Base class for all processing modules."""
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass