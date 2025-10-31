from .document_preprocessor import DocumentPreprocessor
from .text_splitter import TextSplitter
from .output_formatter import OutputFormatter
from .embedding_generator import EmbeddingGenerator
from .embedding_storage import EmbeddingStorage
import os
from django.conf import settings

class Chunking:
    def chunking(self,file_path, 
                 chunk_size, 
                 chunk_overlap, 
                 model_name, 
                 db_path="./chromaDB", 
                 collection_name="documents",
                 ):
        try:
            doc = DocumentPreprocessor().process(file_path)
            text = TextSplitter(chunk_size,chunk_overlap).process(doc)
            emb = EmbeddingGenerator(model_name).process(text)
            formatted_chunks = OutputFormatter().process(text, source_file=file_path)
            storage = EmbeddingStorage(db_path, collection_name).process(formatted_chunks, emb)
            
            if storage:
                return f"Successfully stored {len(formatted_chunks)} chunks in ChromaDB"
            else:
                return "Failed to store embeddings"
        except Exception as e:
            print(f"error during chunking process {e}")
            return str(e)

# if __name__ == "__main__":
#     chunker = Chunking()
#     try:
#         result = chunker.chunking("C:\\Users\\hq151\\Downloads\\form_tien_do.docx")
#         print("Chunking result:", result)
#     except Exception as e:
#         print(f"An error occurred: {e}")