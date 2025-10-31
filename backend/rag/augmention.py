from .chroma_retrieval import ChromaConfig
from .chroma_retrieval import ChromaRerankRetriever

class Augmention:
    def __init__(self,  
                 db_path="./chromaDB", 
                 collection_name="documents", 
                 reranking_methods=["cross_encoder", "bi-encoder"]):
        self.chroma_config = ChromaConfig(
            persist_directory = db_path,
            collection_name = collection_name
        )
        self.retriever = ChromaRerankRetriever(
            chroma_config=self.chroma_config,
            reranking_methods = reranking_methods
        )

    
    def get_augmented_response(self, query: str, top_k: int = 5):
        resuilt = self.retriever.search(query, n_retrieve=10, n_final=top_k)
        return resuilt
    
if __name__ == "__main__":
    augmention = Augmention()
    query = "Tên trường đại học là gì?"
    augmented_response = augmention.get_augmented_response(query, top_k=3)
    
    print("Augmented Response:")
    print(augmented_response[0].document)