import chromadb
from typing import List, Dict, Optional
from dataclasses import dataclass

from .reranking import RetrievedDocument, RerankingPipeline

@dataclass
class ChromaConfig:
    """ Cấu hình cho kết nối ChromaDB"""
    collection_name: str
    host: str = "localhost"
    port: int = 8000
    persist_directory: Optional[str] = None

class ChromaRetriever:
    """ Lớp kết nối DB Chroma và reranking"""
    
    def __init__(self, config: ChromaConfig):
        self.config = config
        self.client = None
        self.collection = None
        self._connect()
    
    def _connect(self):
        """ Kết nối đến ChromaDB"""
        try:
            if self.config.persist_directory:
                self.client = chromadb.PersistentClient(path=self.config.persist_directory)
            else:
                self.client = chromadb.Client()
            
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name
            )
            print(f"Kết nối thành công đến ChromaDB collection: {self.config.collection_name}")
        except Exception as e:
            raise ConnectionError(f"Không thể kết nối đến ChromaDB: {e}")
        
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        """ Truy vấn các tài liệu từ ChromaDB và convert sang định dạng RetrievedDocument
        
        Args:
            query (str): Câu truy vấn
            top_k (int): Số lượng tài liệu cần truy xuất
        Returns:
            List[RetrievedDocument]: Danh sách tài liệu truy xuất
        """
        try:
            # Truy vấn ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            # Chuyển đổi kết quả sang định dạng RetrievedDocument
            retrieved_docs = []
            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []
            ids = results['ids'][0] if results['ids'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            
            for i, (doc, distance, doc_id) in enumerate(zip(documents, distances, ids)):
                # Convert khoảng cách thành điểm số (giả sử khoảng cách càng nhỏ điểm càng cao)
                similarity_score = 1 / (1 + distance) if distance is not None else 0.0
                
                # Tạo đối tượng RetrievedDocument
                retrieved_doc = RetrievedDocument(
                    document=doc,
                    retrieval_score=similarity_score,
                    doc_id=str(doc_id)
                )
                
                # Thêm metadata nếu có
                if metadatas and i < len(metadatas) and metadatas[i]:
                    retrieved_doc.metadata = metadatas[i]

                retrieved_docs.append(retrieved_doc)
            print(f"Truy xuất {len(retrieved_docs)} tài liệu từ ChromaDB")
            return retrieved_docs
        except Exception as e:
            raise RuntimeError(f"Lỗi khi truy vấn ChromaDB: {e}")
    
    def get_collection_stats(self) -> Dict:
        """ Lấy thông tin thống kê về collection hiện tại"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.config.collection_name,
                "document_count": count
            }
        except Exception as e:
            return {"error": str(e)}
        
class ChromaRerankRetriever:
    """ Kết hợp ChromaDB retrieval và reranking"""
    
    def __init__(self, chroma_config: ChromaConfig, reranking_methods: List[str] = None):
        self.retriever = ChromaRetriever(chroma_config)
        if reranking_methods is None:
            reranking_methods = ["cross_encoder"]
            
        self.reranking_pipeline = self._setup_reranking_pipeline(reranking_methods)
        self.default_method = reranking_methods[0] if reranking_methods else None
        
    def _setup_reranking_pipeline(self, methods: List[str]) -> RerankingPipeline:
        """ Thiết lập pipeline reranking với các phương pháp đã chọn"""
        pipeline = RerankingPipeline()
        
        for method in methods:
            if method == "cross_encoder":
                from .reranking import CrossEncoderReranker
                pipeline.add_reranker('cross_encoder', CrossEncoderReranker())
            elif method == "bi-encoder":
                from .reranking import BiEncoderReranker
                pipeline.add_reranker("bi-encoder", BiEncoderReranker())
            elif method == "hybrid":
                from .reranking import HybridReranker
                pipeline.add_reranker("hybrid", HybridReranker())
            else:
                print("Warning: Phương pháp reranking không hợp lệ, bỏ qua:", method)
                
        return pipeline
    
    def search(self, query: str,
               n_retrieve: int = 10,
               n_final: int = 3,
               rerank_method: str = None) -> List[RetrievedDocument]:
        """ pipeline tìm kiếm: truy xuất --> rerank --> trả về top kết quả
        Args:
            query (str): Câu truy vấn
            n_retrieve (int): Số tài liệu truy xuất ban đầu từ ChromaDB
            n_final (int): Số tài liệu cuối cùng sau khi rerank
            rerank_method (str): Phương pháp reranking sử dụng
        Returns:
            List[RetrievedDocument]: Danh sách top tài liệu sau khi rerank
        """
        
        # Bước 1: truy vấn tài liệu từ chromaDB
        retrieved_docs = self.retriever.retrieve(query, n_retrieve)
        
        if not retrieved_docs:
            print("Không tìm thấy tài liệu nào từ ChromaDB")
            return []
        
        # Bước 2: rerank và trả về kết quả
        method = rerank_method or self.default_method
        if method and method in self.reranking_pipeline.get_available_methods():
            reranked_docs = self.reranking_pipeline.rerank(
                query, retrieved_docs, method, top_k=n_final
            )
            print(f'Rerank với phương pháp: {method}, trả về {len(reranked_docs)} tài liệu')
        else:
            # Không rerank, trả về top n_final từ truy vấn ban đầu
            reranked_docs = sorted(retrieved_docs,
                                   key=lambda x: x.retrieval_score,
                                   reverse=True)[:n_final]
            print(f'Không sử dụng reranking, trả về {len(reranked_docs)} tài liệu từ truy vấn ban đầu')
        
        return reranked_docs
    
    def multi_method_search(self, query: str,
                            methods: List[str],
                            weights: List[float] = None,
                            n_retrieve: int = 10,
                            n_final: int = 3) -> List[RetrievedDocument]:
        """ Tìm kiếm với nhiều phương pháp reranking kết hợp"""
        
        # Retrieve
        retrieved_docs = self.retriever.retrieve(query, n_retrieve)
        
        if not retrieved_docs:
            print("Không tìm thấy tài liệu nào từ ChromaDB")
            return []
        
        # Multi-method rerank
        reranked_docs = self.reranking_pipeline.multi_rerank(
                query, retrieved_docs, methods, weights, top_k=n_final
            )
        print(f'Multi-method rerank với các phương pháp: {methods}, trả về {len(reranked_docs)} tài liệu')
        return reranked_docs
    
    def get_system_info(self) -> Dict:
        return {
            "chroma_stats": self.retriever.get_collection_stats(),
            "available_rerankers": self.reranking_pipeline.get_available_methods(),
            "default_method": self.default_method
        }
        
if __name__ == "__main__":
    # Ví dụ sử dụng ChromaRerankRetriever
    chroma_config = ChromaConfig(
        collection_name="documents",
        persist_directory="./chroma_db"
    )
    
    rerank_retriever = ChromaRerankRetriever(
        chroma_config,
        reranking_methods=["cross_encoder", "bi-encoder"]
    )
    
    query = "Thông tin của tôi"
    results = rerank_retriever.search(query, n_retrieve=10, n_final=3, rerank_method="cross_encoder")
    
    for i, doc in enumerate(results):
        print(f"Rank {i+1}: Content={doc.document[:300]}...")
    