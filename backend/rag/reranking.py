from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievedDocument:
    """Data class cho các tài liệu được truy xuất cùng điểm số """
    
    def __init__(self, document: str, retrieval_score: float=0.0, doc_id: str=None):
        self.document = document
        self.retrieval_score = retrieval_score
        self.doc_id = doc_id
        self.rerank_scores = {}  # Lưu điểm số từ các mô hình reranker khác nhau

    def add_rerank_score(self, method: str, score: float):
        """ Thêm điểm số reranking từ method đặc biệt """
        self.rerank_scores[method] = score
        
    def get_best_score(self, method: str=None) -> float:
        """ Lấy điểm số tốt nhất có sẵn """
        if method and method in self.rerank_scores:
            return self.rerank_scores[method]
        elif self.rerank_scores:
            return max(self.rerank_scores.values())
        return self.retrieval_score
    
class BaseReranker(ABC):
    """Base class cho các mô hình reranker."""
    
    def __init__(self, model_name: str=None):
        self.model_name = model_name
        self.model = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self):
        """ Tải mô hình reranker """
        pass
    
    @abstractmethod
    def rerank(self, query: str, documents: List[RetrievedDocument], top_k: int=None) -> List[RetrievedDocument]:
        """ Rerank các tài liệu dựa trên truy vấn """
        pass
    
    def _ensure_model_loaded(self):
        if not self._is_loaded:
            self.load_model()
            self._is_loaded = True
    
class CrossEncoderReranker(BaseReranker):
    """Reranker sử dụng mô hình Cross-Encoder."""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        super().__init__(model_name)
    
    def load_model(self):
        """ Tải mô hình Cross-Encoder """
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Đã tải mô hình Cross-Encoder: {self.model_name}")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình Cross-Encoder: {e}")
            raise e
    
    def rerank(self, query: str, documents: List[RetrievedDocument], 
               top_k: int=None) -> List[RetrievedDocument]:
        """ Rerank sử dụng điểm cross-encoder """
        self._ensure_model_loaded()
        
        if not documents:
            return documents
        
        # Chuẩn bị cặp query-document
        pairs = [(query, doc.document) for doc in documents]
        
        # Tính điểm số
        scores = self.model.predict(pairs)
        
        # Cập nhật điểm số
        for doc, score in zip(documents, scores):
            doc.add_rerank_score('cross-encoder', score)
        
        # Sắp xếp tài liệu theo điểm rerank
        reranked_docs = sorted(documents, 
                              key=lambda x: x.get_best_score(self.get_method_name()), 
                              reverse=True)
        
        if top_k:
            return reranked_docs[:top_k]
        
        return reranked_docs
    
    def get_method_name(self) -> str:
        return 'cross-encoder'

class BiEncoderReranker(BaseReranker):
    """Reranker sử dụng Bi-Encoder để tính điểm ban đầu và Cross-Encoder để rerank."""
    
    def __init__(self, bi_encoder_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                 bi_encoder_weight: float = 0.5, cross_encoder_weight: float = 0.5):
        super().__init__(bi_encoder_model)
        self.cross_encoder_model_name = cross_encoder_model
        self.bi_encoder_weight = bi_encoder_weight
        self.cross_encoder_weight = cross_encoder_weight
        self.bi_encoder = None
        self.cross_encoder = None
    
    def load_model(self):
        """ Tải mô hình Bi-Encoder và Cross-Encoder """
        try:
            self.bi_encoder = SentenceTransformer(self.model_name)
            self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)
            logger.info(f"Đã tải mô hình Bi-Encoder: {self.model_name} và Cross-Encoder: {self.cross_encoder_model_name}")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình Bi-Encoder hoặc Cross-Encoder: {e}")
            raise e
    
    def _calculate_bi_encoder_scores(self, query: str, documents: List[str]) -> List[float]:
        """ Tính điểm Bi-Encoder """
        query_emb = self.bi_encoder.encode([query], convert_to_tensor=True)
        doc_embs = self.bi_encoder.encode(documents, convert_to_tensor=True)
        similarities = self.bi_encoder.similarity(query_emb, doc_embs)
        return similarities[0].tolist()
    
    def _calculate_cross_encoder_scores(self, query: str, documents: List[str]) -> List[float]:
        """ Tính điểm Cross-Encoder """
        pairs = [(query, doc) for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        return scores.tolist()
    
    def rerank(self, query: str, documents: List[RetrievedDocument], top_k: int=None) -> List[RetrievedDocument]:
        """ Rerank sử dụng kết hợp Bi-Encoder và Cross-Encoder """
        self._ensure_model_loaded()
        
        if not documents:
            return documents
        
        docs_text = [doc.document for doc in documents]
        
        # Tính điểm Bi-Encoder
        bi_encoder_scores = self._calculate_bi_encoder_scores(query, docs_text)
        
        # Tính điểm Cross-Encoder
        cross_encoder_scores = self._calculate_cross_encoder_scores(query, docs_text)
        
        # Chuẩn hóa điểm số
        bi_encoder_scores = np.array(bi_encoder_scores)
        cross_encoder_scores = np.array(cross_encoder_scores)
        
        if len(bi_encoder_scores) > 1:
            bi_encoder_scores = (bi_encoder_scores - bi_encoder_scores.min()) / (bi_encoder_scores.max() - bi_encoder_scores.min() + 1e-8)
        if len(cross_encoder_scores) > 1:
            cross_encoder_scores = (cross_encoder_scores - cross_encoder_scores.min()) / (cross_encoder_scores.max() - cross_encoder_scores.min() + 1e-8)
        
        # Kết hợp điểm số
        combined_scores = (self.bi_encoder_weight * bi_encoder_scores) + (self.cross_encoder_weight * cross_encoder_scores)
        
        # Cập nhật điểm số
        for doc, score in zip(documents, combined_scores):
            doc.add_rerank_score('bi-encoder', score)
        
        # Sắp xếp tài liệu theo điểm rerank
        reranked_docs = sorted(documents,
                              key=lambda x: x.get_best_score(self.get_method_name()),
                              reverse=True)
        
        return reranked_docs[:top_k] if top_k else reranked_docs
    
    def get_method_name(self) -> str:
        return 'bi-encoder'

class HybridReranker(BaseReranker):
    """Hybrid reranking kết hợp BM25 (sparse) và semantic (dense) retrieval."""
    
    def __init__(self, semantic_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 bm25_weight: float = 0.3, semantic_weight: float = 0.7):
        super().__init__(semantic_model)
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.tfidf_vectorizer = None
        self.semantic_model = None
        self.corpus_dfidf = None
    
    def load_model(self):
        """ Tải mô hình BM25 và mô hình semantic"""
        try:
            self.semantic_model = SentenceTransformer(self.model_name)
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
            logger.info(f'Đã tải mô hình hybrid: {self.model_name} + TF-IDF')
        except Exception as e:
            logger.error(f'Lỗi khi tải mô hình hybrid: {e}')
            raise e
    
    def _fit_tfidf(self, documents: List[str]):
        """ Fit TF-IDF trên tập tài liệu """
        self.corpus_dfidf = self.tfidf_vectorizer.fit_transform(documents)
        
    def _calculate_bm25_scores(self, query: str, documents: List[str]) -> List[float]:
        """ Tính điểm số BM25 dùng TF-IDF"""
        if self.corpus_dfidf is None:
            self._fit_tfidf(documents)
            
        query_tfidf = self.tfidf_vectorizer.transform([query])
        scores = (query_tfidf * self.corpus_dfidf.T).toarray()[0]
        return scores.tolist()
    
    def _calculate_semantic_scores(self, query: str, documents: List[str]) -> List[float]:
        """Tính điểm tương đồng ngữ nghĩa"""
        query_emb = self.semantic_model.encode([query])
        doc_embs = self.semantic_model.encode(documents)
        
        similarities = self.semantic_model.similarity(query_emb, doc_embs)
        return similarities[0].tolist()
    
    def rerank(self, query: str, documents: List[RetrievedDocument], top_k = None) -> List[RetrievedDocument]:
        """ Rerank sử dụng điểm hybrid BM25 + semantic"""
        self._ensure_model_loaded()
        
        if not documents:
            return documents
        
        docs_text = [doc.document for doc in documents]
        
        # Tính điểm số BM25 và semantic
        bm25_scores = self._calculate_bm25_scores(query, docs_text)
        semantic_scores = self._calculate_semantic_scores(query, docs_text)
        
        # Chuẩn hóa điểm số
        bm25_scores = np.array(bm25_scores)
        semantic_scores = np.array(semantic_scores)
        
        if len(bm25_scores) > 1:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        if len(semantic_scores) > 1:
            semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
        
        # Kết hợp điểm số
        hybrid_scores = (self.bm25_weight * bm25_scores) + (self.semantic_weight * semantic_scores)
        
        # Cập nhật điểm số
        for doc, score in zip(documents, hybrid_scores):
            doc.add_rerank_score('hybrid', score)
        
        # Sắp xếp tài liệu theo điểm rerank
        reranked_docs = sorted(documents,
                                key=lambda x: x.get_best_score(self.get_method_name()), 
                                reverse=True)
        
        return reranked_docs[:top_k] if top_k else reranked_docs
    
    def get_method_name(self) -> str:
        return 'hybrid'
    
class RerankingPipeline:
    """Pipeline để quản lý nhiều phương pháp reranking."""
    
    def __init__(self):
        self.rerankers = {}
        
    def add_reranker(self, name: str, reranker: BaseReranker):
        """ Thêm một mô hình reranker vào pipeline """
        self.rerankers[name] = reranker
        logger.info(f"Đã thêm reranker: {name}")
    
    def remove_reranker(self, name: str):
        """ Xóa một mô hình reranker khỏi pipeline """
        if name in self.rerankers:
            del self.rerankers[name]
            logger.info(f"Đã xóa reranker: {name}")
        else:
            logger.warning(f"Reranker {name} không tồn tại trong pipeline.")
    
    def rerank(self, query: str, documents: List[RetrievedDocument],
               method: str, top_k: int=None) -> List[RetrievedDocument]:
        """ Rerank sử dụng phương pháp đã chọn """
        if method not in self.rerankers:
            logger.warning(f"Phương pháp rerank {method} không tồn tại. Tồn tại các phương pháp: {list(self.rerankers.keys())}")
        
        return self.rerankers[method].rerank(query, documents, top_k)
    
    def multi_rerank(self, query: str, documents: List[RetrievedDocument],
                     methods: List[str], weights: Optional[List[float]]=None,
                     top_k: int=None) -> List[RetrievedDocument]:
        """ Rerank sử dụng nhiều phương pháp và kết hợp điểm số """
        
        if weights is None:
            weights = [1.0] * len(methods)
        
        if len(methods) != len(weights):
            raise ValueError("Số lượng phương pháp và trọng số phải bằng nhau.")

        # Áp dụng từng phương pháp rerank
        for method in methods:
            if method not in self.rerankers:
                logger.warning(f"Phương pháp rerank {method} không tồn tại.")
                continue
            self.rerankers[method].rerank(query, documents)
        
        # Kết hợp điểm số từ các phương pháp
        for doc in documents:
            combined_score = 0.0
            total_weight = 0.0
            
            for i, method in enumerate(methods):
                if method in doc.rerank_scores:
                    # Chuan hóa điểm số
                    method_scores = [d.rerank_scores.get(method, 0) for d in documents if method in d.rerank_scores]
                    if len(method_scores) > 1:
                        min_score, max_score = min(method_scores), max(method_scores)
                        normalized_score = (doc.rerank_scores[method] - min_score) / (max_score - min_score + 1e-8)
                    else:
                        normalized_score = 1.0
                    
                    w = weights[i] if weights else 1.0
                    combined_score += w * normalized_score
                    total_weight += w
            
            if total_weight > 0:
                doc.add_rerank_score('combined', combined_score / total_weight)
        
        # Sắp xếp tài liệu theo điểm số kết hợp
        reranked_docs = sorted(documents, 
                              key=lambda x: x.get_best_score('combined'), 
                              reverse=True)
        
        return reranked_docs[:top_k] if top_k else reranked_docs
    
    def get_available_methods(self) -> List[str]:
        """ Lấy danh sách các phương pháp reranking có sẵn """
        return list(self.rerankers.keys())

if __name__ == "__main__":
    # Import ChromaDB configuration
    from chroma_retrieval import ChromaConfig, ChromaRerankRetriever
    
    # Setup ChromaDB configuration
    chroma_config = ChromaConfig(
        collection_name="documents",
        persist_directory="./chroma_db"  # or None for in-memory
    )
    
    # Create RAG system with specific reranking methods
    chroma_rerank_retriever = ChromaRerankRetriever(
        chroma_config=chroma_config,
        reranking_methods=["cross_encoder", "bi-encoder", "hybrid"]
    )
    
    query = "Bì thư trung ương đoàn là ai?"
    
    # Example 1: Simple search with default reranking
    results = chroma_rerank_retriever.search(query, n_retrieve=10, n_final=3)
    
    print("Single Search Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. Score: {doc.get_best_score():.4f}")
        print(f"   Doc ID: {doc.doc_id}")
        print(f"   Text: {doc.document[:350]}...")
        print()
        
    print("-" * 50)
    
    # # Example 2: Multi-method search
    # multi_results = chroma_rerank_retriever.multi_method_search(
    #     query=query,
    #     methods=["cross_encoder", "hybrid"],
    #     weights=[0.7, 0.3],
    #     n_final=3
    # )
    
    # print("Multi-method Search Results:")
    # for i, doc in enumerate(multi_results, 1):
    #     print(f"{i}. Combined Score: {doc.get_best_score('combined'):.4f}")
    #     print(f"   Doc ID: {doc.doc_id}")
    #     print(f"   Text: {doc.document}...")
    #     print()
    
    # print("-" * 50)
    
    # # Example 3: System information
    # info = chroma_rerank_retriever.get_system_info()
    # print("System Info:", info) 