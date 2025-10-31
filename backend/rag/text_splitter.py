from .text_processor import TextProcessor
from typing import List, Dict, Any
import re
import google.generativeai as genai
import os
from dotenv import load_dotenv


class SplitHelper:
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap


    def _recursive_split_helper(self, text: str, separators: List[str], separator_index: int) -> List[str]:
        """Helper function for recursive splitting."""
        if not text or len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        # If we've exhausted all separators, do character-level splitting
        if separator_index >= len(separators):
            return self._split_by_character_simple(text)
        
        current_separator = separators[separator_index]
        
        # If current separator is empty string (character level)
        if current_separator == "":
            return self._split_by_character_simple(text)
        
        # Try to split by current separator
        if current_separator in text:
            splits = text.split(current_separator)
            
            # Reconstruct chunks with separator (except for the last piece)
            chunks = []
            current_chunk = ""
            
            for i, split in enumerate(splits):
                # Add separator back (except for last split)
                piece = split + (current_separator if i < len(splits) - 1 else "")
                
                # If adding this piece would exceed chunk size
                if len(current_chunk) + len(piece) > self.chunk_size:
                    # If current chunk is not empty, add it to results
                    if current_chunk.strip():
                        chunks.extend(self._finalize_chunk(current_chunk.strip()))
                    
                    # If the piece itself is too large, recursively split it
                    if len(piece) > self.chunk_size:
                        chunks.extend(self._recursive_split_helper(piece, separators, separator_index + 1))
                        current_chunk = ""
                    else:
                        current_chunk = piece
                else:
                    current_chunk += piece
            
            # Add remaining chunk
            if current_chunk.strip():
                chunks.extend(self._finalize_chunk(current_chunk.strip()))
            
            # Apply overlap if we have multiple chunks
            if len(chunks) > 1:
                return self._apply_overlap_to_chunks(chunks)
            else:
                return chunks
        
        # If separator not found, try next separator
        return self._recursive_split_helper(text, separators, separator_index + 1)

    def _split_by_character_simple(self, text: str) -> List[str]:
        """Simple character-level splitting when all other methods fail."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk = text[i:i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        
        return chunks

    def _finalize_chunk(self, chunk: str) -> List[str]:
        """Finalize a chunk, splitting further if needed."""
        if len(chunk) <= self.chunk_size:
            return [chunk]
        
        # If still too large, split by character
        return self._split_by_character_simple(chunk)

    def _apply_overlap_to_chunks(self, chunks: List[str]) -> List[str]:
        """Apply overlap between chunks."""
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk remains as is
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]
                
                # Get last N characters/words from previous chunk for overlap
                overlap_text = self._get_overlap_text(prev_chunk, self.overlap)
                
                # Combine overlap with current chunk
                if overlap_text and not chunk.startswith(overlap_text):
                    overlapped_chunk = overlap_text + " " + chunk
                else:
                    overlapped_chunk = chunk
                
                overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk."""
        if not text or overlap_size <= 0:
            return ""
        
        # Try to get overlap by words first
        words = text.split()
        if len(words) <= overlap_size:
            return text
        
        # Get last 'overlap_size' words
        overlap_words = words[-overlap_size:]
        return " ".join(overlap_words)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling Vietnamese punctuation."""
        # Vietnamese sentence endings
        sentence_endings = r'[.!?。！？]+'
        
        # Split by sentence endings followed by whitespace or end of string
        sentences = re.split(r'(' + sentence_endings + r')\s+', text)
        
        # Reconstruct sentences with their punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = (sentences[i] + sentences[i + 1]).strip()
            else:
                sentence = sentences[i].strip()
            
            if sentence:
                result.append(sentence)
        
        # Handle the last part if it doesn't end with punctuation
        if len(sentences) % 2 != 0 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result

    def _group_sentences_semantically(self, sentences: List[str]) -> List[str]:
        """Group sentences into chunks based on semantic similarity."""
        if len(sentences) <= 2:
            return [' '.join(sentences)]
        
        chunks = []
        current_chunk = [sentences[0]]
        current_length = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            
            # Check if adding this sentence would exceed chunk size
            potential_length = current_length + len(sentence) + 1  # +1 for space
            
            if potential_length <= self.chunk_size:
                # Simple heuristic: check for topic changes based on keywords
                if self._should_group_sentences(current_chunk[-1], sentence):
                    current_chunk.append(sentence)
                    current_length = potential_length
                else:
                    # Start new chunk
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = len(sentence)
            else:
                # Current chunk is full, start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _should_group_sentences(self, prev_sentence: str, current_sentence: str) -> bool:
        """
        Simple heuristic to determine if sentences should be grouped together.
        This is a basic implementation - can be enhanced with actual embeddings.
        """
        # Convert to lowercase for comparison
        prev_lower = prev_sentence.lower()
        current_lower = current_sentence.lower()
        
        # Check for connecting words that indicate continuation
        connecting_words = [
            'tuy nhiên', 'nhưng', 'và', 'vì vậy', 'do đó', 'bởi vì', 'ngoài ra',
            'moreover', 'however', 'therefore', 'furthermore', 'additionally',
            'in addition', 'consequently', 'as a result', 'because', 'since'
        ]
        
        current_starts_with_connector = any(
            current_lower.startswith(word) for word in connecting_words
        )
        
        if current_starts_with_connector:
            return True
        
        # Check for shared keywords (simple word overlap)
        prev_words = set(re.findall(r'\b\w+\b', prev_lower))
        current_words = set(re.findall(r'\b\w+\b', current_lower))
        
        # Remove common stop words
        stop_words = {
            'là', 'của', 'và', 'có', 'trong', 'với', 'cho', 'về', 'từ', 'được', 'một',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
        }
        
        prev_words -= stop_words
        current_words -= stop_words
        
        # Calculate word overlap ratio
        if prev_words and current_words:
            overlap = len(prev_words.intersection(current_words))
            total_unique = len(prev_words.union(current_words))
            overlap_ratio = overlap / total_unique if total_unique > 0 else 0
            
            # Group if there's significant word overlap
            return overlap_ratio > 0.2
        
        return True  # Default to grouping

    def _cluster_sentences(self, sentences: List[str]) -> List[List[int]]:
        """
        Cluster sentences based on semantic similarity using simple features.
        Returns list of clusters, where each cluster is a list of sentence indices.
        """
        # Calculate similarity matrix between sentences
        similarity_matrix = self._calculate_sentence_similarity_matrix(sentences)
        
        # Perform hierarchical clustering based on similarity
        clusters = self._hierarchical_clustering(similarity_matrix, sentences)
        
        return clusters

    def _calculate_sentence_similarity_matrix(self, sentences: List[str]) -> List[List[float]]:
        """
        Calculate similarity matrix between sentences using keyword overlap and other features.
        """
        n = len(sentences)
        similarity_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # Pre-process sentences for comparison
        processed_sentences = []
        for sentence in sentences:
            processed = self._extract_sentence_features(sentence)
            processed_sentences.append(processed)
        
        # Calculate pairwise similarities
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity = self._calculate_sentence_pair_similarity(
                        processed_sentences[i], 
                        processed_sentences[j]
                    )
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
        
        return similarity_matrix

    def _extract_sentence_features(self, sentence: str) -> Dict[str, Any]:
        """
        Extract features from a sentence for similarity comparison.
        """
        sentence_lower = sentence.lower()
        
        # Extract words (removing stop words)
        stop_words = {
            'là', 'của', 'và', 'có', 'trong', 'với', 'cho', 'về', 'từ', 'được', 'một', 
            'này', 'đó', 'các', 'những', 'để', 'khi', 'nếu', 'như', 'theo', 'nhưng',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
        
        words = set(re.findall(r'\b\w+\b', sentence_lower)) - stop_words
        
        # Extract key phrases (simple noun phrases)
        noun_phrases = self._extract_simple_noun_phrases(sentence_lower)
        
        # Extract named entities (simple pattern matching)
        entities = self._extract_simple_entities(sentence)
        
        # Calculate sentence length and structure features
        length = len(sentence)
        word_count = len(sentence.split())
        
        return {
            'words': words,
            'noun_phrases': noun_phrases,
            'entities': entities,
            'length': length,
            'word_count': word_count,
            'original': sentence
        }

    def _extract_simple_noun_phrases(self, sentence: str) -> set:
        """
        Extract simple noun phrases using basic patterns.
        """
        noun_phrases = set()
        
        # Pattern for Vietnamese noun phrases (adjective + noun)
        patterns = [
            r'\b[a-záàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ]+\s+[a-záàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ]+\b',
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # Proper nouns
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sentence)
            noun_phrases.update(matches)
        
        return noun_phrases

    def _extract_simple_entities(self, sentence: str) -> set:
        """
        Extract simple entities using pattern matching.
        """
        entities = set()
        
        # Numbers
        numbers = re.findall(r'\b\d+(?:[.,]\d+)*\b', sentence)
        entities.update(numbers)
        
        # Dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', sentence)
        entities.update(dates)
        
        # Proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', sentence)
        entities.update(proper_nouns)
        
        return entities

    def _calculate_sentence_pair_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Calculate similarity between two sentences based on their features.
        """
        # Word overlap similarity
        words1, words2 = features1['words'], features2['words']
        if words1 or words2:
            word_intersection = len(words1.intersection(words2))
            word_union = len(words1.union(words2))
            word_similarity = word_intersection / word_union if word_union > 0 else 0
        else:
            word_similarity = 0
        
        # Noun phrase similarity
        phrases1, phrases2 = features1['noun_phrases'], features2['noun_phrases']
        if phrases1 or phrases2:
            phrase_intersection = len(phrases1.intersection(phrases2))
            phrase_union = len(phrases1.union(phrases2))
            phrase_similarity = phrase_intersection / phrase_union if phrase_union > 0 else 0
        else:
            phrase_similarity = 0
        
        # Entity similarity
        entities1, entities2 = features1['entities'], features2['entities']
        if entities1 or entities2:
            entity_intersection = len(entities1.intersection(entities2))
            entity_union = len(entities1.union(entities2))
            entity_similarity = entity_intersection / entity_union if entity_union > 0 else 0
        else:
            entity_similarity = 0
        
        # Length similarity (normalized difference)
        len1, len2 = features1['length'], features2['length']
        length_diff = abs(len1 - len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        length_similarity = 1 - length_diff
        
        # Weighted combination of similarities
        similarity = (
            0.5 * word_similarity +      # Word overlap is most important
            0.25 * phrase_similarity +   # Noun phrases add context
            0.15 * entity_similarity +   # Entities provide specificity
            0.1 * length_similarity      # Length provides structure info
        )
        
        return similarity

    def _hierarchical_clustering(self, similarity_matrix: List[List[float]], sentences: List[str]) -> List[List[int]]:
        """
        Perform simple hierarchical clustering based on similarity matrix.
        """
        n = len(sentences)
        
        # Initialize each sentence as its own cluster
        clusters = [[i] for i in range(n)]
        
        # Similarity threshold for merging clusters
        merge_threshold = 0.3
        
        # Keep merging until no more merges are possible or we reach reasonable number of clusters
        max_clusters = max(2, n // 3)  # Don't reduce too much
        
        while len(clusters) > max_clusters:
            best_merge_similarity = -1
            best_merge_indices = None
            
            # Find the best pair of clusters to merge
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Calculate average similarity between clusters
                    cluster_similarity = self._calculate_cluster_similarity(
                        clusters[i], clusters[j], similarity_matrix
                    )
                    
                    if cluster_similarity > best_merge_similarity and cluster_similarity > merge_threshold:
                        best_merge_similarity = cluster_similarity
                        best_merge_indices = (i, j)
            
            # If no good merge found, break
            if best_merge_indices is None:
                break
            
            # Merge the best pair
            i, j = best_merge_indices
            merged_cluster = clusters[i] + clusters[j]
            
            # Remove the original clusters and add merged one
            new_clusters = []
            for k, cluster in enumerate(clusters):
                if k != i and k != j:
                    new_clusters.append(cluster)
            new_clusters.append(merged_cluster)
            
            clusters = new_clusters
        
        return clusters

    def _calculate_cluster_similarity(self, cluster1: List[int], cluster2: List[int], 
                                    similarity_matrix: List[List[float]]) -> float:
        """
        Calculate similarity between two clusters using average linkage.
        """
        if not cluster1 or not cluster2:
            return 0.0
        
        total_similarity = 0.0
        pair_count = 0
        
        for i in cluster1:
            for j in cluster2:
                total_similarity += similarity_matrix[i][j]
                pair_count += 1
        
        return total_similarity / pair_count if pair_count > 0 else 0.0

    def _clusters_to_chunks(self, clusters: List[List[int]], sentences: List[str]) -> List[str]:
        """
        Convert sentence clusters to text chunks.
        """
        chunks = []
        
        for cluster in clusters:
            # Sort sentences in cluster by their original order
            cluster.sort()
            
            # Combine sentences in the cluster
            cluster_sentences = [sentences[i] for i in cluster]
            chunk_text = ' '.join(cluster_sentences)
            
            # Check if chunk is too small, merge with adjacent if possible
            if len(chunk_text) < self.chunk_size // 4 and len(chunks) > 0:
                # Try to merge with previous chunk if combined size is reasonable
                prev_chunk = chunks[-1]
                if len(prev_chunk) + len(chunk_text) + 1 <= self.chunk_size:
                    chunks[-1] = prev_chunk + ' ' + chunk_text
                    continue
            
            chunks.append(chunk_text)
        
        return chunks

    
    def _llm_analyze_and_split(self, model, text: str) -> List[str]:
        """
        Use LLM to analyze text and suggest optimal split points.
        """
        # Create prompt for semantic analysis
        prompt = self._create_llm_analysis_prompt(text)
        
        try:
            response = model.generate_content(prompt)
            split_points = self._parse_llm_response(response.text)
            
            # Apply the suggested splits
            chunks = self._apply_llm_splits(text, split_points)
            
            # Validate and adjust chunk sizes
            final_chunks = self._validate_llm_chunks(chunks)
            
            return final_chunks
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            # Fallback to sentence-based splitting
            sentences = self._split_into_sentences(text)
            return self._group_sentences_semantically(sentences)

    def _apply_overlap_to_chunks(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between chunks using word-based overlap.
        """
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i-1]
                overlap_text = self._get_overlap_text(prev_chunk, self.overlap)
                
                # Add overlap to current chunk if it doesn't already start with it
                if overlap_text and not chunk.startswith(overlap_text.strip()):
                    overlapped_chunk = overlap_text + " " + chunk
                    
                    # Check if combined chunk is too long
                    if len(overlapped_chunk) > self.chunk_size * 1.2:  # Allow 20% overflow
                        overlapped_chunks.append(chunk)  # Use original chunk
                    else:
                        overlapped_chunks.append(overlapped_chunk)
                else:
                    overlapped_chunks.append(chunk)
        
        return overlapped_chunks

    def _split_recursive(self, text: str) -> List[str]:
        """
        Split text recursively using multiple separators with priority order.
        Tries to split by paragraphs first, then sentences, then words, then characters.
        """
        if not text:
            return []
        
        # Define separators in order of preference (most semantic to least semantic)
        # Vietnamese-specific separators included
        separators = [
            "\n\n",  # Double newline (paragraph breaks)
            "\n",    # Single newline
            ". ",    # English sentence endings
            "。 ",   # Vietnamese sentence endings (if using Asian punctuation)
            "! ",    # Exclamation
            "? ",    # Question
            "; ",    # Semicolon
            ", ",    # Comma
            " ",     # Space (word boundary)
            ""       # Character level (last resort)
        ]
        
        return self._recursive_split_helper(text, separators, 0)

    def _pre_split_for_llm(self, text: str) -> List[str]:
        """
        Pre-split very long text into manageable sections for LLM analysis.
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        max_section_size = 8000  # Leave room for prompt overhead
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= max_section_size:
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If single paragraph is too long, split it
                if len(paragraph) > max_section_size:
                    # Use recursive splitting for very long paragraphs
                    sub_chunks = self._split_recursive(paragraph)
                    # Group sub-chunks to fit within size limit
                    temp_chunk = ""
                    for sub_chunk in sub_chunks:
                        if len(temp_chunk) + len(sub_chunk) + 1 <= max_section_size:
                            if temp_chunk:
                                temp_chunk += ' ' + sub_chunk
                            else:
                                temp_chunk = sub_chunk
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sub_chunk
                    
                    current_chunk = temp_chunk
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _create_llm_analysis_prompt(self, text: str) -> str:
        """
        Create a prompt for LLM to analyze text structure and suggest splits.
        """
        prompt = f"""
    Phân tích văn bản sau và xác định các điểm phân chia tối ưu dựa trên ngữ nghĩa và cấu trúc. 
    Mỗi đoạn sau khi chia nên có độ dài khoảng {self.chunk_size} ký tự và có nội dung liên quan chặt chẽ với nhau.

    Quy tắc:
    1. Ưu tiên chia tại ranh giới đoạn văn
    2. Tránh chia giữa câu
    3. Giữ các ý liên quan trong cùng một chunk
    4. Đảm bảo mỗi chunk có thể hiểu được độc lập

    Văn bản cần phân tích:
    ---
    {text}
    ---

    Hãy trả về danh sách các vị trí chia (character index) theo format:
    SPLIT_POINTS: [vị_trí_1, vị_trí_2, vị_trí_3, ...]

    Ví dụ: SPLIT_POINTS: [245, 567, 891]
    """
        return prompt

    def _parse_llm_response(self, response_text: str) -> List[int]:
        """
        Parse LLM response to extract split points.
        """
        split_points = []
        
        try:
            # Look for SPLIT_POINTS pattern
            import re
            match = re.search(r'SPLIT_POINTS:\s*\[([\d,\s]+)\]', response_text)
            
            if match:
                points_str = match.group(1)
                # Extract numbers
                points = re.findall(r'\d+', points_str)
                split_points = [int(p) for p in points]
            else:
                # Fallback: try to find any numbers that could be positions
                numbers = re.findall(r'\b\d{2,}\b', response_text)
                if numbers:
                    split_points = [int(n) for n in numbers[:10]]  # Limit to first 10
        
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
        
        return sorted(split_points)

    def _apply_llm_splits(self, text: str, split_points: List[int]) -> List[str]:
        """
        Apply the split points suggested by LLM to create chunks.
        """
        if not split_points:
            return [text]
        
        chunks = []
        start = 0
        
        for split_point in split_points:
            # Validate split point
            if split_point <= start or split_point >= len(text):
                continue
            
            # Adjust split point to nearby sentence boundary
            adjusted_point = self._adjust_split_point(text, split_point, start)
            
            chunk = text[start:adjusted_point].strip()
            if chunk:
                chunks.append(chunk)
            
            start = adjusted_point
        
        # Add remaining text
        if start < len(text):
            remaining = text[start:].strip()
            if remaining:
                chunks.append(remaining)
        
        return chunks

    def _adjust_split_point(self, text: str, suggested_point: int, min_point: int) -> int:
        """
        Adjust split point to nearby sentence or paragraph boundary.
        """
        # Search window around suggested point
        window = 100
        start_search = max(min_point, suggested_point - window)
        end_search = min(len(text), suggested_point + window)
        
        # Look for paragraph breaks first
        for i in range(suggested_point, end_search):
            if i + 1 < len(text) and text[i:i+2] == '\n\n':
                return i + 2
        
        for i in range(suggested_point, start_search, -1):
            if i + 1 < len(text) and text[i:i+2] == '\n\n':
                return i + 2
        
        # Look for sentence endings
        sentence_endings = ['. ', '! ', '? ', '。 ', '！ ', '？ ']
        
        for i in range(suggested_point, end_search):
            for ending in sentence_endings:
                if text[i:i+len(ending)] == ending:
                    return i + len(ending)
        
        for i in range(suggested_point, start_search, -1):
            for ending in sentence_endings:
                if text[i:i+len(ending)] == ending:
                    return i + len(ending)
        
        # If no good boundary found, return original point
        return suggested_point

    def _validate_llm_chunks(self, chunks: List[str]) -> List[str]:
        """
        Validate and adjust chunks to meet size requirements.
        """
        final_chunks = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # Chunk too large, split it further
                sub_chunks = self._split_recursive(chunk)
                final_chunks.extend(sub_chunks)
        
        # Merge very small chunks with adjacent ones
        merged_chunks = []
        min_chunk_size = self.chunk_size // 4
        
        for chunk in final_chunks:
            if len(chunk) < min_chunk_size and merged_chunks:
                # Try to merge with previous chunk
                prev_chunk = merged_chunks[-1]
                if len(prev_chunk) + len(chunk) + 1 <= self.chunk_size:
                    merged_chunks[-1] = prev_chunk + ' ' + chunk
                    continue
            
            merged_chunks.append(chunk)
        
        # Apply overlap if specified
        if self.overlap > 0 and len(merged_chunks) > 1:
            return self._apply_overlap_to_chunks(merged_chunks)
        
        return merged_chunks


class TextSplitter(TextProcessor):
    """Handles text splitting with multiple methods."""
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.split_helper = SplitHelper(chunk_size, overlap)
    
    def process(self, text: str, method: str = "recursive") -> List[str]:
        if method == "character":
            return self._split_by_character(text)
        elif method == "token":
            return self._split_by_token(text)
        elif method == "recursive":
            return self._split_recursive(text)
        elif method == "semantic":
            return self._split_semantic(text)
        elif method == "cluster_semantic":
            return self._split_cluster_semantic(text)
        elif method == "llm":
            return self._split_llm_semantic(text)
        else:
            raise ValueError(f"Unknown splitting method: {method}")

    def _split_by_character(self, text: str) -> List[str]:
        """Split text by character count with overlap."""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this would be the last chunk and it's small, extend to include all remaining text
            if end >= len(text):
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # Find a good breaking point near the chunk boundary
            chunk_end = end
            
            # Try to break at sentence boundaries first
            sentence_breaks = ['.', '!', '?', '。', '！', '？']  # Include Vietnamese punctuation
            for i in range(max(start, end - 50), min(end + 50, len(text))):
                if i > start and i < len(text) and text[i] in sentence_breaks and i + 1 < len(text) and text[i + 1] == ' ':
                    chunk_end = i + 1
                    break
            
            # If no sentence break found, try paragraph breaks
            if chunk_end == end:
                for i in range(max(start, end - 100), min(end + 100, len(text))):
                    if i > start and i < len(text) - 1 and text[i:i+2] == '\n\n':
                        chunk_end = i + 2
                        break
            
            # If no paragraph break, try word boundaries
            if chunk_end == end:
                for i in range(end, min(end + 50, len(text))):
                    if i < len(text) and text[i] == ' ':
                        chunk_end = i
                        break
                
                # If still no good break point, try backwards
                if chunk_end == end:
                    for i in range(end - 1, max(start + self.chunk_size // 2, start), -1):
                        if i < len(text) and text[i] == ' ':
                            chunk_end = i
                            break
            
            # Extract chunk and clean it
            chunk = text[start:chunk_end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Calculate next start position with overlap
            next_start = chunk_end - self.overlap
            
            # Ensure we don't go backwards
            if next_start <= start:
                next_start = chunk_end
            
            start = next_start
        
        return chunks

    def _split_by_token(self, text: str) -> List[str]:
        """Split text by token count using word-based tokenization for Vietnamese text."""
        if not text:
            return []
        
        # Simple word-based tokenization for Vietnamese
        # Vietnamese uses spaces to separate words, similar to English
        words = text.split()
        
        if not words:
            return []
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for word in words:
            # Add word to current chunk
            current_chunk.append(word)
            current_token_count += 1
            
            # If we've reached the chunk size, finalize the chunk
            if current_token_count >= self.chunk_size:
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if self.overlap > 0 and len(current_chunk) > self.overlap:
                    # Keep the last 'overlap' words for the next chunk
                    overlap_words = current_chunk[-self.overlap:]
                    current_chunk = overlap_words
                    current_token_count = len(overlap_words)
                else:
                    current_chunk = []
                    current_token_count = 0
        
        # Add remaining words as the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks

    def _split_recursive(self, text: str) -> List[str]:
        """
        Split text recursively using multiple separators with priority order.
        Tries to split by paragraphs first, then sentences, then words, then characters.
        """
        if not text:
            return []
        
        # Define separators in order of preference (most semantic to least semantic)
        # Vietnamese-specific separators included
        separators = [
            "\n\n",  # Double newline (paragraph breaks)
            "\n",    # Single newline
            ". ",    # English sentence endings
            "。 ",   # Vietnamese sentence endings (if using Asian punctuation)
            "! ",    # Exclamation
            "? ",    # Question
            "; ",    # Semicolon
            ", ",    # Comma
            " ",     # Space (word boundary)
            ""       # Character level (last resort)
        ]
        
        return self.split_helper._recursive_split_helper(text, separators, 0)

    def _split_semantic(self, text: str) -> List[str]:
        """
        Split text based on semantic similarity using sentence embeddings.
        Groups semantically similar sentences together into chunks.
        """
        if not text or len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        # Split text into sentences first
        sentences = self.split_helper._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text.strip()]
        
        # If we don't have enough sentences for semantic analysis, fall back to recursive
        if len(sentences) < 3:
            return self._split_recursive(text)
        
        try:
            # Simple semantic grouping based on sentence similarity
            chunks = self.split_helper._group_sentences_semantically(sentences)
            
            # Ensure chunks don't exceed size limits
            final_chunks = []
            for chunk in chunks:
                if len(chunk) <= self.chunk_size:
                    final_chunks.append(chunk.strip())
                else:
                    # If chunk is too large, split it recursively
                    sub_chunks = self._split_recursive(chunk)
                    final_chunks.extend(sub_chunks)
            
            return [chunk for chunk in final_chunks if chunk.strip()]
            
        except Exception:
            # Fallback to recursive splitting if semantic analysis fails
            return self._split_recursive(text)

    def _split_cluster_semantic(self, text: str) -> List[str]:
        """
        Phân tích cấu trúc: Chia văn bản thành câu và trích xuất các đặc trưng (từ khóa, cụm danh từ, thực thể)
        """
        if not text or len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        # Split text into sentences first
        sentences = self.split_helper._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text.strip()]
        
        # If too few sentences for clustering, fall back to semantic grouping
        if len(sentences) < 5:
            return self._split_semantic(text)
        
        try:
            # Perform clustering-based semantic chunking
            clusters = self.split_helper._cluster_sentences(sentences)
            
            # Convert clusters to text chunks
            chunks = self.split_helper._clusters_to_chunks(clusters, sentences)
            
            # Ensure chunks don't exceed size limits
            final_chunks = []
            for chunk in chunks:
                if len(chunk) <= self.chunk_size:
                    final_chunks.append(chunk.strip())
                else:
                    # If chunk is too large, split it recursively
                    sub_chunks = self._split_recursive(chunk)
                    final_chunks.extend(sub_chunks)
            
            return [chunk for chunk in final_chunks if chunk.strip()]
            
        except Exception:
            # Fallback to regular semantic splitting if clustering fails
            return self._split_semantic(text)

    def _split_llm_semantic(self, text: str) -> List[str]:
        """
        Split text using LLM-based semantic analysis.
        Uses Gemini API to analyze text structure and find optimal split points.
        """
        if not text or len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        try:
            # Configure Gemini API
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # If text is very long, pre-split into manageable sections
            if len(text) > 10000:  # 10k characters threshold
                pre_chunks = self.split_helper._pre_split_for_llm(text)
                final_chunks = []
                
                for pre_chunk in pre_chunks:
                    llm_chunks = self.split_helper._llm_analyze_and_split(model, pre_chunk)
                    final_chunks.extend(llm_chunks)
                
                return final_chunks
            else:
                return self.split_helper._llm_analyze_and_split(model, text)
                
        except Exception as e:
            print(f"LLM semantic splitting failed: {e}")
            # Fallback to cluster semantic splitting
            return self._split_cluster_semantic(text)

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between chunks using word-based overlap.
        """
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i-1]
                overlap_text = self.split_helper._get_overlap_text(prev_chunk, self.overlap)
                
                # Add overlap to current chunk if it doesn't already start with it
                if overlap_text and not chunk.startswith(overlap_text.strip()):
                    overlapped_chunk = overlap_text + " " + chunk
                    
                    # Check if combined chunk is too long
                    if len(overlapped_chunk) > self.chunk_size * 1.2:  # Allow 20% overflow
                        overlapped_chunks.append(chunk)  # Use original chunk
                    else:
                        overlapped_chunks.append(overlapped_chunk)
                else:
                    overlapped_chunks.append(chunk)
        
        return overlapped_chunks
