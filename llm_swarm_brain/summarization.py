"""
Summarization and Compression System

Implements intelligent output compression to prevent information explosion
during signal propagation through the neural network.
"""

from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, field
from datetime import datetime

from llm_swarm_brain.utils import EmbeddingManager


logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of text compression/summarization"""
    original_text: str
    compressed_text: str
    compression_ratio: float
    key_concepts: List[str] = field(default_factory=list)
    information_loss: float = 0.0  # Estimated 0-1
    timestamp: datetime = field(default_factory=datetime.now)


class SummarizationNeuron:
    """
    Specialized neuron for compressing verbose outputs

    Prevents information explosion by intelligently summarizing
    outputs before propagation to other neurons.
    """

    def __init__(
        self,
        max_output_length: int = 200,
        compression_threshold: int = 300,
        preserve_key_concepts: bool = True
    ):
        """
        Initialize summarization neuron

        Args:
            max_output_length: Maximum length of compressed output
            compression_threshold: Only compress if longer than this
            preserve_key_concepts: Whether to extract and preserve key concepts
        """
        self.max_output_length = max_output_length
        self.compression_threshold = compression_threshold
        self.preserve_key_concepts = preserve_key_concepts

        self.embedding_manager = EmbeddingManager(device="cpu")
        self.total_compressions = 0
        self.total_characters_saved = 0

        logger.info(f"Initialized SummarizationNeuron (max_len={max_output_length})")

    def should_compress(self, text: str) -> bool:
        """
        Determine if text needs compression

        Args:
            text: Input text

        Returns:
            True if compression is needed
        """
        return len(text) > self.compression_threshold

    def compress(
        self,
        text: str,
        context: Optional[str] = None,
        preserve_concepts: Optional[List[str]] = None
    ) -> CompressionResult:
        """
        Compress/summarize text

        Args:
            text: Text to compress
            context: Optional context for compression
            preserve_concepts: Specific concepts to preserve

        Returns:
            Compression result
        """
        if not self.should_compress(text):
            # No compression needed
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                compression_ratio=1.0,
                key_concepts=[],
                information_loss=0.0
            )

        # Extract key concepts
        key_concepts = self._extract_key_concepts(text) if self.preserve_key_concepts else []

        # Add user-specified concepts to preserve
        if preserve_concepts:
            key_concepts.extend(preserve_concepts)
            key_concepts = list(set(key_concepts))  # Remove duplicates

        # Perform compression
        compressed_text = self._compress_text(
            text,
            max_length=self.max_output_length,
            key_concepts=key_concepts,
            context=context
        )

        # Calculate metrics
        compression_ratio = len(compressed_text) / len(text)
        characters_saved = len(text) - len(compressed_text)

        # Estimate information loss (simplified)
        information_loss = self._estimate_information_loss(text, compressed_text)

        self.total_compressions += 1
        self.total_characters_saved += characters_saved

        logger.debug(
            f"Compressed {len(text)} → {len(compressed_text)} chars "
            f"(ratio={compression_ratio:.2f}, loss={information_loss:.2f})"
        )

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            compression_ratio=compression_ratio,
            key_concepts=key_concepts,
            information_loss=information_loss
        )

    def _extract_key_concepts(self, text: str, top_k: int = 5) -> List[str]:
        """
        Extract key concepts from text

        Simplified implementation using basic heuristics.
        Full version would use NER or topic modeling.

        Args:
            text: Input text
            top_k: Number of key concepts to extract

        Returns:
            List of key concepts
        """
        # Simple extraction: find capitalized words and important terms
        words = text.split()

        # Heuristic: capitalized words (potential named entities)
        capitalized = [
            word.strip('.,;:!?"()[]')
            for word in words
            if word and word[0].isupper() and len(word) > 2
        ]

        # Heuristic: longer words (often more important)
        long_words = [
            word.strip('.,;:!?"()[]').lower()
            for word in words
            if len(word) > 8
        ]

        # Combine and deduplicate
        concepts = list(set(capitalized + long_words))[:top_k]

        return concepts

    def _compress_text(
        self,
        text: str,
        max_length: int,
        key_concepts: List[str],
        context: Optional[str]
    ) -> str:
        """
        Perform actual text compression

        Simplified implementation. Full version would use:
        - Extractive summarization
        - Abstractive summarization (with LLM)
        - Concept preservation

        Args:
            text: Text to compress
            max_length: Maximum output length
            key_concepts: Concepts to preserve
            context: Optional context

        Returns:
            Compressed text
        """
        # Strategy 1: Sentence extraction (extractive summarization)
        sentences = self._split_into_sentences(text)

        if not sentences:
            return text[:max_length]

        # Score sentences by importance
        sentence_scores = []
        for sentence in sentences:
            score = self._score_sentence(sentence, key_concepts)
            sentence_scores.append((sentence, score))

        # Sort by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Select sentences until we reach max_length
        selected_sentences = []
        current_length = 0

        for sentence, score in sentence_scores:
            sentence_len = len(sentence)
            if current_length + sentence_len <= max_length:
                selected_sentences.append(sentence)
                current_length += sentence_len
            else:
                # Try to fit a truncated version
                remaining = max_length - current_length
                if remaining > 50:  # Only if enough space
                    selected_sentences.append(sentence[:remaining] + "...")
                break

        # Reconstruct in original order
        # (simplified: just join selected sentences)
        compressed = " ".join(selected_sentences)

        # Ensure we're within limit
        if len(compressed) > max_length:
            compressed = compressed[:max_length - 3] + "..."

        return compressed

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting (could be improved with nltk)
        # Split on . ! ? followed by space and capital letter
        import re
        sentences = re.split(r'[.!?]+\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentence(self, sentence: str, key_concepts: List[str]) -> float:
        """
        Score sentence importance

        Args:
            sentence: Sentence to score
            key_concepts: Key concepts to look for

        Returns:
            Importance score
        """
        score = 0.0

        # Factor 1: Contains key concepts
        for concept in key_concepts:
            if concept.lower() in sentence.lower():
                score += 2.0

        # Factor 2: Sentence length (prefer medium-length sentences)
        length = len(sentence.split())
        if 10 <= length <= 25:
            score += 1.0
        elif length < 5:
            score -= 0.5  # Too short, probably not informative

        # Factor 3: Position (first and last sentences often important)
        # (This would require knowing sentence position, skipping for now)

        # Factor 4: Contains important words
        important_words = [
            'important', 'critical', 'key', 'main', 'primary',
            'because', 'therefore', 'thus', 'conclude', 'result'
        ]
        for word in important_words:
            if word in sentence.lower():
                score += 0.5

        return score

    def _estimate_information_loss(
        self,
        original: str,
        compressed: str
    ) -> float:
        """
        Estimate information loss from compression

        Uses semantic similarity between original and compressed.

        Args:
            original: Original text
            compressed: Compressed text

        Returns:
            Estimated information loss (0-1)
        """
        # Use embedding similarity
        try:
            orig_embedding = self.embedding_manager.generate_embedding(original)
            comp_embedding = self.embedding_manager.generate_embedding(compressed)

            similarity = self.embedding_manager.cosine_similarity(
                orig_embedding,
                comp_embedding
            )

            # Information loss is inverse of similarity
            loss = 1.0 - similarity

            return max(0.0, min(1.0, loss))

        except Exception as e:
            logger.warning(f"Could not estimate information loss: {e}")
            # Fallback: use compression ratio
            return 1.0 - (len(compressed) / len(original))

    def batch_compress(
        self,
        texts: Dict[str, str],
        preserve_concepts: Optional[List[str]] = None
    ) -> Dict[str, CompressionResult]:
        """
        Compress multiple texts in batch

        Args:
            texts: Dictionary mapping identifiers to texts
            preserve_concepts: Concepts to preserve across all compressions

        Returns:
            Dictionary mapping identifiers to compression results
        """
        results = {}

        for identifier, text in texts.items():
            results[identifier] = self.compress(
                text,
                preserve_concepts=preserve_concepts
            )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return {
            "total_compressions": self.total_compressions,
            "total_characters_saved": self.total_characters_saved,
            "avg_characters_saved": (
                self.total_characters_saved / self.total_compressions
                if self.total_compressions > 0 else 0
            ),
            "max_output_length": self.max_output_length,
            "compression_threshold": self.compression_threshold
        }

    def __repr__(self) -> str:
        return (
            f"SummarizationNeuron(compressions={self.total_compressions}, "
            f"saved={self.total_characters_saved} chars)"
        )
