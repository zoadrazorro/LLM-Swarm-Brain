"""
Conceptual Thread Tracking System

Tracks concepts as they flow through the neural network,
enabling analysis of how ideas propagate and transform.
"""

from typing import List, Dict, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np
import logging

from llm_swarm_brain.utils import EmbeddingManager


logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """Represents a single concept"""
    concept_id: str
    name: str
    embedding: np.ndarray
    first_seen: datetime = field(default_factory=datetime.now)
    occurrence_count: int = 0


@dataclass
class ConceptualThread:
    """Tracks a concept's journey through the network"""
    concept_id: str
    concept_name: str
    neuron_path: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    activation_levels: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    related_concepts: Set[str] = field(default_factory=set)


class ConceptExtractor:
    """
    Extracts concepts from text

    Uses embeddings and simple NLP to identify key concepts.
    """

    def __init__(self):
        self.embedding_manager = EmbeddingManager(device="cpu")

        # Common stop words to filter
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'
        }

    def extract_concepts(
        self,
        text: str,
        max_concepts: int = 10,
        min_word_length: int = 4
    ) -> List[str]:
        """
        Extract concepts from text

        Args:
            text: Input text
            max_concepts: Maximum concepts to extract
            min_word_length: Minimum word length to consider

        Returns:
            List of concept strings
        """
        # Tokenize and clean
        words = text.lower().split()
        words = [
            word.strip('.,;:!?"()[]{}')
            for word in words
        ]

        # Filter candidates
        candidates = []
        for word in words:
            if (
                len(word) >= min_word_length and
                word not in self.stop_words and
                word.isalpha()  # Only alphabetic
            ):
                candidates.append(word)

        # Look for multi-word concepts (simple bigrams)
        bigrams = []
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            if (
                w1 not in self.stop_words and
                w2 not in self.stop_words and
                len(w1) >= min_word_length and
                len(w2) >= min_word_length
            ):
                bigrams.append(f"{w1} {w2}")

        # Combine single words and bigrams
        all_candidates = candidates + bigrams

        # Score candidates by frequency
        from collections import Counter
        concept_counts = Counter(all_candidates)

        # Get top concepts
        top_concepts = [
            concept for concept, count in concept_counts.most_common(max_concepts)
        ]

        return top_concepts


class ConceptualThreadTracker:
    """
    Tracks conceptual threads through the neural network

    Monitors how concepts:
    - Flow between neurons
    - Transform and evolve
    - Combine with other concepts
    - Contribute to final outputs
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_threads_per_concept: int = 5
    ):
        """
        Initialize conceptual thread tracker

        Args:
            similarity_threshold: Threshold for concept matching
            max_threads_per_concept: Max threads to track per concept
        """
        self.similarity_threshold = similarity_threshold
        self.max_threads_per_concept = max_threads_per_concept

        self.concept_extractor = ConceptExtractor()
        self.embedding_manager = EmbeddingManager(device="cpu")

        # Concept database
        self.concepts: Dict[str, Concept] = {}

        # Active threads
        self.active_threads: Dict[str, List[ConceptualThread]] = defaultdict(list)

        # Completed threads (archived)
        self.completed_threads: List[ConceptualThread] = []

        # Concept co-occurrence graph
        self.concept_cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        self.total_concepts_tracked = 0
        self.total_threads_created = 0

        logger.info("Initialized ConceptualThreadTracker")

    def process_neuron_output(
        self,
        neuron_id: str,
        output_text: str,
        activation_level: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Process neuron output and track concepts

        Args:
            neuron_id: Neuron that produced output
            output_text: Output text
            activation_level: Activation level
            timestamp: Timestamp of output
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Extract concepts from output
        concept_names = self.concept_extractor.extract_concepts(output_text)

        # Track each concept
        for concept_name in concept_names:
            self._track_concept(
                concept_name=concept_name,
                neuron_id=neuron_id,
                output_text=output_text,
                activation_level=activation_level,
                timestamp=timestamp
            )

        # Track co-occurrences
        self._track_cooccurrences(concept_names)

    def _track_concept(
        self,
        concept_name: str,
        neuron_id: str,
        output_text: str,
        activation_level: float,
        timestamp: datetime
    ):
        """
        Track a single concept occurrence

        Args:
            concept_name: Name of concept
            neuron_id: Neuron ID
            output_text: Full output text
            activation_level: Activation level
            timestamp: Timestamp
        """
        # Get or create concept
        concept_id = self._get_or_create_concept(concept_name)

        # Update concept occurrence count
        self.concepts[concept_id].occurrence_count += 1

        # Find or create thread for this concept
        thread = self._find_or_create_thread(concept_id, concept_name)

        # Add to thread
        thread.neuron_path.append(neuron_id)
        thread.activation_levels.append(activation_level)
        thread.timestamps.append(timestamp)

        # Extract transformation (simplified: just store relevant snippet)
        snippet = self._extract_relevant_snippet(output_text, concept_name)
        thread.transformations.append(snippet)

    def _get_or_create_concept(self, concept_name: str) -> str:
        """
        Get existing concept or create new one

        Args:
            concept_name: Name of concept

        Returns:
            Concept ID
        """
        # Check if concept already exists (by similarity)
        concept_embedding = self.embedding_manager.generate_embedding(concept_name)

        for concept_id, concept in self.concepts.items():
            similarity = self.embedding_manager.cosine_similarity(
                concept_embedding,
                concept.embedding
            )

            if similarity >= self.similarity_threshold:
                # Existing concept
                return concept_id

        # Create new concept
        concept_id = f"concept_{self.total_concepts_tracked}"
        self.concepts[concept_id] = Concept(
            concept_id=concept_id,
            name=concept_name,
            embedding=concept_embedding
        )

        self.total_concepts_tracked += 1
        logger.debug(f"New concept tracked: {concept_name} (id={concept_id})")

        return concept_id

    def _find_or_create_thread(
        self,
        concept_id: str,
        concept_name: str
    ) -> ConceptualThread:
        """
        Find active thread or create new one

        Args:
            concept_id: Concept ID
            concept_name: Concept name

        Returns:
            Conceptual thread
        """
        # Check active threads for this concept
        active = self.active_threads[concept_id]

        # Use most recent thread if it exists and isn't too old
        if active:
            most_recent = active[-1]
            time_since_last = datetime.now() - most_recent.timestamps[-1]

            # If less than 10 seconds, continue this thread
            if time_since_last.total_seconds() < 10:
                return most_recent

        # Create new thread
        thread = ConceptualThread(
            concept_id=concept_id,
            concept_name=concept_name
        )

        self.active_threads[concept_id].append(thread)
        self.total_threads_created += 1

        # Limit threads per concept
        if len(self.active_threads[concept_id]) > self.max_threads_per_concept:
            # Archive oldest thread
            oldest = self.active_threads[concept_id].pop(0)
            self.completed_threads.append(oldest)

        return thread

    def _extract_relevant_snippet(
        self,
        text: str,
        concept: str,
        window: int = 50
    ) -> str:
        """
        Extract snippet containing concept

        Args:
            text: Full text
            concept: Concept to find
            window: Characters around concept

        Returns:
            Relevant snippet
        """
        # Find concept in text (case-insensitive)
        lower_text = text.lower()
        lower_concept = concept.lower()

        pos = lower_text.find(lower_concept)

        if pos == -1:
            # Concept not found directly, return first part
            return text[:100]

        # Extract window around concept
        start = max(0, pos - window)
        end = min(len(text), pos + len(concept) + window)

        snippet = text[start:end]

        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet

    def _track_cooccurrences(self, concepts: List[str]):
        """
        Track concept co-occurrences

        Args:
            concepts: List of concepts occurring together
        """
        # Record all pairs
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:
                # Normalize ordering
                pair = tuple(sorted([concept_a, concept_b]))
                self.concept_cooccurrence[pair] += 1

    def get_concept_flow(
        self,
        concept_name: str,
        max_threads: int = 5
    ) -> List[ConceptualThread]:
        """
        Get flow of a specific concept through network

        Args:
            concept_name: Concept to track
            max_threads: Maximum threads to return

        Returns:
            List of threads for this concept
        """
        # Find concept ID
        concept_id = None
        for cid, concept in self.concepts.items():
            if concept.name.lower() == concept_name.lower():
                concept_id = cid
                break

        if not concept_id:
            return []

        # Get threads (active + completed)
        all_threads = (
            self.active_threads.get(concept_id, []) +
            [t for t in self.completed_threads if t.concept_id == concept_id]
        )

        # Sort by recency
        all_threads.sort(key=lambda t: t.timestamps[-1] if t.timestamps else datetime.min, reverse=True)

        return all_threads[:max_threads]

    def get_related_concepts(
        self,
        concept_name: str,
        top_k: int = 5
    ) -> List[Tuple[str, int]]:
        """
        Get concepts related to given concept

        Args:
            concept_name: Source concept
            top_k: Number of related concepts to return

        Returns:
            List of (concept_name, cooccurrence_count) tuples
        """
        related = []

        for (concept_a, concept_b), count in self.concept_cooccurrence.items():
            if concept_a.lower() == concept_name.lower():
                related.append((concept_b, count))
            elif concept_b.lower() == concept_name.lower():
                related.append((concept_a, count))

        # Sort by count
        related.sort(key=lambda x: x[1], reverse=True)

        return related[:top_k]

    def get_network_flow_summary(self) -> Dict[str, Any]:
        """
        Get summary of conceptual flow through network

        Returns:
            Summary dictionary
        """
        # Top concepts by occurrence
        top_concepts = sorted(
            self.concepts.items(),
            key=lambda x: x[1].occurrence_count,
            reverse=True
        )[:10]

        # Most common concept pairs
        top_pairs = sorted(
            self.concept_cooccurrence.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "total_concepts": len(self.concepts),
            "total_threads": self.total_threads_created,
            "active_threads": sum(len(threads) for threads in self.active_threads.values()),
            "completed_threads": len(self.completed_threads),
            "top_concepts": [
                {
                    "name": concept.name,
                    "occurrences": concept.occurrence_count
                }
                for _, concept in top_concepts
            ],
            "top_concept_pairs": [
                {
                    "concepts": list(pair),
                    "cooccurrences": count
                }
                for pair, count in top_pairs
            ]
        }

    def visualize_concept_flow(
        self,
        concept_name: str
    ) -> str:
        """
        Create text visualization of concept flow

        Args:
            concept_name: Concept to visualize

        Returns:
            Text visualization
        """
        threads = self.get_concept_flow(concept_name, max_threads=3)

        if not threads:
            return f"No threads found for concept: {concept_name}"

        lines = [f"=== Concept Flow: {concept_name} ===\n"]

        for i, thread in enumerate(threads, 1):
            lines.append(f"\nThread {i}:")
            lines.append(f"Path: {' → '.join(thread.neuron_path[:5])}" +
                        (" ..." if len(thread.neuron_path) > 5 else ""))

            if thread.transformations:
                lines.append(f"Latest transformation: {thread.transformations[-1][:100]}...")

            if thread.related_concepts:
                lines.append(f"Related: {', '.join(list(thread.related_concepts)[:3])}")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        return {
            "total_concepts_tracked": self.total_concepts_tracked,
            "total_threads_created": self.total_threads_created,
            "active_concepts": len(self.concepts),
            "active_threads": sum(len(threads) for threads in self.active_threads.values()),
            "completed_threads": len(self.completed_threads),
            "concept_pairs_tracked": len(self.concept_cooccurrence)
        }

    def __repr__(self) -> str:
        return (
            f"ConceptualThreadTracker(concepts={len(self.concepts)}, "
            f"threads={self.total_threads_created})"
        )
