"""
Attention Windowing System

Implements selective broadcasting based on role relevance.
Instead of broadcasting to all neurons, only broadcast to
"listening" neurons that are relevant to the content.
"""

from typing import List, Dict, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging

from llm_swarm_brain.config import NeuronRole
from llm_swarm_brain.utils import EmbeddingManager


logger = logging.getLogger(__name__)


@dataclass
class AttentionWindow:
    """Defines which neurons are "listening" for specific content"""
    content_embedding: np.ndarray
    listening_neurons: Set[str] = field(default_factory=set)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    window_size: int = 5  # Max neurons in window


class AttentionWindowManager:
    """
    Manages attention windows for selective broadcasting

    Instead of global broadcast to all neurons, intelligently
    routes information only to relevant neurons based on:
    - Role relevance to content
    - Current attention state
    - Historical activation patterns
    """

    def __init__(
        self,
        max_window_size: int = 5,
        relevance_threshold: float = 0.6,
        use_historical_patterns: bool = True
    ):
        """
        Initialize attention window manager

        Args:
            max_window_size: Maximum neurons to include in attention window
            relevance_threshold: Minimum relevance score to include neuron
            use_historical_patterns: Use activation history for windowing
        """
        self.max_window_size = max_window_size
        self.relevance_threshold = relevance_threshold
        self.use_historical_patterns = use_historical_patterns

        self.embedding_manager = EmbeddingManager(device="cpu")

        # Role affinity matrix (which roles are relevant to each other)
        self.role_affinity = self._build_role_affinity_matrix()

        # Historical co-activation patterns
        self.coactivation_counts: Dict[Tuple[str, str], int] = {}

        self.total_windows_created = 0
        self.total_broadcasts_filtered = 0

        logger.info(
            f"Initialized AttentionWindowManager "
            f"(window_size={max_window_size}, threshold={relevance_threshold})"
        )

    def _build_role_affinity_matrix(self) -> Dict[NeuronRole, Dict[NeuronRole, float]]:
        """
        Build affinity matrix between neuron roles

        Defines which roles are relevant to each other based on
        cognitive processing flow.

        Returns:
            Nested dictionary of role affinities (0-1)
        """
        affinity = {}

        # Define affinities (simplified version)
        # In full version, this could be learned from data

        # Perception roles
        perception_roles = [
            NeuronRole.VISUAL_PERCEPTION,
            NeuronRole.SEMANTIC_PERCEPTION,
            NeuronRole.PATTERN_RECOGNITION,
            NeuronRole.ANOMALY_DETECTION
        ]

        # Memory roles (using first instance of each type)
        memory_roles = [
            NeuronRole.SHORT_TERM_MEMORY_1,
            NeuronRole.EPISODIC_MEMORY_1,
            NeuronRole.SEMANTIC_MEMORY_1,
            NeuronRole.WORKING_MEMORY_1
        ]

        # Reasoning roles
        reasoning_roles = [
            NeuronRole.LOGICAL_REASONING,
            NeuronRole.CREATIVE_THINKING,
            NeuronRole.CAUSAL_ANALYSIS,
            NeuronRole.HYPOTHESIS_GENERATION
        ]

        # Action roles
        action_roles = [
            NeuronRole.ACTION_PLANNING,
            NeuronRole.DECISION_MAKING,
            NeuronRole.OUTPUT_SYNTHESIS,
            NeuronRole.SELF_CRITIQUE
        ]

        all_roles = perception_roles + memory_roles + reasoning_roles + action_roles

        # Initialize all affinities to baseline
        for role_a in all_roles:
            affinity[role_a] = {}
            for role_b in all_roles:
                if role_a == role_b:
                    affinity[role_a][role_b] = 1.0  # Self-affinity
                else:
                    affinity[role_a][role_b] = 0.3  # Baseline

        # Set higher affinities for related roles
        role_groups = [perception_roles, memory_roles, reasoning_roles, action_roles]

        # Within-group affinity
        for group in role_groups:
            for role_a in group:
                for role_b in group:
                    if role_a != role_b:
                        affinity[role_a][role_b] = 0.7

        # Cross-group affinities (processing pipeline)
        # Perception → Memory
        for perc in perception_roles:
            for mem in memory_roles:
                affinity[perc][mem] = 0.8
                affinity[mem][perc] = 0.5

        # Memory → Reasoning
        for mem in memory_roles:
            for reas in reasoning_roles:
                affinity[mem][reas] = 0.8
                affinity[reas][mem] = 0.6

        # Reasoning → Action
        for reas in reasoning_roles:
            for act in action_roles:
                affinity[reas][act] = 0.8
                affinity[act][reas] = 0.5

        # Self-critique has high affinity with all action roles
        for role in all_roles:
            affinity[NeuronRole.SELF_CRITIQUE][role] = 0.6
            affinity[role][NeuronRole.SELF_CRITIQUE] = 0.6

        return affinity

    def create_attention_window(
        self,
        content: str,
        source_neuron_id: Optional[str],
        source_role: Optional[NeuronRole],
        all_neurons: Dict[str, Any],
        current_activations: Optional[Dict[str, float]] = None
    ) -> AttentionWindow:
        """
        Create attention window for selective broadcasting

        Args:
            content: Content to broadcast
            source_neuron_id: ID of broadcasting neuron
            source_role: Role of broadcasting neuron
            all_neurons: All available neurons
            current_activations: Current activation levels

        Returns:
            Attention window with listening neurons
        """
        self.total_windows_created += 1

        # Generate content embedding
        content_embedding = self.embedding_manager.generate_embedding(content)

        # Calculate relevance scores for all neurons
        relevance_scores = {}

        for neuron_id, neuron in all_neurons.items():
            # Skip source neuron
            if neuron_id == source_neuron_id:
                continue

            # Calculate relevance
            relevance = self._calculate_relevance(
                content=content,
                content_embedding=content_embedding,
                target_neuron=neuron,
                source_role=source_role,
                current_activation=current_activations.get(neuron_id, 0.0) if current_activations else 0.0,
                source_neuron_id=source_neuron_id
            )

            relevance_scores[neuron_id] = relevance

        # Select top-k most relevant neurons above threshold
        sorted_neurons = sorted(
            relevance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        listening_neurons = set()
        filtered_scores = {}

        for neuron_id, relevance in sorted_neurons[:self.max_window_size]:
            if relevance >= self.relevance_threshold:
                listening_neurons.add(neuron_id)
                filtered_scores[neuron_id] = relevance

        # Track how many neurons were filtered out
        self.total_broadcasts_filtered += (len(all_neurons) - len(listening_neurons))

        window = AttentionWindow(
            content_embedding=content_embedding,
            listening_neurons=listening_neurons,
            relevance_scores=filtered_scores,
            window_size=len(listening_neurons)
        )

        logger.debug(
            f"Created attention window: {len(listening_neurons)}/{len(all_neurons)} neurons listening "
            f"(filtered {len(all_neurons) - len(listening_neurons)})"
        )

        return window

    def _calculate_relevance(
        self,
        content: str,
        content_embedding: np.ndarray,
        target_neuron: Any,
        source_role: Optional[NeuronRole],
        current_activation: float,
        source_neuron_id: Optional[str]
    ) -> float:
        """
        Calculate relevance of content to target neuron

        Combines multiple factors:
        1. Semantic similarity (content vs neuron role)
        2. Role affinity (source role vs target role)
        3. Current activation level
        4. Historical co-activation patterns

        Args:
            content: Broadcast content
            content_embedding: Content embedding
            target_neuron: Target neuron
            source_role: Source neuron's role
            current_activation: Target's current activation
            source_neuron_id: Source neuron ID

        Returns:
            Relevance score (0-1)
        """
        factors = []

        # Factor 1: Semantic similarity
        target_role_embedding = self.embedding_manager.generate_embedding(
            target_neuron.role.value.replace('_', ' ')
        )
        semantic_similarity = self.embedding_manager.cosine_similarity(
            content_embedding,
            target_role_embedding
        )
        factors.append(('semantic', semantic_similarity, 0.4))

        # Factor 2: Role affinity
        if source_role and source_role in self.role_affinity:
            role_affinity = self.role_affinity[source_role].get(
                target_neuron.role,
                0.3
            )
            factors.append(('role_affinity', role_affinity, 0.3))

        # Factor 3: Current activation (boosted if already active)
        activation_boost = current_activation * 0.5
        factors.append(('activation', activation_boost, 0.2))

        # Factor 4: Historical co-activation
        if self.use_historical_patterns and source_neuron_id:
            coactivation_score = self._get_coactivation_score(
                source_neuron_id,
                target_neuron.neuron_id
            )
            factors.append(('coactivation', coactivation_score, 0.1))

        # Weighted combination
        total_relevance = sum(score * weight for _, score, weight in factors)

        return min(1.0, max(0.0, total_relevance))

    def _get_coactivation_score(
        self,
        neuron_a_id: str,
        neuron_b_id: str
    ) -> float:
        """
        Get historical co-activation score

        Args:
            neuron_a_id: First neuron ID
            neuron_b_id: Second neuron ID

        Returns:
            Co-activation score (0-1)
        """
        # Check both orderings
        key1 = (neuron_a_id, neuron_b_id)
        key2 = (neuron_b_id, neuron_a_id)

        count = self.coactivation_counts.get(key1, 0) + self.coactivation_counts.get(key2, 0)

        # Normalize (assume max co-activations is 100)
        return min(1.0, count / 100.0)

    def record_coactivation(self, neuron_a_id: str, neuron_b_id: str):
        """
        Record that two neurons fired together

        Args:
            neuron_a_id: First neuron
            neuron_b_id: Second neuron
        """
        key = (neuron_a_id, neuron_b_id)
        self.coactivation_counts[key] = self.coactivation_counts.get(key, 0) + 1

    def get_listening_neurons(
        self,
        window: AttentionWindow,
        include_scores: bool = False
    ) -> List[str] | List[Tuple[str, float]]:
        """
        Get neurons that should receive broadcast

        Args:
            window: Attention window
            include_scores: Whether to include relevance scores

        Returns:
            List of neuron IDs (or tuples with scores)
        """
        if include_scores:
            return [
                (nid, window.relevance_scores.get(nid, 0.0))
                for nid in window.listening_neurons
            ]
        else:
            return list(window.listening_neurons)

    def get_stats(self) -> Dict[str, Any]:
        """Get attention windowing statistics"""
        avg_filtered = (
            self.total_broadcasts_filtered / self.total_windows_created
            if self.total_windows_created > 0
            else 0
        )

        return {
            "total_windows_created": self.total_windows_created,
            "total_broadcasts_filtered": self.total_broadcasts_filtered,
            "avg_broadcasts_filtered": avg_filtered,
            "max_window_size": self.max_window_size,
            "relevance_threshold": self.relevance_threshold,
            "coactivation_patterns": len(self.coactivation_counts)
        }

    def __repr__(self) -> str:
        return (
            f"AttentionWindowManager(windows={self.total_windows_created}, "
            f"filtered={self.total_broadcasts_filtered})"
        )
