"""
Global Workspace Theory (GWT) Implementation

Implements Baars' Global Workspace Theory for conscious processing:
- Competitive selection of information for broadcast
- Global workspace for information integration
- Conscious vs unconscious processing
- Attention mechanisms

Reference: Baars, B. J. (1988). A cognitive theory of consciousness.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging

from llm_swarm_brain.neuron import Phi3Neuron, NeuronSignal
from llm_swarm_brain.utils import calculate_global_workspace_salience, softmax


logger = logging.getLogger(__name__)


@dataclass
class BroadcastMessage:
    """Information broadcast in the global workspace"""
    content: str
    source_neuron_id: str
    salience: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GlobalWorkspace:
    """
    Global Workspace for conscious information processing

    Implements:
    - Competitive selection for broadcast
    - Global broadcast to all neurons
    - Consciousness threshold for awareness
    - Attention modulation
    """

    def __init__(
        self,
        capacity: int = 5,
        broadcast_threshold: float = 0.7,
        consciousness_threshold: float = 0.8
    ):
        """
        Initialize global workspace

        Args:
            capacity: Maximum simultaneous broadcasts
            broadcast_threshold: Minimum salience for broadcast
            consciousness_threshold: Threshold for conscious processing
        """
        self.capacity = capacity
        self.broadcast_threshold = broadcast_threshold
        self.consciousness_threshold = consciousness_threshold

        # Current broadcasts
        self.active_broadcasts: List[BroadcastMessage] = []

        # History
        self.broadcast_history: List[BroadcastMessage] = []

        # Conscious content (high salience broadcasts)
        self.conscious_content: List[BroadcastMessage] = []

        # Attention state
        self.attention_focus: Set[str] = set()  # Neuron IDs with attentional boost

        logger.info(
            f"Initialized GlobalWorkspace (capacity={capacity}, "
            f"broadcast_threshold={broadcast_threshold})"
        )

    def compete_for_broadcast(
        self,
        neuron_outputs: Dict[str, str],
        neuron_activations: Dict[str, float]
    ) -> List[BroadcastMessage]:
        """
        Competition for global workspace access

        Only the most salient information wins broadcast rights.

        Args:
            neuron_outputs: Dictionary mapping neuron_id to output
            neuron_activations: Dictionary mapping neuron_id to activation level

        Returns:
            List of messages selected for broadcast
        """
        # Calculate salience for each neuron's output
        salience_scores = calculate_global_workspace_salience(
            neuron_outputs,
            neuron_activations
        )

        # Apply attention modulation (boost attended neurons)
        modulated_scores = []
        for neuron_id, salience in salience_scores:
            if neuron_id in self.attention_focus:
                salience *= 1.5  # Attentional boost

            modulated_scores.append((neuron_id, salience))

        # Sort by salience
        modulated_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top-k for broadcast
        selected_broadcasts = []
        for neuron_id, salience in modulated_scores[:self.capacity]:
            if salience >= self.broadcast_threshold:
                broadcast = BroadcastMessage(
                    content=neuron_outputs[neuron_id],
                    source_neuron_id=neuron_id,
                    salience=salience,
                    metadata={
                        "activation": neuron_activations.get(neuron_id, 0.0),
                        "attended": neuron_id in self.attention_focus
                    }
                )

                selected_broadcasts.append(broadcast)

                # Check if this qualifies as conscious content
                if salience >= self.consciousness_threshold:
                    self.conscious_content.append(broadcast)
                    logger.info(
                        f"Conscious broadcast from {neuron_id} "
                        f"(salience={salience:.3f})"
                    )

        return selected_broadcasts

    def broadcast(
        self,
        messages: List[BroadcastMessage],
        all_neurons: Dict[str, Phi3Neuron]
    ):
        """
        Broadcast messages to all neurons in the network

        This is the core GWT mechanism: winning information
        is globally broadcast and becomes available to all processors.

        Args:
            messages: Messages to broadcast
            all_neurons: All neurons in the network
        """
        # Update active broadcasts
        self.active_broadcasts = messages

        # Add to history
        self.broadcast_history.extend(messages)

        # Create global context from broadcasts
        global_context = self._create_global_context(messages)

        # Broadcast to all neurons (global availability)
        for neuron in all_neurons.values():
            for message in messages:
                # Don't send a neuron its own broadcast
                if message.source_neuron_id != neuron.neuron_id:
                    signal = NeuronSignal(
                        content=message.content,
                        source_role=None,  # Global workspace source
                        activation_level=message.salience,
                        metadata={
                            **message.metadata,
                            "is_global_broadcast": True,
                            "global_context": global_context
                        }
                    )
                    neuron.receive_signal(signal, source_weight=1.0)

        logger.debug(f"Broadcast {len(messages)} messages globally")

    def _create_global_context(self, broadcasts: List[BroadcastMessage]) -> Dict[str, Any]:
        """
        Create integrated global context from broadcasts

        Args:
            broadcasts: Current broadcasts

        Returns:
            Global context dictionary
        """
        return {
            "broadcast_count": len(broadcasts),
            "max_salience": max([b.salience for b in broadcasts]) if broadcasts else 0.0,
            "source_neurons": [b.source_neuron_id for b in broadcasts],
            "attention_focus": list(self.attention_focus),
            "conscious_processing": any(
                b.salience >= self.consciousness_threshold for b in broadcasts
            )
        }

    def set_attention(self, neuron_ids: List[str]):
        """
        Set attention focus to specific neurons

        Attended neurons receive salience boost in competition.

        Args:
            neuron_ids: List of neuron IDs to attend to
        """
        self.attention_focus = set(neuron_ids)
        logger.info(f"Attention focus set to: {neuron_ids}")

    def update_attention(self, recent_activations: Dict[str, float], top_k: int = 3):
        """
        Update attention based on recent activations (bottom-up attention)

        Args:
            recent_activations: Recent neuron activations
            top_k: Number of top neurons to attend
        """
        # Sort by activation
        sorted_neurons = sorted(
            recent_activations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Focus on top-k
        self.attention_focus = set([nid for nid, _ in sorted_neurons[:top_k]])

    def get_conscious_summary(self) -> str:
        """
        Get summary of conscious content

        Returns:
            String summary of recent conscious broadcasts
        """
        if not self.conscious_content:
            return "No conscious content yet."

        recent_conscious = self.conscious_content[-5:]  # Last 5 conscious broadcasts

        summary_lines = ["=== CONSCIOUS CONTENT ==="]
        for i, broadcast in enumerate(recent_conscious, 1):
            summary_lines.append(
                f"{i}. [{broadcast.source_neuron_id}] "
                f"(salience={broadcast.salience:.3f}): "
                f"{broadcast.content[:100]}..."
            )

        return "\n".join(summary_lines)

    def get_workspace_state(self) -> Dict[str, Any]:
        """Get current workspace state"""
        return {
            "active_broadcasts": len(self.active_broadcasts),
            "total_broadcasts": len(self.broadcast_history),
            "conscious_broadcasts": len(self.conscious_content),
            "attention_focus": list(self.attention_focus),
            "current_saliences": [b.salience for b in self.active_broadcasts],
            "consciousness_active": any(
                b.salience >= self.consciousness_threshold
                for b in self.active_broadcasts
            )
        }

    def clear_broadcasts(self):
        """Clear active broadcasts (new moment of processing)"""
        self.active_broadcasts.clear()

    def __repr__(self) -> str:
        return (
            f"GlobalWorkspace(active={len(self.active_broadcasts)}, "
            f"conscious={len(self.conscious_content)}, "
            f"attention={len(self.attention_focus)})"
        )


class ConsciousnessMonitor:
    """
    Monitors consciousness-related metrics

    Tracks:
    - Levels of processing (conscious vs unconscious)
    - Integration across workspace
    - Temporal dynamics of consciousness
    """

    def __init__(self):
        self.consciousness_timeline: List[Tuple[datetime, float]] = []
        self.integration_scores: List[float] = []

    def record_consciousness_level(self, level: float):
        """
        Record current consciousness level

        Args:
            level: Consciousness level (0-1)
        """
        self.consciousness_timeline.append((datetime.now(), level))

    def calculate_integration(
        self,
        broadcasts: List[BroadcastMessage]
    ) -> float:
        """
        Calculate level of integration across broadcasts

        Measures how well information from different sources
        is integrated in the workspace.

        Args:
            broadcasts: Current broadcasts

        Returns:
            Integration score (0-1)
        """
        if len(broadcasts) < 2:
            return 0.0

        # Diversity of sources
        unique_sources = len(set(b.source_neuron_id for b in broadcasts))
        source_diversity = unique_sources / len(broadcasts)

        # Salience uniformity (more uniform = better integration)
        saliences = [b.salience for b in broadcasts]
        salience_std = np.std(saliences)
        salience_uniformity = 1.0 - min(1.0, salience_std)

        # Combined integration score
        integration = (source_diversity + salience_uniformity) / 2.0

        self.integration_scores.append(integration)

        return integration

    def get_average_consciousness_level(self, window_seconds: float = 10.0) -> float:
        """
        Get average consciousness level over recent time window

        Args:
            window_seconds: Time window in seconds

        Returns:
            Average consciousness level
        """
        if not self.consciousness_timeline:
            return 0.0

        now = datetime.now()
        recent = [
            level for timestamp, level in self.consciousness_timeline
            if (now - timestamp).total_seconds() <= window_seconds
        ]

        return np.mean(recent) if recent else 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get consciousness metrics"""
        return {
            "total_measurements": len(self.consciousness_timeline),
            "current_level": self.consciousness_timeline[-1][1] if self.consciousness_timeline else 0.0,
            "average_level": np.mean([l for _, l in self.consciousness_timeline]) if self.consciousness_timeline else 0.0,
            "average_integration": np.mean(self.integration_scores) if self.integration_scores else 0.0
        }
