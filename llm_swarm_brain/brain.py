"""
PhiBrain: Complete LLM-Swarm-Brain Architecture

Integrates all components:
- 16 Phi-3 neurons (8 per GPU)
- Neural orchestration
- Global Workspace Theory
- Integrated Information Theory
- Memory systems
- Positronic Dialectical Logic-Gated Coherence Framework
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from llm_swarm_brain.neuron import Phi3Neuron
from llm_swarm_brain.orchestrator import NeuralOrchestrator
from llm_swarm_brain.gw_theory import GlobalWorkspace, ConsciousnessMonitor
from llm_swarm_brain.positronic_framework import PositronicFramework
from llm_swarm_brain.config import (
    BrainConfig,
    NeuronRole,
    NEURON_ARCHITECTURE,
    DEFAULT_CONNECTIONS
)
from llm_swarm_brain.utils import CircularBuffer, calculate_phi


logger = logging.getLogger(__name__)


class MemorySystem:
    """
    Multi-level memory system

    Implements:
    - Short-term memory (immediate context)
    - Episodic memory (event sequences)
    - Semantic memory (knowledge)
    - Memory consolidation
    """

    def __init__(
        self,
        short_term_capacity: int = 10,
        episodic_capacity: int = 100
    ):
        self.short_term = CircularBuffer(short_term_capacity)
        self.episodic = CircularBuffer(episodic_capacity)
        self.semantic: Dict[str, Any] = {}

        self.consolidation_count = 0

    def add_to_short_term(self, item: Any):
        """Add item to short-term memory"""
        self.short_term.append(item)

    def add_to_episodic(self, episode: Dict[str, Any]):
        """Add episode to episodic memory"""
        episode["timestamp"] = datetime.now()
        self.episodic.append(episode)

    def add_to_semantic(self, key: str, value: Any):
        """Add knowledge to semantic memory"""
        self.semantic[key] = value

    def consolidate(self):
        """
        Consolidate memories (short-term → episodic/semantic)

        Simplified consolidation: recent short-term items
        become episodic memories.
        """
        recent_items = self.short_term.get_recent(5)

        for item in recent_items:
            episode = {
                "content": item,
                "consolidated_at": datetime.now()
            }
            self.add_to_episodic(episode)

        self.consolidation_count += 1
        logger.info(f"Memory consolidation #{self.consolidation_count} complete")

    def get_context(self, max_items: int = 5) -> Dict[str, Any]:
        """Get memory context for processing"""
        return {
            "short_term": self.short_term.get_recent(max_items),
            "recent_episodes": self.episodic.get_recent(3),
            "semantic_keys": list(self.semantic.keys())[:10]
        }


class PhiBrain:
    """
    Complete LLM-Swarm-Brain with 16 Phi-3 neurons

    Architecture:
    - GPU 0: Perception (4) + Memory (4) neurons
    - GPU 1: Reasoning (4) + Action (4) neurons

    Features:
    - Global Workspace Theory for conscious processing
    - Integrated Information Theory metrics
    - Multi-level memory system
    - Hebbian learning
    - Attention mechanisms
    - Positronic Dialectical Logic-Gated Coherence Framework
    """

    def __init__(
        self,
        config: Optional[BrainConfig] = None,
        load_models: bool = True,
        enable_positronic: bool = True
    ):
        """
        Initialize PhiBrain

        Args:
            config: Brain configuration (uses default if None)
            load_models: Whether to load models immediately
            enable_positronic: Enable positronic dialectical framework
        """
        self.config = config or BrainConfig()
        self.load_models = load_models
        self.enable_positronic = enable_positronic

        # Core components
        self.orchestrator = NeuralOrchestrator(self.config)
        self.global_workspace = GlobalWorkspace(
            capacity=self.config.global_workspace_capacity,
            broadcast_threshold=self.config.broadcast_threshold,
            consciousness_threshold=self.config.consciousness_threshold
        )
        self.consciousness_monitor = ConsciousnessMonitor()
        self.memory = MemorySystem(
            short_term_capacity=self.config.short_term_capacity,
            episodic_capacity=self.config.episodic_capacity
        )

        # Positronic framework for dialectical reasoning and coherence
        self.positronic = PositronicFramework(
            coherence_threshold=0.7,
            enable_dialectical_reasoning=enable_positronic,
            enforce_positronic_laws=enable_positronic
        ) if enable_positronic else None

        # Neurons (initialized in _initialize_neurons)
        self.perception_neurons: List[Phi3Neuron] = []
        self.memory_neurons: List[Phi3Neuron] = []
        self.reasoning_neurons: List[Phi3Neuron] = []
        self.action_neurons: List[Phi3Neuron] = []

        # Processing state
        self.processing_count = 0
        self.total_consciousness_level = 0.0

        # Initialize the brain
        self._initialize_neurons()
        self._setup_network()

        logger.info("PhiBrain initialized successfully")

    def _initialize_neurons(self):
        """Initialize all 16 neurons according to architecture"""
        neuron_count = 0

        # GPU 0: Perception + Memory layers
        for layer_name, roles in NEURON_ARCHITECTURE["gpu_0"].items():
            for role in roles:
                neuron_id = f"gpu0_{layer_name}_{role.value}"
                neuron = Phi3Neuron(
                    role=role,
                    gpu_id=0,
                    neuron_id=neuron_id,
                    activation_threshold=self.config.activation_threshold,
                    model_name=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    load_model=self.load_models
                )

                self.orchestrator.add_neuron(neuron)

                if layer_name == "perception":
                    self.perception_neurons.append(neuron)
                else:  # memory
                    self.memory_neurons.append(neuron)

                neuron_count += 1

        # GPU 1: Reasoning + Action layers
        for layer_name, roles in NEURON_ARCHITECTURE["gpu_1"].items():
            for role in roles:
                neuron_id = f"gpu1_{layer_name}_{role.value}"
                neuron = Phi3Neuron(
                    role=role,
                    gpu_id=1,
                    neuron_id=neuron_id,
                    activation_threshold=self.config.activation_threshold,
                    model_name=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    load_model=self.load_models
                )

                self.orchestrator.add_neuron(neuron)

                if layer_name == "reasoning":
                    self.reasoning_neurons.append(neuron)
                else:  # action
                    self.action_neurons.append(neuron)

                neuron_count += 1

        logger.info(f"Initialized {neuron_count} neurons across 2 GPUs")

    def _setup_network(self):
        """Setup neural connections"""
        self.orchestrator.setup_connections(DEFAULT_CONNECTIONS)

    def think(
        self,
        input_text: str,
        max_steps: int = None,
        use_memory: bool = True,
        use_global_workspace: bool = True
    ) -> Dict[str, Any]:
        """
        Process input through the brain

        This is the main thinking/processing method.

        Args:
            input_text: Input text to process
            max_steps: Maximum processing steps
            use_memory: Whether to use memory system
            use_global_workspace: Whether to use global workspace

        Returns:
            Dictionary with complete processing results
        """
        self.processing_count += 1
        logger.info(f"=== Processing #{self.processing_count}: '{input_text[:50]}...' ===")

        # Prepare global context
        global_context = {}

        if use_memory:
            global_context["memory"] = self.memory.get_context()

        # Initial processing through neural network
        processing_result = self.orchestrator.process(
            input_text=input_text,
            max_steps=max_steps,
            global_context=global_context
        )

        # Global workspace processing (if enabled)
        workspace_result = None
        if use_global_workspace:
            workspace_result = self._process_global_workspace(processing_result)

        # Update memory
        if use_memory:
            self.memory.add_to_short_term(input_text)

            # Memory consolidation (periodic)
            if self.processing_count % self.config.consolidation_interval == 0:
                self.memory.consolidate()

        # Calculate consciousness level
        consciousness_level = self._calculate_consciousness_level(
            processing_result,
            workspace_result
        )
        self.consciousness_monitor.record_consciousness_level(consciousness_level)
        self.total_consciousness_level += consciousness_level

        # Compile complete result
        result = {
            "input": input_text,
            "processing_id": self.processing_count,
            "timestamp": datetime.now(),
            "neural_processing": processing_result,
            "global_workspace": workspace_result,
            "consciousness_level": consciousness_level,
            "memory_context": self.memory.get_context() if use_memory else None,
            "brain_metrics": self._get_brain_metrics()
        }

        return result

    def _process_global_workspace(
        self,
        processing_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process through global workspace (GWT)

        Args:
            processing_result: Results from neural processing

        Returns:
            Global workspace processing results
        """
        # Extract neuron outputs and activations from final step
        if not processing_result["steps"]:
            return None

        final_step = processing_result["steps"][-1]
        neuron_outputs = final_step["outputs"]
        neuron_activations = final_step["activations"]

        # Competition for broadcast
        broadcasts = self.global_workspace.compete_for_broadcast(
            neuron_outputs,
            neuron_activations
        )

        # Global broadcast
        if broadcasts:
            self.global_workspace.broadcast(
                broadcasts,
                self.orchestrator.neurons
            )

        # Update attention (bottom-up)
        self.global_workspace.update_attention(neuron_activations)

        # Calculate integration
        integration = self.consciousness_monitor.calculate_integration(broadcasts)

        # Positronic framework: Coherence validation
        coherence_report = None
        dialectical_result = None

        if self.positronic:
            # Validate coherence
            coherence_report = self.positronic.validate_coherence(
                outputs=neuron_outputs,
                activations=neuron_activations,
                prior_context=self.memory.get_context() if self.memory else None,
                enforce=True
            )

            # Apply dialectical reasoning to top broadcast
            if broadcasts and broadcasts[0].salience >= self.config.consciousness_threshold:
                dialectical_result = self.positronic.apply_dialectical_reasoning(
                    thesis_output=broadcasts[0].content,
                    thesis_activation=broadcasts[0].salience,
                    generate_antithesis=True
                )

        return {
            "broadcasts": [
                {
                    "source": b.source_neuron_id,
                    "salience": b.salience,
                    "content": b.content[:200] + "..." if len(b.content) > 200 else b.content,
                    "is_conscious": b.salience >= self.config.consciousness_threshold
                }
                for b in broadcasts
            ],
            "integration_score": integration,
            "workspace_state": self.global_workspace.get_workspace_state(),
            "conscious_summary": self.global_workspace.get_conscious_summary(),
            "positronic_coherence": {
                "is_coherent": coherence_report.is_coherent if coherence_report else True,
                "coherence_score": coherence_report.coherence_score if coherence_report else 1.0,
                "violations": coherence_report.violations if coherence_report else [],
                "logical_consistency": coherence_report.logical_consistency if coherence_report else 1.0
            } if self.positronic else None,
            "dialectical_synthesis": {
                "thesis": dialectical_result.thesis[:100] + "..." if dialectical_result and len(dialectical_result.thesis) > 100 else dialectical_result.thesis if dialectical_result else None,
                "antithesis": dialectical_result.antithesis[:100] + "..." if dialectical_result and len(dialectical_result.antithesis) > 100 else dialectical_result.antithesis if dialectical_result else None,
                "synthesis": dialectical_result.synthesis[:100] + "..." if dialectical_result and len(dialectical_result.synthesis) > 100 else dialectical_result.synthesis if dialectical_result else None,
                "confidence": dialectical_result.confidence if dialectical_result else 0.0
            } if self.positronic and dialectical_result else None
        }

    def _calculate_consciousness_level(
        self,
        processing_result: Dict[str, Any],
        workspace_result: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall consciousness level

        Based on:
        - Network integration (Phi from IIT)
        - Global workspace activity
        - Broadcast salience

        Args:
            processing_result: Neural processing results
            workspace_result: Global workspace results

        Returns:
            Consciousness level (0-1)
        """
        factors = []

        # Factor 1: Network integration (IIT)
        if "network_metrics" in processing_result:
            phi = processing_result["network_metrics"].get("phi", 0.0)
            factors.append(min(1.0, phi / self.config.phi_threshold))

        # Factor 2: Global workspace activity
        if workspace_result and "broadcasts" in workspace_result:
            broadcast_count = len(workspace_result["broadcasts"])
            max_salience = max(
                [b["salience"] for b in workspace_result["broadcasts"]],
                default=0.0
            )
            factors.append(max_salience)

        # Factor 3: Network activation
        if "network_metrics" in processing_result:
            activation = processing_result["network_metrics"].get("mean_activation", 0.0)
            factors.append(activation)

        # Combine factors
        consciousness_level = sum(factors) / len(factors) if factors else 0.0

        return consciousness_level

    def _get_brain_metrics(self) -> Dict[str, Any]:
        """Get comprehensive brain metrics"""
        metrics = {
            "network": self.orchestrator.get_network_stats(),
            "consciousness": self.consciousness_monitor.get_metrics(),
            "global_workspace": self.global_workspace.get_workspace_state(),
            "memory": {
                "short_term_size": len(self.memory.short_term.get_all()),
                "episodic_size": len(self.memory.episodic.get_all()),
                "semantic_size": len(self.memory.semantic),
                "consolidations": self.memory.consolidation_count
            },
            "processing_count": self.processing_count,
            "avg_consciousness": self.total_consciousness_level / max(1, self.processing_count)
        }

        # Add positronic framework metrics
        if self.positronic:
            metrics["positronic"] = self.positronic.get_framework_stats()

        return metrics

    def get_summary(self) -> str:
        """Get human-readable summary of brain state"""
        lines = ["=" * 70]
        lines.append("PHI-3 SWARM BRAIN SUMMARY")
        lines.append("=" * 70)

        # Architecture
        lines.append("\n📊 ARCHITECTURE:")
        lines.append(f"  Total Neurons: {len(self.orchestrator.neurons)}")
        lines.append(f"  - Perception: {len(self.perception_neurons)} (GPU 0)")
        lines.append(f"  - Memory: {len(self.memory_neurons)} (GPU 0)")
        lines.append(f"  - Reasoning: {len(self.reasoning_neurons)} (GPU 1)")
        lines.append(f"  - Action: {len(self.action_neurons)} (GPU 1)")

        # Processing stats
        lines.append(f"\n🧠 PROCESSING:")
        lines.append(f"  Total Thoughts: {self.processing_count}")
        lines.append(f"  Avg Consciousness: {self.total_consciousness_level / max(1, self.processing_count):.3f}")

        # Network stats
        network_stats = self.orchestrator.get_network_stats()
        lines.append(f"\n🔗 NETWORK:")
        lines.append(f"  Total Connections: {network_stats['total_connections']}")
        lines.append(f"  Active Neurons: {network_stats['active_neurons']}")
        lines.append(f"  Total Propagations: {network_stats['total_propagations']}")

        # Global workspace
        gw_state = self.global_workspace.get_workspace_state()
        lines.append(f"\n💭 CONSCIOUSNESS:")
        lines.append(f"  Active Broadcasts: {gw_state['active_broadcasts']}")
        lines.append(f"  Total Broadcasts: {gw_state['total_broadcasts']}")
        lines.append(f"  Conscious Broadcasts: {gw_state['conscious_broadcasts']}")

        # Memory
        lines.append(f"\n💾 MEMORY:")
        lines.append(f"  Short-term: {len(self.memory.short_term.get_all())} items")
        lines.append(f"  Episodic: {len(self.memory.episodic.get_all())} episodes")
        lines.append(f"  Semantic: {len(self.memory.semantic)} entries")

        lines.append("=" * 70)

        return "\n".join(lines)

    def visualize_state(self) -> str:
        """Visualize current brain state"""
        return self.orchestrator.visualize_activations()

    def reset(self):
        """Reset brain to initial state"""
        self.orchestrator.reset_all_neurons()
        self.global_workspace.clear_broadcasts()
        self.memory.short_term.clear()
        logger.info("Brain reset complete")

    def __repr__(self) -> str:
        return (
            f"PhiBrain(neurons={len(self.orchestrator.neurons)}, "
            f"thoughts={self.processing_count}, "
            f"consciousness={self.total_consciousness_level / max(1, self.processing_count):.3f})"
        )
