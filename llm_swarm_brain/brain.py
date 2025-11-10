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
from llm_swarm_brain.neuron_api import APINeuron
from llm_swarm_brain.orchestrator import NeuralOrchestrator
from llm_swarm_brain.gw_theory import GlobalWorkspace, ConsciousnessMonitor
from llm_swarm_brain.positronic_framework import PositronicFramework
from llm_swarm_brain.summarization import SummarizationNeuron
from llm_swarm_brain.attention_windowing import AttentionWindowManager
from llm_swarm_brain.conceptual_threading import ConceptualThreadTracker
from llm_swarm_brain.meta_orchestration import MetaOrchestrator, PerformanceMetrics
from llm_swarm_brain.config import (
    BrainConfig,
    NeuronRole,
    NEURON_ARCHITECTURE,
    DEFAULT_CONNECTIONS
)

# Support for 64-neuron config
try:
    from llm_swarm_brain import config_64
    HAS_64_CONFIG = True
except ImportError:
    HAS_64_CONFIG = False
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
        enable_positronic: bool = True,
        use_api: bool = False,
        api_key: Optional[str] = None,
        use_64_neurons: bool = False
    ):
        """
        Initialize PhiBrain

        Args:
            config: Brain configuration (uses default if None)
            load_models: Whether to load models immediately (ignored if use_api=True)
            enable_positronic: Enable positronic dialectical framework
            use_api: Use API-based neurons instead of local models
            api_key: API key for Hyperbolic (or set HYPERBOLIC_API_KEY env var)
            use_64_neurons: Use 64-neuron architecture instead of 8-neuron
        """
        # Use 64-neuron config if requested
        if use_64_neurons:
            if not HAS_64_CONFIG:
                raise ImportError("64-neuron config not available")
            self.config = config or config_64.BrainConfig()
            self._neuron_architecture = config_64.NEURON_ARCHITECTURE
            self._default_connections = config_64.DEFAULT_CONNECTIONS
            self._role_prompts = config_64.ROLE_PROMPTS
            self._neuron_role_enum = config_64.NeuronRole
        else:
            self.config = config or BrainConfig()
            self._neuron_architecture = NEURON_ARCHITECTURE
            self._default_connections = DEFAULT_CONNECTIONS
            from llm_swarm_brain.config import ROLE_PROMPTS
            self._role_prompts = ROLE_PROMPTS
            self._neuron_role_enum = NeuronRole
        
        self.use_api = use_api
        self.api_key = api_key
        self.use_64_neurons = use_64_neurons
        self.load_models = load_models if not use_api else False
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

        # NEW ENHANCEMENTS
        # Summarization neuron for output compression
        self.summarization_neuron = SummarizationNeuron(
            max_output_length=200,
            compression_threshold=300,
            preserve_key_concepts=True
        )

        # Attention windowing for selective broadcasting
        self.attention_manager = AttentionWindowManager(
            max_window_size=5,
            relevance_threshold=0.6,
            use_historical_patterns=True
        )

        # Conceptual thread tracking
        self.concept_tracker = ConceptualThreadTracker(
            similarity_threshold=0.7,
            max_threads_per_concept=5
        )

        # Meta-orchestration for dynamic tuning
        self.meta_orchestrator = MetaOrchestrator(
            adaptation_rate=0.1,
            enable_auto_tuning=True
        )

        # Neurons (initialized in _initialize_neurons)
        self.perception_neurons: List[Phi3Neuron] = []
        self.memory_neurons: List[Phi3Neuron] = []
        self.reasoning_neurons: List[Phi3Neuron] = []
        self.action_neurons: List[Phi3Neuron] = []

        # Processing state
        self.processing_count = 0
        self.total_consciousness_level = 0.0
        self.processing_start_time: Optional[datetime] = None

        # Initialize the brain
        self._initialize_neurons()
        self._setup_network()

        logger.info("PhiBrain initialized successfully with all enhancements")

    def _initialize_neurons(self):
        """Initialize all neurons according to architecture (API or local)"""
        neuron_count = 0
        
        # Choose neuron class based on mode
        NeuronClass = APINeuron if self.use_api else Phi3Neuron
        mode_str = "API-based" if self.use_api else "local GPU-based"
        arch_str = "64-neuron" if self.use_64_neurons else "8-neuron"

        # Iterate through all GPUs (or API endpoints)
        for gpu_id in range(self.config.gpu_count):
            gpu_key = f"gpu_{gpu_id}"

            if gpu_key not in self._neuron_architecture:
                continue

            # Get layers for this GPU
            layers = self._neuron_architecture[gpu_key]

            for layer_name, roles in layers.items():
                for role in roles:
                    neuron_id = f"n{gpu_id}_{layer_name}_{role.value}"
                    
                    if self.use_api:
                        neuron = APINeuron(
                            role=role,
                            gpu_id=gpu_id,
                            neuron_id=neuron_id,
                            activation_threshold=self.config.activation_threshold,
                            model_name=self.config.model_name,
                            max_tokens=self.config.max_tokens,
                            temperature=self.config.temperature,
                            api_key=self.api_key
                        )
                    else:
                        neuron = Phi3Neuron(
                            role=role,
                            gpu_id=gpu_id,
                            neuron_id=neuron_id,
                            activation_threshold=self.config.activation_threshold,
                            model_name=self.config.model_name,
                            max_tokens=self.config.max_tokens,
                            temperature=self.config.temperature,
                            load_model=self.load_models
                        )

                    self.orchestrator.add_neuron(neuron)

                    # Categorize neurons by layer
                    if "perception" in layer_name or "attention" in layer_name:
                        self.perception_neurons.append(neuron)
                    elif "memory" in layer_name:
                        self.memory_neurons.append(neuron)
                    elif "reasoning" in layer_name or "creative" in layer_name or "analytical" in layer_name:
                        self.reasoning_neurons.append(neuron)
                    elif "synthesis" in layer_name or "meta" in layer_name or "meta_cognitive" in layer_name:
                        self.action_neurons.append(neuron)

                    neuron_count += 1

        logger.info(
            f"Initialized {neuron_count} {mode_str} neurons ({arch_str} architecture) "
            f"(Perception: {len(self.perception_neurons)}, "
            f"Memory: {len(self.memory_neurons)}, "
            f"Reasoning: {len(self.reasoning_neurons)}, "
            f"Synthesis/Meta: {len(self.action_neurons)})"
        )

    def _setup_network(self):
        """Setup neural connections"""
        self.orchestrator.setup_connections(self._default_connections)

    def think(
        self,
        input_text: str,
        max_steps: int = None,
        use_memory: bool = True,
        use_global_workspace: bool = True,
        enable_enhancements: bool = True
    ) -> Dict[str, Any]:
        """
        Process input through the brain

        This is the main thinking/processing method with full enhancements.

        Args:
            input_text: Input text to process
            max_steps: Maximum processing steps
            use_memory: Whether to use memory system
            use_global_workspace: Whether to use global workspace
            enable_enhancements: Enable new enhancements (summarization, etc.)

        Returns:
            Dictionary with complete processing results
        """
        self.processing_count += 1
        self.processing_start_time = datetime.now()
        logger.info(f"=== Processing #{self.processing_count}: '{input_text[:50]}...' ===")

        # === META-ORCHESTRATION: Analyze task complexity ===
        complexity_estimate = None
        adjusted_params = None

        if enable_enhancements:
            memory_context = self.memory.get_context() if use_memory else None
            complexity_estimate = self.meta_orchestrator.analyze_task(
                input_text,
                context={"memory": memory_context} if memory_context else None
            )

            # Get recommendations
            current_performance = self._get_current_performance()
            param_recommendations = self.meta_orchestrator.recommend_parameters(
                complexity_estimate,
                current_performance
            )

            # Apply adjustments
            adjusted_params = self.meta_orchestrator.apply_adjustments(param_recommendations)

            # Use adjusted parameters
            if max_steps is None:
                max_steps = adjusted_params["max_propagation_steps"]

            logger.info(
                f"Task complexity: {complexity_estimate.complexity_score:.2f}, "
                f"Adjusted threshold: {adjusted_params['activation_threshold']:.2f}"
            )

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

        # === CONCEPT TRACKING: Track concepts in neuron outputs ===
        if enable_enhancements and processing_result["steps"]:
            for step in processing_result["steps"]:
                for neuron_id, output in step["outputs"].items():
                    if output:
                        activation = step["activations"].get(neuron_id, 0.0)
                        self.concept_tracker.process_neuron_output(
                            neuron_id=neuron_id,
                            output_text=output,
                            activation_level=activation
                        )

        # Global workspace processing (if enabled)
        workspace_result = None
        if use_global_workspace:
            workspace_result = self._process_global_workspace(
                processing_result,
                enable_enhancements=enable_enhancements
            )

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

        # === PERFORMANCE TRACKING ===
        processing_time = (datetime.now() - self.processing_start_time).total_seconds()

        if enable_enhancements:
            # Record performance for meta-orchestration
            performance = PerformanceMetrics(
                avg_consciousness_level=consciousness_level,
                avg_coherence_score=(
                    workspace_result.get("positronic_coherence", {}).get("coherence_score", 0.0)
                    if workspace_result else 0.0
                ),
                avg_integration_score=(
                    workspace_result.get("integration_score", 0.0)
                    if workspace_result else 0.0
                ),
                avg_neurons_activated=(
                    processing_result["network_metrics"].get("active_neuron_ratio", 0.0)
                    if "network_metrics" in processing_result else 0.0
                ),
                avg_processing_time=processing_time
            )
            self.meta_orchestrator.record_performance(performance)

        # Compile complete result
        result = {
            "input": input_text,
            "processing_id": self.processing_count,
            "timestamp": datetime.now(),
            "processing_time": processing_time,
            "neural_processing": processing_result,
            "global_workspace": workspace_result,
            "consciousness_level": consciousness_level,
            "memory_context": self.memory.get_context() if use_memory else None,
            "brain_metrics": self._get_brain_metrics()
        }

        # Add enhancement results if enabled
        if enable_enhancements:
            result["enhancements"] = {
                "task_complexity": {
                    "score": complexity_estimate.complexity_score if complexity_estimate else 0.0,
                    "factors": complexity_estimate.factors if complexity_estimate else {},
                },
                "adjusted_parameters": adjusted_params,
                "summarization_stats": self.summarization_neuron.get_stats(),
                "attention_stats": self.attention_manager.get_stats(),
                "concept_stats": self.concept_tracker.get_stats(),
                "meta_orchestration_stats": self.meta_orchestrator.get_stats()
            }

        return result

    def _get_current_performance(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics snapshot"""
        if self.processing_count == 0:
            return None

        return PerformanceMetrics(
            avg_consciousness_level=self.total_consciousness_level / max(1, self.processing_count),
            avg_coherence_score=0.8,  # Simplified
            avg_integration_score=0.7,  # Simplified
            avg_neurons_activated=0.5,  # Simplified
            avg_processing_time=2.0  # Simplified
        )

    def _process_global_workspace(
        self,
        processing_result: Dict[str, Any],
        enable_enhancements: bool = True
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

        # Add enhancement metrics
        metrics["enhancements"] = {
            "summarization": self.summarization_neuron.get_stats(),
            "attention_windowing": self.attention_manager.get_stats(),
            "concept_tracking": self.concept_tracker.get_stats(),
            "meta_orchestration": self.meta_orchestrator.get_stats()
        }

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
