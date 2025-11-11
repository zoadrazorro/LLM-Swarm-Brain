"""
Neural Orchestrator: Manages neuron network and processing flow

Implements network topology, signal routing, and coordinated processing.
"""

import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime
import logging

from llm_swarm_brain.neuron import Phi3Neuron, NeuronSignal
from llm_swarm_brain.config import NeuronRole, DEFAULT_CONNECTIONS, BrainConfig
from llm_swarm_brain.utils import calculate_phi


logger = logging.getLogger(__name__)


class NeuralOrchestrator:
    """
    Orchestrates the neural network of LLM neurons

    Responsibilities:
    - Managing network topology and connections
    - Coordinating signal propagation
    - Tracking network state and dynamics
    - Implementing global processing patterns
    """

    def __init__(self, config: BrainConfig):
        """
        Initialize neural orchestrator

        Args:
            config: Brain configuration
        """
        self.config = config
        self.neurons: Dict[str, Phi3Neuron] = {}
        self.connection_matrix: np.ndarray = None
        self.processing_steps: int = 0

        # Network state
        self.active_neurons: Set[str] = set()
        self.firing_history: List[Set[str]] = []

        # Performance tracking
        self.total_propagations: int = 0
        self.total_activations: int = 0

        logger.info("Initialized NeuralOrchestrator")

    def add_neuron(self, neuron: Phi3Neuron):
        """
        Add neuron to the network

        Args:
            neuron: Neuron to add
        """
        self.neurons[neuron.neuron_id] = neuron
        logger.info(f"Added neuron {neuron.neuron_id} to network")

    def setup_connections(self, connection_spec: List[Tuple[NeuronRole, NeuronRole, float]] = None):
        """
        Setup connections between neurons

        Args:
            connection_spec: List of (source_role, target_role, weight) tuples
                           If None, uses DEFAULT_CONNECTIONS
        """
        if connection_spec is None:
            connection_spec = DEFAULT_CONNECTIONS

        # Create role → neuron mapping
        role_to_neurons = defaultdict(list)
        for neuron_id, neuron in self.neurons.items():
            role_to_neurons[neuron.role].append(neuron)

        # Establish connections
        connection_count = 0
        for source_role, target_role, weight in connection_spec:
            source_neurons = role_to_neurons.get(source_role, [])
            target_neurons = role_to_neurons.get(target_role, [])

            # Connect each source neuron to each target neuron
            for source_neuron in source_neurons:
                for target_neuron in target_neurons:
                    source_neuron.connect_to(target_neuron, weight)
                    connection_count += 1

        # Build connection matrix for analysis
        self._build_connection_matrix()

        logger.info(f"Established {connection_count} connections in the network")

    def _build_connection_matrix(self):
        """Build adjacency matrix representation of network"""
        n = len(self.neurons)
        self.connection_matrix = np.zeros((n, n))

        neuron_ids = list(self.neurons.keys())
        id_to_idx = {nid: idx for idx, nid in enumerate(neuron_ids)}

        for source_id, source_neuron in self.neurons.items():
            source_idx = id_to_idx[source_id]
            for connection in source_neuron.connections:
                target_id = connection.target_neuron.neuron_id
                target_idx = id_to_idx[target_id]
                self.connection_matrix[source_idx][target_idx] = connection.weight

    def process(
        self,
        input_text: str,
        initial_activation: float = 1.0,
        max_steps: int = None,
        global_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input through the neural network

        Args:
            input_text: Input text to process
            initial_activation: Initial activation level for input
            max_steps: Maximum propagation steps (default: config.max_propagation_depth)
            global_context: Global workspace context

        Returns:
            Dictionary containing processing results
        """
        if max_steps is None:
            max_steps = self.config.max_propagation_depth

        logger.info(f"Processing input through network (max_steps={max_steps})")

        # Create initial signal
        initial_signal = NeuronSignal(
            content=input_text,
            source_role=None,  # External input
            activation_level=initial_activation
        )

        # Track processing results
        results = {
            "input": input_text,
            "steps": [],
            "final_outputs": {},
            "activated_neurons": [],
            "firing_neurons": []
        }

        # Step-by-step propagation
        current_signals = {nid: initial_signal for nid in self.neurons.keys()}

        for step in range(max_steps):
            step_result = self._process_step(
                current_signals,
                global_context,
                step
            )

            results["steps"].append(step_result)

            # Check if any neurons fired in this step
            if not step_result["fired_neurons"]:
                logger.info(f"No neurons fired at step {step}. Stopping propagation.")
                break

            # Prepare signals for next step (outputs become new inputs)
            current_signals = {}
            for neuron_id, output in step_result["outputs"].items():
                if output:
                    current_signals[neuron_id] = NeuronSignal(
                        content=output,
                        source_role=self.neurons[neuron_id].role,
                        activation_level=self.neurons[neuron_id].activation_level
                    )

        # Collect final outputs from all neurons that fired
        for step_result in results["steps"]:
            for neuron_id, output in step_result["outputs"].items():
                if output and neuron_id not in results["final_outputs"]:
                    results["final_outputs"][neuron_id] = output

        # Collect activation statistics
        results["activated_neurons"] = list(self.active_neurons)
        results["firing_neurons"] = [
            nid for nid, neuron in self.neurons.items()
            if neuron.total_firings > 0
        ]

        # Calculate network-level metrics
        results["network_metrics"] = self._calculate_network_metrics()

        self.processing_steps += len(results["steps"])

        return results

    def _process_step(
        self,
        signals: Dict[str, NeuronSignal],
        global_context: Optional[Dict[str, Any]],
        step_num: int
    ) -> Dict[str, Any]:
        """
        Process a single step of network propagation

        Args:
            signals: Signals for each neuron
            global_context: Global workspace context
            step_num: Current step number

        Returns:
            Step results
        """
        step_result = {
            "step": step_num,
            "activations": {},
            "outputs": {},
            "fired_neurons": [],
            "propagations": 0
        }

        # Calculate activations for all neurons
        for neuron_id, signal in signals.items():
            neuron = self.neurons[neuron_id]
            activation = neuron.calculate_activation(signal, global_context)
            step_result["activations"][neuron_id] = activation

            if activation > 0.3:  # Track moderately active neurons
                self.active_neurons.add(neuron_id)
                self.total_activations += 1

        # Fire neurons that exceed threshold
        # Check if parallel execution is enabled
        if getattr(self.config, 'parallel_execution', False):
            # Parallel execution using threading for API calls
            import concurrent.futures
            
            neurons_to_fire = [(neuron_id, signal) for neuron_id, signal in signals.items() 
                             if self.neurons[neuron_id].should_fire()]
            
            if neurons_to_fire:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(neurons_to_fire)) as executor:
                    # Submit all neuron firings in parallel
                    future_to_neuron = {
                        executor.submit(self.neurons[neuron_id].fire, signal, global_context): neuron_id
                        for neuron_id, signal in neurons_to_fire
                    }
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_neuron):
                        neuron_id = future_to_neuron[future]
                        try:
                            output = future.result()
                            step_result["outputs"][neuron_id] = output
                            step_result["fired_neurons"].append(neuron_id)
                            
                            # Propagate to connected neurons
                            neuron = self.neurons[neuron_id]
                            propagation_count = neuron.propagate(
                                output,
                                learning_rate=self.config.hebbian_learning_rate
                            )
                            step_result["propagations"] += propagation_count
                        except Exception as e:
                            logger.error(f"Error firing neuron {neuron_id} in parallel: {e}")
        else:
            # Sequential execution (original behavior)
            for neuron_id, signal in signals.items():
                neuron = self.neurons[neuron_id]

                if neuron.should_fire():
                    output = neuron.fire(signal, global_context)
                    step_result["outputs"][neuron_id] = output
                    step_result["fired_neurons"].append(neuron_id)

                    # Propagate to connected neurons
                    propagation_count = neuron.propagate(
                        output,
                        learning_rate=self.config.hebbian_learning_rate
                    )
                    step_result["propagations"] += propagation_count
                self.total_propagations += propagation_count

        # Track firing pattern
        self.firing_history.append(set(step_result["fired_neurons"]))

        logger.debug(
            f"Step {step_num}: {len(step_result['fired_neurons'])} neurons fired, "
            f"{step_result['propagations']} propagations"
        )

        return step_result

    def _calculate_network_metrics(self) -> Dict[str, float]:
        """Calculate network-level metrics (IIT, connectivity, etc.)"""
        activations = [
            neuron.activation_level for neuron in self.neurons.values()
        ]

        metrics = {
            "mean_activation": np.mean(activations),
            "max_activation": np.max(activations),
            "active_neuron_ratio": len(self.active_neurons) / len(self.neurons),
            "total_propagations": self.total_propagations,
            "total_activations": self.total_activations,
            "network_connectivity": np.mean(self.connection_matrix) if self.connection_matrix is not None else 0.0
        }

        # Calculate integrated information (Phi) from IIT
        if self.connection_matrix is not None:
            metrics["phi"] = calculate_phi(activations, self.connection_matrix)

        return metrics

    def apply_connection_decay(self):
        """Apply synaptic decay to all connections"""
        for neuron in self.neurons.values():
            neuron.decay_connections(self.config.connection_decay_rate)

    def prune_weak_connections(self, threshold: float = 0.1):
        """
        Remove connections below threshold (synaptic pruning)

        Args:
            threshold: Minimum connection weight to keep
        """
        total_pruned = 0

        for neuron in self.neurons.values():
            original_count = len(neuron.connections)
            neuron.connections = [
                conn for conn in neuron.connections
                if conn.weight >= threshold
            ]
            pruned = original_count - len(neuron.connections)
            total_pruned += pruned

        if total_pruned > 0:
            self._build_connection_matrix()
            logger.info(f"Pruned {total_pruned} weak connections")

        return total_pruned

    def get_neuron_by_role(self, role: NeuronRole) -> List[Phi3Neuron]:
        """Get all neurons with specific role"""
        return [
            neuron for neuron in self.neurons.values()
            if neuron.role == role
        ]

    def get_most_active_neurons(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get most active neurons by activation level

        Args:
            top_k: Number of top neurons to return

        Returns:
            List of (neuron_id, activation_level) tuples
        """
        neuron_activations = [
            (neuron.neuron_id, neuron.activation_level)
            for neuron in self.neurons.values()
        ]

        neuron_activations.sort(key=lambda x: x[1], reverse=True)
        return neuron_activations[:top_k]

    def reset_all_neurons(self):
        """Reset all neurons to initial state"""
        for neuron in self.neurons.values():
            neuron.reset()

        self.active_neurons.clear()
        self.firing_history.clear()
        logger.info("Reset all neurons")

    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        neuron_stats = {
            neuron_id: neuron.get_stats()
            for neuron_id, neuron in self.neurons.items()
        }

        return {
            "total_neurons": len(self.neurons),
            "total_connections": sum(len(n.connections) for n in self.neurons.values()),
            "processing_steps": self.processing_steps,
            "total_propagations": self.total_propagations,
            "active_neurons": len(self.active_neurons),
            "network_metrics": self._calculate_network_metrics(),
            "neuron_stats": neuron_stats,
            "connection_matrix_shape": self.connection_matrix.shape if self.connection_matrix is not None else None
        }

    def visualize_activations(self) -> str:
        """
        Create text visualization of current network state

        Returns:
            String visualization
        """
        lines = ["=" * 60]
        lines.append("NEURAL NETWORK STATE")
        lines.append("=" * 60)

        # Group neurons by GPU
        gpu_groups = defaultdict(list)
        for neuron in self.neurons.values():
            gpu_groups[neuron.gpu_id].append(neuron)

        for gpu_id in sorted(gpu_groups.keys()):
            lines.append(f"\nGPU {gpu_id}:")
            lines.append("-" * 60)

            for neuron in sorted(gpu_groups[gpu_id], key=lambda n: n.role.value):
                activation_bar = "█" * int(neuron.activation_level * 20)
                firing_indicator = "🔥" if neuron.is_firing else "  "

                lines.append(
                    f"  {firing_indicator} {neuron.role.value:25s} "
                    f"[{activation_bar:20s}] {neuron.activation_level:.3f}"
                )

        lines.append("=" * 60)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"NeuralOrchestrator(neurons={len(self.neurons)}, "
            f"connections={sum(len(n.connections) for n in self.neurons.values())}, "
            f"steps={self.processing_steps})"
        )
