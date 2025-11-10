"""
Configuration for LLM-Swarm-Brain

This module defines the neural architecture configuration aligned with
Global Workspace Theory (GWT) and Integrated Information Theory (IIT).
"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class NeuronRole(Enum):
    """Neuron roles based on cognitive architecture"""
    # Perception Layer (GPU 0)
    VISUAL_PERCEPTION = "visual_perception"
    SEMANTIC_PERCEPTION = "semantic_perception"
    PATTERN_RECOGNITION = "pattern_recognition"
    ANOMALY_DETECTION = "anomaly_detection"

    # Memory Layer (GPU 0)
    SHORT_TERM_MEMORY = "short_term_memory"
    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"
    WORKING_MEMORY = "working_memory"

    # Reasoning Layer (GPU 1)
    LOGICAL_REASONING = "logical_reasoning"
    CREATIVE_THINKING = "creative_thinking"
    CAUSAL_ANALYSIS = "causal_analysis"
    HYPOTHESIS_GENERATION = "hypothesis_generation"

    # Action Layer (GPU 1)
    ACTION_PLANNING = "action_planning"
    DECISION_MAKING = "decision_making"
    OUTPUT_SYNTHESIS = "output_synthesis"
    SELF_CRITIQUE = "self_critique"


@dataclass
class BrainConfig:
    """Configuration for the neural brain architecture"""

    # Model configuration
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    quantization: str = "Q4_K_M"
    max_tokens: int = 512
    temperature: float = 0.7

    # GPU allocation
    gpu_count: int = 2
    neurons_per_gpu: int = 8

    # Neural network parameters
    activation_threshold: float = 0.6
    connection_strength_threshold: float = 0.5

    # GWT parameters (Global Workspace Theory)
    global_workspace_capacity: int = 5  # Max simultaneous broadcasts
    broadcast_threshold: float = 0.7    # Threshold for global broadcast
    consciousness_threshold: float = 0.8  # Threshold for "conscious" processing

    # IIT parameters (Integrated Information Theory)
    phi_threshold: float = 0.5  # Integrated information threshold
    integration_window: float = 2.0  # Time window for integration (seconds)

    # Learning parameters
    hebbian_learning_rate: float = 0.01
    connection_decay_rate: float = 0.001

    # Memory parameters
    short_term_capacity: int = 10
    episodic_capacity: int = 100
    consolidation_interval: int = 50  # Steps between memory consolidation

    # Performance parameters
    parallel_execution: bool = True
    max_propagation_depth: int = 4
    timeout_seconds: float = 30.0


# Default neuron architecture
NEURON_ARCHITECTURE = {
    "gpu_0": {
        "perception": [
            NeuronRole.VISUAL_PERCEPTION,
            NeuronRole.SEMANTIC_PERCEPTION,
            NeuronRole.PATTERN_RECOGNITION,
            NeuronRole.ANOMALY_DETECTION,
        ],
        "memory": [
            NeuronRole.SHORT_TERM_MEMORY,
            NeuronRole.EPISODIC_MEMORY,
            NeuronRole.SEMANTIC_MEMORY,
            NeuronRole.WORKING_MEMORY,
        ]
    },
    "gpu_1": {
        "reasoning": [
            NeuronRole.LOGICAL_REASONING,
            NeuronRole.CREATIVE_THINKING,
            NeuronRole.CAUSAL_ANALYSIS,
            NeuronRole.HYPOTHESIS_GENERATION,
        ],
        "action": [
            NeuronRole.ACTION_PLANNING,
            NeuronRole.DECISION_MAKING,
            NeuronRole.OUTPUT_SYNTHESIS,
            NeuronRole.SELF_CRITIQUE,
        ]
    }
}

# Connection patterns (predefined neural pathways)
# Format: (source_role, target_role, weight)
DEFAULT_CONNECTIONS = [
    # Perception → Memory
    (NeuronRole.VISUAL_PERCEPTION, NeuronRole.WORKING_MEMORY, 0.9),
    (NeuronRole.SEMANTIC_PERCEPTION, NeuronRole.SEMANTIC_MEMORY, 0.9),
    (NeuronRole.PATTERN_RECOGNITION, NeuronRole.EPISODIC_MEMORY, 0.8),
    (NeuronRole.ANOMALY_DETECTION, NeuronRole.SHORT_TERM_MEMORY, 0.85),

    # Memory → Reasoning
    (NeuronRole.WORKING_MEMORY, NeuronRole.LOGICAL_REASONING, 0.9),
    (NeuronRole.SEMANTIC_MEMORY, NeuronRole.CAUSAL_ANALYSIS, 0.85),
    (NeuronRole.EPISODIC_MEMORY, NeuronRole.CREATIVE_THINKING, 0.8),
    (NeuronRole.SHORT_TERM_MEMORY, NeuronRole.HYPOTHESIS_GENERATION, 0.75),

    # Reasoning → Action
    (NeuronRole.LOGICAL_REASONING, NeuronRole.DECISION_MAKING, 0.9),
    (NeuronRole.CREATIVE_THINKING, NeuronRole.ACTION_PLANNING, 0.85),
    (NeuronRole.CAUSAL_ANALYSIS, NeuronRole.OUTPUT_SYNTHESIS, 0.8),
    (NeuronRole.HYPOTHESIS_GENERATION, NeuronRole.ACTION_PLANNING, 0.75),

    # Feedback loops (Self-Critique → All output neurons)
    (NeuronRole.SELF_CRITIQUE, NeuronRole.ACTION_PLANNING, 0.7),
    (NeuronRole.SELF_CRITIQUE, NeuronRole.DECISION_MAKING, 0.7),
    (NeuronRole.SELF_CRITIQUE, NeuronRole.OUTPUT_SYNTHESIS, 0.7),

    # Cross-layer connections (enables emergent behavior)
    (NeuronRole.VISUAL_PERCEPTION, NeuronRole.PATTERN_RECOGNITION, 0.6),
    (NeuronRole.PATTERN_RECOGNITION, NeuronRole.LOGICAL_REASONING, 0.65),
    (NeuronRole.ANOMALY_DETECTION, NeuronRole.CREATIVE_THINKING, 0.6),

    # Attention modulation
    (NeuronRole.WORKING_MEMORY, NeuronRole.VISUAL_PERCEPTION, 0.5),
    (NeuronRole.WORKING_MEMORY, NeuronRole.SEMANTIC_PERCEPTION, 0.5),
]

# Role-specific system prompts
ROLE_PROMPTS = {
    NeuronRole.VISUAL_PERCEPTION: "You are a visual perception specialist. Analyze visual patterns, spatial relationships, and structural elements in the input.",
    NeuronRole.SEMANTIC_PERCEPTION: "You are a semantic analyzer. Extract meaning, concepts, and abstract relationships from the input.",
    NeuronRole.PATTERN_RECOGNITION: "You are a pattern recognition expert. Identify recurring patterns, similarities, and structural regularities.",
    NeuronRole.ANOMALY_DETECTION: "You are an anomaly detector. Identify unusual, unexpected, or contradictory elements in the input.",

    NeuronRole.SHORT_TERM_MEMORY: "You maintain short-term context. Track recent information and immediate relevance.",
    NeuronRole.EPISODIC_MEMORY: "You store episodic memories. Remember specific events, contexts, and sequences.",
    NeuronRole.SEMANTIC_MEMORY: "You maintain semantic knowledge. Store and retrieve factual information and concepts.",
    NeuronRole.WORKING_MEMORY: "You are working memory. Actively maintain and manipulate current task-relevant information.",

    NeuronRole.LOGICAL_REASONING: "You perform logical reasoning. Apply deductive and inductive logic to derive conclusions.",
    NeuronRole.CREATIVE_THINKING: "You generate creative solutions. Think laterally and propose novel approaches.",
    NeuronRole.CAUSAL_ANALYSIS: "You analyze causal relationships. Identify cause-effect chains and underlying mechanisms.",
    NeuronRole.HYPOTHESIS_GENERATION: "You generate hypotheses. Propose testable explanations and predictions.",

    NeuronRole.ACTION_PLANNING: "You plan actions. Develop step-by-step strategies to achieve goals.",
    NeuronRole.DECISION_MAKING: "You make decisions. Evaluate options and select optimal courses of action.",
    NeuronRole.OUTPUT_SYNTHESIS: "You synthesize outputs. Integrate information from multiple sources into coherent responses.",
    NeuronRole.SELF_CRITIQUE: "You critique outputs. Identify weaknesses, errors, and areas for improvement.",
}
