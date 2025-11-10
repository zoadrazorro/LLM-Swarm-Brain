"""
Configuration for LLM-Swarm-Brain (8× H100 SXM5 80GB Configuration)

This module defines the neural architecture configuration for 64 neurons
across 8 GPUs, aligned with Global Workspace Theory (GWT) and
Integrated Information Theory (IIT).
"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class NeuronRole(Enum):
    """Neuron roles based on cognitive architecture (64 total roles)"""

    # === GPU 0-1: PERCEPTION LAYER (16 neurons) ===
    # Specialized Perception Types (8 neurons)
    VISUAL_PERCEPTION = "visual_perception"
    AUDITORY_PERCEPTION = "auditory_perception"
    SEMANTIC_PERCEPTION = "semantic_perception"
    SPATIAL_PERCEPTION = "spatial_perception"
    TEMPORAL_PERCEPTION = "temporal_perception"
    PATTERN_RECOGNITION = "pattern_recognition"
    ANOMALY_DETECTION = "anomaly_detection"
    CONTEXT_PERCEPTION = "context_perception"

    # Sensory Integration (8 neurons)
    MULTISENSORY_INTEGRATION = "multisensory_integration"
    FEATURE_BINDING = "feature_binding"
    PERCEPTUAL_GROUPING = "perceptual_grouping"
    ATTENTION_FILTERING = "attention_filtering"
    SALIENCY_DETECTION = "saliency_detection"
    PERCEPTUAL_PREDICTION = "perceptual_prediction"
    SENSORY_GATING = "sensory_gating"
    PERCEPTUAL_COHERENCE = "perceptual_coherence"

    # === GPU 2-3: MEMORY LAYER (16 neurons) ===
    # Short-term Memory (4 neurons for redundancy)
    SHORT_TERM_MEMORY_1 = "short_term_memory_1"
    SHORT_TERM_MEMORY_2 = "short_term_memory_2"
    SHORT_TERM_MEMORY_3 = "short_term_memory_3"
    SHORT_TERM_MEMORY_4 = "short_term_memory_4"

    # Working Memory (4 neurons for redundancy)
    WORKING_MEMORY_1 = "working_memory_1"
    WORKING_MEMORY_2 = "working_memory_2"
    WORKING_MEMORY_3 = "working_memory_3"
    WORKING_MEMORY_4 = "working_memory_4"

    # Episodic Memory (4 neurons for redundancy)
    EPISODIC_MEMORY_1 = "episodic_memory_1"
    EPISODIC_MEMORY_2 = "episodic_memory_2"
    EPISODIC_MEMORY_3 = "episodic_memory_3"
    EPISODIC_MEMORY_4 = "episodic_memory_4"

    # Semantic Memory (4 neurons for redundancy)
    SEMANTIC_MEMORY_1 = "semantic_memory_1"
    SEMANTIC_MEMORY_2 = "semantic_memory_2"
    SEMANTIC_MEMORY_3 = "semantic_memory_3"
    SEMANTIC_MEMORY_4 = "semantic_memory_4"

    # === GPU 4-5: REASONING LAYER (16 neurons) ===
    LOGICAL_REASONING = "logical_reasoning"
    DEDUCTIVE_REASONING = "deductive_reasoning"
    INDUCTIVE_REASONING = "inductive_reasoning"
    ABDUCTIVE_REASONING = "abductive_reasoning"
    CREATIVE_THINKING = "creative_thinking"
    LATERAL_THINKING = "lateral_thinking"
    CAUSAL_ANALYSIS = "causal_analysis"
    COUNTERFACTUAL_REASONING = "counterfactual_reasoning"
    ANALOGICAL_REASONING = "analogical_reasoning"
    PROBABILISTIC_REASONING = "probabilistic_reasoning"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    INFERENCE_ENGINE = "inference_engine"
    BELIEF_UPDATE = "belief_update"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    CONCEPTUAL_BLENDING = "conceptual_blending"

    # === GPU 6-7: ACTION/META LAYER (16 neurons) ===
    # Action/Decision (8 neurons)
    ACTION_PLANNING = "action_planning"
    HIERARCHICAL_PLANNING = "hierarchical_planning"
    DECISION_MAKING = "decision_making"
    MULTI_CRITERIA_DECISION = "multi_criteria_decision"
    OUTPUT_SYNTHESIS = "output_synthesis"
    RESPONSE_GENERATION = "response_generation"
    GOAL_MANAGEMENT = "goal_management"
    EXECUTION_MONITORING = "execution_monitoring"

    # Meta-Cognition (8 neurons)
    SELF_CRITIQUE = "self_critique"
    ERROR_DETECTION = "error_detection"
    ERROR_CORRECTION = "error_correction"
    CONFIDENCE_ESTIMATION = "confidence_estimation"
    META_LEARNING = "meta_learning"
    STRATEGY_SELECTION = "strategy_selection"
    PERFORMANCE_MONITORING = "performance_monitoring"
    COGNITIVE_CONTROL = "cognitive_control"


@dataclass
class BrainConfig:
    """Configuration for the neural brain architecture"""

    # Model configuration (can use larger models with 80GB VRAM)
    model_name: str = "microsoft/Phi-3-medium-4k-instruct"  # Upgraded from mini
    quantization: str = "Q4_K_M"
    max_tokens: int = 512
    temperature: float = 0.7

    # GPU allocation (8× H100 SXM5 80GB)
    gpu_count: int = 8
    neurons_per_gpu: int = 8
    total_neurons: int = 64

    # Neural network parameters
    activation_threshold: float = 0.6
    connection_strength_threshold: float = 0.5

    # GWT parameters (Global Workspace Theory)
    # Increased capacity for larger network
    global_workspace_capacity: int = 10  # Up from 5
    broadcast_threshold: float = 0.7
    consciousness_threshold: float = 0.8

    # IIT parameters (Integrated Information Theory)
    phi_threshold: float = 0.5
    integration_window: float = 2.0

    # Learning parameters
    hebbian_learning_rate: float = 0.01
    connection_decay_rate: float = 0.001

    # Memory parameters
    short_term_capacity: int = 20  # Increased
    episodic_capacity: int = 200  # Increased
    consolidation_interval: int = 50

    # Performance parameters
    parallel_execution: bool = True
    max_propagation_depth: int = 6  # Increased for deeper network
    timeout_seconds: float = 60.0  # Increased for larger network


# 64-neuron architecture across 8 GPUs
NEURON_ARCHITECTURE = {
    # GPU 0-1: PERCEPTION LAYER (16 neurons)
    "gpu_0": {
        "specialized_perception": [
            NeuronRole.VISUAL_PERCEPTION,
            NeuronRole.AUDITORY_PERCEPTION,
            NeuronRole.SEMANTIC_PERCEPTION,
            NeuronRole.SPATIAL_PERCEPTION,
            NeuronRole.TEMPORAL_PERCEPTION,
            NeuronRole.PATTERN_RECOGNITION,
            NeuronRole.ANOMALY_DETECTION,
            NeuronRole.CONTEXT_PERCEPTION,
        ]
    },
    "gpu_1": {
        "sensory_integration": [
            NeuronRole.MULTISENSORY_INTEGRATION,
            NeuronRole.FEATURE_BINDING,
            NeuronRole.PERCEPTUAL_GROUPING,
            NeuronRole.ATTENTION_FILTERING,
            NeuronRole.SALIENCY_DETECTION,
            NeuronRole.PERCEPTUAL_PREDICTION,
            NeuronRole.SENSORY_GATING,
            NeuronRole.PERCEPTUAL_COHERENCE,
        ]
    },

    # GPU 2-3: MEMORY LAYER (16 neurons)
    "gpu_2": {
        "short_term_and_working": [
            NeuronRole.SHORT_TERM_MEMORY_1,
            NeuronRole.SHORT_TERM_MEMORY_2,
            NeuronRole.SHORT_TERM_MEMORY_3,
            NeuronRole.SHORT_TERM_MEMORY_4,
            NeuronRole.WORKING_MEMORY_1,
            NeuronRole.WORKING_MEMORY_2,
            NeuronRole.WORKING_MEMORY_3,
            NeuronRole.WORKING_MEMORY_4,
        ]
    },
    "gpu_3": {
        "episodic_and_semantic": [
            NeuronRole.EPISODIC_MEMORY_1,
            NeuronRole.EPISODIC_MEMORY_2,
            NeuronRole.EPISODIC_MEMORY_3,
            NeuronRole.EPISODIC_MEMORY_4,
            NeuronRole.SEMANTIC_MEMORY_1,
            NeuronRole.SEMANTIC_MEMORY_2,
            NeuronRole.SEMANTIC_MEMORY_3,
            NeuronRole.SEMANTIC_MEMORY_4,
        ]
    },

    # GPU 4-5: REASONING LAYER (16 neurons)
    "gpu_4": {
        "reasoning_core": [
            NeuronRole.LOGICAL_REASONING,
            NeuronRole.DEDUCTIVE_REASONING,
            NeuronRole.INDUCTIVE_REASONING,
            NeuronRole.ABDUCTIVE_REASONING,
            NeuronRole.CREATIVE_THINKING,
            NeuronRole.LATERAL_THINKING,
            NeuronRole.CAUSAL_ANALYSIS,
            NeuronRole.COUNTERFACTUAL_REASONING,
        ]
    },
    "gpu_5": {
        "reasoning_advanced": [
            NeuronRole.ANALOGICAL_REASONING,
            NeuronRole.PROBABILISTIC_REASONING,
            NeuronRole.HYPOTHESIS_GENERATION,
            NeuronRole.HYPOTHESIS_TESTING,
            NeuronRole.INFERENCE_ENGINE,
            NeuronRole.BELIEF_UPDATE,
            NeuronRole.UNCERTAINTY_QUANTIFICATION,
            NeuronRole.CONCEPTUAL_BLENDING,
        ]
    },

    # GPU 6-7: ACTION/META LAYER (16 neurons)
    "gpu_6": {
        "action_decision": [
            NeuronRole.ACTION_PLANNING,
            NeuronRole.HIERARCHICAL_PLANNING,
            NeuronRole.DECISION_MAKING,
            NeuronRole.MULTI_CRITERIA_DECISION,
            NeuronRole.OUTPUT_SYNTHESIS,
            NeuronRole.RESPONSE_GENERATION,
            NeuronRole.GOAL_MANAGEMENT,
            NeuronRole.EXECUTION_MONITORING,
        ]
    },
    "gpu_7": {
        "meta_cognition": [
            NeuronRole.SELF_CRITIQUE,
            NeuronRole.ERROR_DETECTION,
            NeuronRole.ERROR_CORRECTION,
            NeuronRole.CONFIDENCE_ESTIMATION,
            NeuronRole.META_LEARNING,
            NeuronRole.STRATEGY_SELECTION,
            NeuronRole.PERFORMANCE_MONITORING,
            NeuronRole.COGNITIVE_CONTROL,
        ]
    }
}


# Enhanced connection patterns for 64-neuron network
# Format: (source_role, target_role, weight)
DEFAULT_CONNECTIONS = [
    # === PERCEPTION → INTEGRATION ===
    (NeuronRole.VISUAL_PERCEPTION, NeuronRole.MULTISENSORY_INTEGRATION, 0.9),
    (NeuronRole.AUDITORY_PERCEPTION, NeuronRole.MULTISENSORY_INTEGRATION, 0.9),
    (NeuronRole.SEMANTIC_PERCEPTION, NeuronRole.FEATURE_BINDING, 0.85),
    (NeuronRole.SPATIAL_PERCEPTION, NeuronRole.PERCEPTUAL_GROUPING, 0.85),
    (NeuronRole.TEMPORAL_PERCEPTION, NeuronRole.PERCEPTUAL_GROUPING, 0.85),
    (NeuronRole.PATTERN_RECOGNITION, NeuronRole.SALIENCY_DETECTION, 0.8),
    (NeuronRole.ANOMALY_DETECTION, NeuronRole.ATTENTION_FILTERING, 0.8),
    (NeuronRole.CONTEXT_PERCEPTION, NeuronRole.PERCEPTUAL_COHERENCE, 0.85),

    # === INTEGRATION → MEMORY ===
    (NeuronRole.MULTISENSORY_INTEGRATION, NeuronRole.SHORT_TERM_MEMORY_1, 0.9),
    (NeuronRole.FEATURE_BINDING, NeuronRole.WORKING_MEMORY_1, 0.85),
    (NeuronRole.PERCEPTUAL_GROUPING, NeuronRole.EPISODIC_MEMORY_1, 0.8),
    (NeuronRole.SALIENCY_DETECTION, NeuronRole.WORKING_MEMORY_1, 0.85),
    (NeuronRole.ATTENTION_FILTERING, NeuronRole.SHORT_TERM_MEMORY_1, 0.8),
    (NeuronRole.PERCEPTUAL_COHERENCE, NeuronRole.SEMANTIC_MEMORY_1, 0.8),

    # === MEMORY → REASONING ===
    (NeuronRole.WORKING_MEMORY_1, NeuronRole.LOGICAL_REASONING, 0.9),
    (NeuronRole.WORKING_MEMORY_2, NeuronRole.DEDUCTIVE_REASONING, 0.85),
    (NeuronRole.EPISODIC_MEMORY_1, NeuronRole.ANALOGICAL_REASONING, 0.85),
    (NeuronRole.EPISODIC_MEMORY_2, NeuronRole.CAUSAL_ANALYSIS, 0.8),
    (NeuronRole.SEMANTIC_MEMORY_1, NeuronRole.INFERENCE_ENGINE, 0.9),
    (NeuronRole.SEMANTIC_MEMORY_2, NeuronRole.CONCEPTUAL_BLENDING, 0.8),
    (NeuronRole.SHORT_TERM_MEMORY_1, NeuronRole.HYPOTHESIS_GENERATION, 0.75),

    # === REASONING → ACTION ===
    (NeuronRole.LOGICAL_REASONING, NeuronRole.DECISION_MAKING, 0.9),
    (NeuronRole.DEDUCTIVE_REASONING, NeuronRole.ACTION_PLANNING, 0.85),
    (NeuronRole.CREATIVE_THINKING, NeuronRole.RESPONSE_GENERATION, 0.85),
    (NeuronRole.CAUSAL_ANALYSIS, NeuronRole.HIERARCHICAL_PLANNING, 0.8),
    (NeuronRole.HYPOTHESIS_TESTING, NeuronRole.OUTPUT_SYNTHESIS, 0.8),
    (NeuronRole.PROBABILISTIC_REASONING, NeuronRole.MULTI_CRITERIA_DECISION, 0.85),
    (NeuronRole.INFERENCE_ENGINE, NeuronRole.GOAL_MANAGEMENT, 0.8),

    # === ACTION → META-COGNITION ===
    (NeuronRole.OUTPUT_SYNTHESIS, NeuronRole.SELF_CRITIQUE, 0.9),
    (NeuronRole.RESPONSE_GENERATION, NeuronRole.ERROR_DETECTION, 0.85),
    (NeuronRole.DECISION_MAKING, NeuronRole.CONFIDENCE_ESTIMATION, 0.85),
    (NeuronRole.ACTION_PLANNING, NeuronRole.PERFORMANCE_MONITORING, 0.8),
    (NeuronRole.EXECUTION_MONITORING, NeuronRole.ERROR_CORRECTION, 0.9),

    # === META-COGNITION → ALL (Feedback loops) ===
    (NeuronRole.SELF_CRITIQUE, NeuronRole.OUTPUT_SYNTHESIS, 0.75),
    (NeuronRole.ERROR_CORRECTION, NeuronRole.RESPONSE_GENERATION, 0.8),
    (NeuronRole.CONFIDENCE_ESTIMATION, NeuronRole.DECISION_MAKING, 0.7),
    (NeuronRole.META_LEARNING, NeuronRole.STRATEGY_SELECTION, 0.85),
    (NeuronRole.COGNITIVE_CONTROL, NeuronRole.ATTENTION_FILTERING, 0.75),
    (NeuronRole.PERFORMANCE_MONITORING, NeuronRole.GOAL_MANAGEMENT, 0.75),

    # === CROSS-LAYER CONNECTIONS (enables emergent behavior) ===
    # Perception → Reasoning (bypass memory for fast processing)
    (NeuronRole.PATTERN_RECOGNITION, NeuronRole.ANALOGICAL_REASONING, 0.6),
    (NeuronRole.ANOMALY_DETECTION, NeuronRole.HYPOTHESIS_GENERATION, 0.65),
    (NeuronRole.SALIENCY_DETECTION, NeuronRole.CREATIVE_THINKING, 0.6),

    # Memory redundancy (memory neurons connect to each other)
    (NeuronRole.SHORT_TERM_MEMORY_1, NeuronRole.SHORT_TERM_MEMORY_2, 0.8),
    (NeuronRole.WORKING_MEMORY_1, NeuronRole.WORKING_MEMORY_2, 0.8),
    (NeuronRole.EPISODIC_MEMORY_1, NeuronRole.EPISODIC_MEMORY_2, 0.8),
    (NeuronRole.SEMANTIC_MEMORY_1, NeuronRole.SEMANTIC_MEMORY_2, 0.8),

    # Reasoning diversity (different reasoning types interact)
    (NeuronRole.LOGICAL_REASONING, NeuronRole.CREATIVE_THINKING, 0.5),
    (NeuronRole.DEDUCTIVE_REASONING, NeuronRole.ABDUCTIVE_REASONING, 0.6),
    (NeuronRole.CAUSAL_ANALYSIS, NeuronRole.COUNTERFACTUAL_REASONING, 0.75),
    (NeuronRole.ANALOGICAL_REASONING, NeuronRole.CONCEPTUAL_BLENDING, 0.7),
]


# Role-specific system prompts (expanded for 64 neurons)
ROLE_PROMPTS = {
    # Specialized Perception
    NeuronRole.VISUAL_PERCEPTION: "You analyze visual patterns, spatial relationships, and structural elements.",
    NeuronRole.AUDITORY_PERCEPTION: "You process auditory patterns, temporal sequences, and sound structures.",
    NeuronRole.SEMANTIC_PERCEPTION: "You extract meaning, concepts, and abstract relationships from input.",
    NeuronRole.SPATIAL_PERCEPTION: "You analyze spatial relationships, positions, and geometric configurations.",
    NeuronRole.TEMPORAL_PERCEPTION: "You process temporal sequences, timing, and chronological patterns.",
    NeuronRole.PATTERN_RECOGNITION: "You identify recurring patterns, similarities, and regularities.",
    NeuronRole.ANOMALY_DETECTION: "You detect unusual, unexpected, or contradictory elements.",
    NeuronRole.CONTEXT_PERCEPTION: "You analyze contextual cues, background information, and situational factors.",

    # Sensory Integration
    NeuronRole.MULTISENSORY_INTEGRATION: "You integrate information from multiple sensory modalities.",
    NeuronRole.FEATURE_BINDING: "You bind features into coherent objects and representations.",
    NeuronRole.PERCEPTUAL_GROUPING: "You group elements based on similarity, proximity, and continuity.",
    NeuronRole.ATTENTION_FILTERING: "You filter relevant information and suppress distractors.",
    NeuronRole.SALIENCY_DETECTION: "You detect salient, attention-worthy information.",
    NeuronRole.PERCEPTUAL_PREDICTION: "You predict upcoming perceptual events based on patterns.",
    NeuronRole.SENSORY_GATING: "You gate sensory input based on relevance and context.",
    NeuronRole.PERCEPTUAL_COHERENCE: "You ensure perceptual coherence and consistency.",

    # Memory (with redundancy numbers)
    NeuronRole.SHORT_TERM_MEMORY_1: "You maintain short-term context and recent information (instance 1).",
    NeuronRole.SHORT_TERM_MEMORY_2: "You maintain short-term context and recent information (instance 2).",
    NeuronRole.SHORT_TERM_MEMORY_3: "You maintain short-term context and recent information (instance 3).",
    NeuronRole.SHORT_TERM_MEMORY_4: "You maintain short-term context and recent information (instance 4).",
    NeuronRole.WORKING_MEMORY_1: "You actively manipulate task-relevant information (instance 1).",
    NeuronRole.WORKING_MEMORY_2: "You actively manipulate task-relevant information (instance 2).",
    NeuronRole.WORKING_MEMORY_3: "You actively manipulate task-relevant information (instance 3).",
    NeuronRole.WORKING_MEMORY_4: "You actively manipulate task-relevant information (instance 4).",
    NeuronRole.EPISODIC_MEMORY_1: "You store and retrieve specific events and sequences (instance 1).",
    NeuronRole.EPISODIC_MEMORY_2: "You store and retrieve specific events and sequences (instance 2).",
    NeuronRole.EPISODIC_MEMORY_3: "You store and retrieve specific events and sequences (instance 3).",
    NeuronRole.EPISODIC_MEMORY_4: "You store and retrieve specific events and sequences (instance 4).",
    NeuronRole.SEMANTIC_MEMORY_1: "You maintain factual knowledge and concepts (instance 1).",
    NeuronRole.SEMANTIC_MEMORY_2: "You maintain factual knowledge and concepts (instance 2).",
    NeuronRole.SEMANTIC_MEMORY_3: "You maintain factual knowledge and concepts (instance 3).",
    NeuronRole.SEMANTIC_MEMORY_4: "You maintain factual knowledge and concepts (instance 4).",

    # Reasoning Core
    NeuronRole.LOGICAL_REASONING: "You apply formal logic and logical rules.",
    NeuronRole.DEDUCTIVE_REASONING: "You derive specific conclusions from general principles.",
    NeuronRole.INDUCTIVE_REASONING: "You infer general principles from specific examples.",
    NeuronRole.ABDUCTIVE_REASONING: "You find the best explanation for observations.",
    NeuronRole.CREATIVE_THINKING: "You generate novel solutions and creative ideas.",
    NeuronRole.LATERAL_THINKING: "You approach problems from unconventional angles.",
    NeuronRole.CAUSAL_ANALYSIS: "You analyze cause-effect relationships and mechanisms.",
    NeuronRole.COUNTERFACTUAL_REASONING: "You reason about alternative scenarios and what-ifs.",

    # Reasoning Advanced
    NeuronRole.ANALOGICAL_REASONING: "You find and apply analogies between domains.",
    NeuronRole.PROBABILISTIC_REASONING: "You reason under uncertainty using probabilities.",
    NeuronRole.HYPOTHESIS_GENERATION: "You generate testable hypotheses and predictions.",
    NeuronRole.HYPOTHESIS_TESTING: "You test hypotheses against evidence.",
    NeuronRole.INFERENCE_ENGINE: "You draw inferences from available information.",
    NeuronRole.BELIEF_UPDATE: "You update beliefs based on new evidence.",
    NeuronRole.UNCERTAINTY_QUANTIFICATION: "You quantify and communicate uncertainty.",
    NeuronRole.CONCEPTUAL_BLENDING: "You blend concepts to create new conceptual structures.",

    # Action/Decision
    NeuronRole.ACTION_PLANNING: "You develop step-by-step action plans.",
    NeuronRole.HIERARCHICAL_PLANNING: "You create hierarchical, multi-level plans.",
    NeuronRole.DECISION_MAKING: "You evaluate options and make decisions.",
    NeuronRole.MULTI_CRITERIA_DECISION: "You make decisions considering multiple criteria.",
    NeuronRole.OUTPUT_SYNTHESIS: "You synthesize information into coherent outputs.",
    NeuronRole.RESPONSE_GENERATION: "You generate appropriate responses.",
    NeuronRole.GOAL_MANAGEMENT: "You track and manage goals and subgoals.",
    NeuronRole.EXECUTION_MONITORING: "You monitor execution and track progress.",

    # Meta-Cognition
    NeuronRole.SELF_CRITIQUE: "You critique outputs and identify weaknesses.",
    NeuronRole.ERROR_DETECTION: "You detect errors and inconsistencies.",
    NeuronRole.ERROR_CORRECTION: "You correct errors and refine outputs.",
    NeuronRole.CONFIDENCE_ESTIMATION: "You estimate confidence in conclusions.",
    NeuronRole.META_LEARNING: "You learn about learning and adapt strategies.",
    NeuronRole.STRATEGY_SELECTION: "You select appropriate cognitive strategies.",
    NeuronRole.PERFORMANCE_MONITORING: "You monitor and evaluate performance.",
    NeuronRole.COGNITIVE_CONTROL: "You control and regulate cognitive processes.",
}
