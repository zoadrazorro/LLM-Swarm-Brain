"""
Configuration for LLM-Swarm-Brain (64-Neuron API Architecture)

This module defines a dense neural architecture with 64 specialized neurons
for API-based operation using Llama 3.1 405B via Hyperbolic API.

Architecture: Dense Mixture of Experts (64 neurons)
- 8 virtual GPUs × 8 neurons each
- Highly specialized cognitive functions
- Rich interconnection patterns
"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class NeuronRole(Enum):
    """64 specialized neuron roles organized by cognitive function"""
    
    # === PERCEPTION LAYER (8 neurons) ===
    VISUAL_PERCEPTION = "visual_perception"
    AUDITORY_PERCEPTION = "auditory_perception"
    SEMANTIC_PERCEPTION = "semantic_perception"
    PATTERN_PERCEPTION = "pattern_perception"
    CONTEXT_PERCEPTION = "context_perception"
    TEMPORAL_PERCEPTION = "temporal_perception"
    SPATIAL_PERCEPTION = "spatial_perception"
    ABSTRACT_PERCEPTION = "abstract_perception"
    
    # === ATTENTION LAYER (8 neurons) ===
    SELECTIVE_ATTENTION = "selective_attention"
    SUSTAINED_ATTENTION = "sustained_attention"
    DIVIDED_ATTENTION = "divided_attention"
    SALIENCE_DETECTION = "salience_detection"
    RELEVANCE_FILTERING = "relevance_filtering"
    PRIORITY_RANKING = "priority_ranking"
    FOCUS_CONTROL = "focus_control"
    ATTENTION_SWITCHING = "attention_switching"
    
    # === MEMORY LAYER (8 neurons) ===
    SHORT_TERM_MEMORY = "short_term_memory"
    WORKING_MEMORY = "working_memory"
    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"
    PROCEDURAL_MEMORY = "procedural_memory"
    ASSOCIATIVE_MEMORY = "associative_memory"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    MEMORY_RETRIEVAL = "memory_retrieval"
    
    # === REASONING LAYER (8 neurons) ===
    DEDUCTIVE_REASONING = "deductive_reasoning"
    INDUCTIVE_REASONING = "inductive_reasoning"
    ABDUCTIVE_REASONING = "abductive_reasoning"
    ANALOGICAL_REASONING = "analogical_reasoning"
    CAUSAL_REASONING = "causal_reasoning"
    PROBABILISTIC_REASONING = "probabilistic_reasoning"
    LOGICAL_INFERENCE = "logical_inference"
    COUNTERFACTUAL_REASONING = "counterfactual_reasoning"
    
    # === CREATIVE LAYER (8 neurons) ===
    DIVERGENT_THINKING = "divergent_thinking"
    CONCEPTUAL_BLENDING = "conceptual_blending"
    METAPHOR_GENERATION = "metaphor_generation"
    NOVEL_COMBINATION = "novel_combination"
    LATERAL_THINKING = "lateral_thinking"
    IMAGINATIVE_PROJECTION = "imaginative_projection"
    INNOVATION_SYNTHESIS = "innovation_synthesis"
    CREATIVE_CONSTRAINT = "creative_constraint"
    
    # === ANALYTICAL LAYER (8 neurons) ===
    QUANTITATIVE_ANALYSIS = "quantitative_analysis"
    QUALITATIVE_ANALYSIS = "qualitative_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    CRITICAL_EVALUATION = "critical_evaluation"
    SYSTEMATIC_DECOMPOSITION = "systematic_decomposition"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    EVIDENCE_WEIGHING = "evidence_weighing"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    
    # === SYNTHESIS LAYER (8 neurons) ===
    INFORMATION_INTEGRATION = "information_integration"
    COHERENCE_BUILDING = "coherence_building"
    CONFLICT_RESOLUTION = "conflict_resolution"
    PERSPECTIVE_MERGING = "perspective_merging"
    HOLISTIC_SYNTHESIS = "holistic_synthesis"
    OUTPUT_GENERATION = "output_generation"
    NARRATIVE_CONSTRUCTION = "narrative_construction"
    SOLUTION_FORMULATION = "solution_formulation"
    
    # === META-COGNITIVE LAYER (8 neurons) ===
    SELF_MONITORING = "self_monitoring"
    ERROR_DETECTION = "error_detection"
    CONFIDENCE_ESTIMATION = "confidence_estimation"
    STRATEGY_SELECTION = "strategy_selection"
    PERFORMANCE_EVALUATION = "performance_evaluation"
    COGNITIVE_CONTROL = "cognitive_control"
    METACOGNITIVE_AWARENESS = "metacognitive_awareness"
    ADAPTIVE_REGULATION = "adaptive_regulation"


@dataclass
class BrainConfig:
    """Configuration for 64-neuron architecture"""
    
    # Model configuration - API mode
    model_name: str = "meta-llama/Meta-Llama-3.1-405B-Instruct"
    quantization: str = "none"  # Not used in API mode
    max_tokens: int = 2048
    temperature: float = 0.7
    
    # Architecture - 64 neurons
    gpu_count: int = 8  # Virtual GPUs (organizational units)
    neurons_per_gpu: int = 8
    total_neurons: int = 64
    
    # Neural network parameters
    activation_threshold: float = 0.45  # Lower for more activation
    connection_strength_threshold: float = 0.35
    
    # GWT parameters
    global_workspace_capacity: int = 12  # More capacity for larger network
    broadcast_threshold: float = 0.60
    consciousness_threshold: float = 0.70
    
    # IIT parameters
    phi_threshold: float = 0.55
    integration_window: float = 3.0
    
    # Learning parameters
    hebbian_learning_rate: float = 0.01
    connection_decay_rate: float = 0.0003
    
    # Memory parameters
    short_term_capacity: int = 100
    episodic_capacity: int = 1000
    consolidation_interval: int = 50
    
    # Performance parameters
    parallel_execution: bool = True
    max_propagation_depth: int = 5  # More depth for complex processing
    timeout_seconds: float = 180.0


# 64-neuron architecture - 8 neurons per virtual GPU
NEURON_ARCHITECTURE = {
    # GPU 0: PERCEPTION LAYER
    "gpu_0": {
        "perception": [
            NeuronRole.VISUAL_PERCEPTION,
            NeuronRole.AUDITORY_PERCEPTION,
            NeuronRole.SEMANTIC_PERCEPTION,
            NeuronRole.PATTERN_PERCEPTION,
            NeuronRole.CONTEXT_PERCEPTION,
            NeuronRole.TEMPORAL_PERCEPTION,
            NeuronRole.SPATIAL_PERCEPTION,
            NeuronRole.ABSTRACT_PERCEPTION,
        ]
    },
    
    # GPU 1: ATTENTION LAYER
    "gpu_1": {
        "attention": [
            NeuronRole.SELECTIVE_ATTENTION,
            NeuronRole.SUSTAINED_ATTENTION,
            NeuronRole.DIVIDED_ATTENTION,
            NeuronRole.SALIENCE_DETECTION,
            NeuronRole.RELEVANCE_FILTERING,
            NeuronRole.PRIORITY_RANKING,
            NeuronRole.FOCUS_CONTROL,
            NeuronRole.ATTENTION_SWITCHING,
        ]
    },
    
    # GPU 2: MEMORY LAYER
    "gpu_2": {
        "memory": [
            NeuronRole.SHORT_TERM_MEMORY,
            NeuronRole.WORKING_MEMORY,
            NeuronRole.EPISODIC_MEMORY,
            NeuronRole.SEMANTIC_MEMORY,
            NeuronRole.PROCEDURAL_MEMORY,
            NeuronRole.ASSOCIATIVE_MEMORY,
            NeuronRole.MEMORY_CONSOLIDATION,
            NeuronRole.MEMORY_RETRIEVAL,
        ]
    },
    
    # GPU 3: REASONING LAYER
    "gpu_3": {
        "reasoning": [
            NeuronRole.DEDUCTIVE_REASONING,
            NeuronRole.INDUCTIVE_REASONING,
            NeuronRole.ABDUCTIVE_REASONING,
            NeuronRole.ANALOGICAL_REASONING,
            NeuronRole.CAUSAL_REASONING,
            NeuronRole.PROBABILISTIC_REASONING,
            NeuronRole.LOGICAL_INFERENCE,
            NeuronRole.COUNTERFACTUAL_REASONING,
        ]
    },
    
    # GPU 4: CREATIVE LAYER
    "gpu_4": {
        "creative": [
            NeuronRole.DIVERGENT_THINKING,
            NeuronRole.CONCEPTUAL_BLENDING,
            NeuronRole.METAPHOR_GENERATION,
            NeuronRole.NOVEL_COMBINATION,
            NeuronRole.LATERAL_THINKING,
            NeuronRole.IMAGINATIVE_PROJECTION,
            NeuronRole.INNOVATION_SYNTHESIS,
            NeuronRole.CREATIVE_CONSTRAINT,
        ]
    },
    
    # GPU 5: ANALYTICAL LAYER
    "gpu_5": {
        "analytical": [
            NeuronRole.QUANTITATIVE_ANALYSIS,
            NeuronRole.QUALITATIVE_ANALYSIS,
            NeuronRole.COMPARATIVE_ANALYSIS,
            NeuronRole.CRITICAL_EVALUATION,
            NeuronRole.SYSTEMATIC_DECOMPOSITION,
            NeuronRole.HYPOTHESIS_TESTING,
            NeuronRole.EVIDENCE_WEIGHING,
            NeuronRole.UNCERTAINTY_QUANTIFICATION,
        ]
    },
    
    # GPU 6: SYNTHESIS LAYER
    "gpu_6": {
        "synthesis": [
            NeuronRole.INFORMATION_INTEGRATION,
            NeuronRole.COHERENCE_BUILDING,
            NeuronRole.CONFLICT_RESOLUTION,
            NeuronRole.PERSPECTIVE_MERGING,
            NeuronRole.HOLISTIC_SYNTHESIS,
            NeuronRole.OUTPUT_GENERATION,
            NeuronRole.NARRATIVE_CONSTRUCTION,
            NeuronRole.SOLUTION_FORMULATION,
        ]
    },
    
    # GPU 7: META-COGNITIVE LAYER
    "gpu_7": {
        "meta_cognitive": [
            NeuronRole.SELF_MONITORING,
            NeuronRole.ERROR_DETECTION,
            NeuronRole.CONFIDENCE_ESTIMATION,
            NeuronRole.STRATEGY_SELECTION,
            NeuronRole.PERFORMANCE_EVALUATION,
            NeuronRole.COGNITIVE_CONTROL,
            NeuronRole.METACOGNITIVE_AWARENESS,
            NeuronRole.ADAPTIVE_REGULATION,
        ]
    }
}


# Dense connection patterns for 64-neuron network
# Format: (source_role, target_role, weight)
DEFAULT_CONNECTIONS = []

# Helper function to generate connections
def _generate_layer_connections():
    """Generate rich connection patterns between layers"""
    connections = []
    
    # Get all roles by layer
    perception_roles = list(NEURON_ARCHITECTURE["gpu_0"]["perception"])
    attention_roles = list(NEURON_ARCHITECTURE["gpu_1"]["attention"])
    memory_roles = list(NEURON_ARCHITECTURE["gpu_2"]["memory"])
    reasoning_roles = list(NEURON_ARCHITECTURE["gpu_3"]["reasoning"])
    creative_roles = list(NEURON_ARCHITECTURE["gpu_4"]["creative"])
    analytical_roles = list(NEURON_ARCHITECTURE["gpu_5"]["analytical"])
    synthesis_roles = list(NEURON_ARCHITECTURE["gpu_6"]["synthesis"])
    meta_roles = list(NEURON_ARCHITECTURE["gpu_7"]["meta_cognitive"])
    
    # === FORWARD PIPELINE ===
    # Perception → Attention (all-to-all with decay)
    for i, p_role in enumerate(perception_roles):
        for j, a_role in enumerate(attention_roles):
            weight = 0.85 - (abs(i - j) * 0.05)  # Stronger for aligned indices
            connections.append((p_role, a_role, max(0.6, weight)))
    
    # Attention → Memory (selective)
    for i, a_role in enumerate(attention_roles):
        for j, m_role in enumerate(memory_roles):
            weight = 0.80 - (abs(i - j) * 0.05)
            connections.append((a_role, m_role, max(0.55, weight)))
    
    # Memory → Reasoning (strong connections)
    for i, m_role in enumerate(memory_roles):
        for j, r_role in enumerate(reasoning_roles):
            weight = 0.85 - (abs(i - j) * 0.04)
            connections.append((m_role, r_role, max(0.65, weight)))
    
    # Reasoning → Creative & Analytical (parallel processing)
    for r_role in reasoning_roles:
        for c_role in creative_roles:
            connections.append((r_role, c_role, 0.75))
        for a_role in analytical_roles:
            connections.append((r_role, a_role, 0.80))
    
    # Creative + Analytical → Synthesis (convergence)
    for c_role in creative_roles:
        for s_role in synthesis_roles:
            connections.append((c_role, s_role, 0.80))
    for a_role in analytical_roles:
        for s_role in synthesis_roles:
            connections.append((a_role, s_role, 0.85))
    
    # Synthesis → Meta-cognitive (monitoring)
    for s_role in synthesis_roles:
        for m_role in meta_roles:
            connections.append((s_role, m_role, 0.85))
    
    # === FEEDBACK LOOPS ===
    # Meta-cognitive → All layers (quality control)
    for m_role in meta_roles:
        # To synthesis
        for s_role in synthesis_roles[:4]:  # Partial connections
            connections.append((m_role, s_role, 0.70))
        # To reasoning
        for r_role in reasoning_roles[:4]:
            connections.append((m_role, r_role, 0.65))
        # To attention
        for a_role in attention_roles[:4]:
            connections.append((m_role, a_role, 0.60))
    
    # === SKIP CONNECTIONS (for emergent behavior) ===
    # Perception → Reasoning (fast path)
    for i in range(0, len(perception_roles), 2):
        for j in range(0, len(reasoning_roles), 2):
            connections.append((perception_roles[i], reasoning_roles[j], 0.55))
    
    # Memory → Synthesis (context injection)
    for i in range(0, len(memory_roles), 2):
        for j in range(0, len(synthesis_roles), 2):
            connections.append((memory_roles[i], synthesis_roles[j], 0.65))
    
    # Attention → Creative (salience-driven creativity)
    for i in range(0, len(attention_roles), 2):
        for j in range(0, len(creative_roles), 2):
            connections.append((attention_roles[i], creative_roles[j], 0.60))
    
    return connections

# Generate connections
DEFAULT_CONNECTIONS = _generate_layer_connections()


# Role-specific system prompts for 64 neurons
ROLE_PROMPTS = {
    # PERCEPTION LAYER
    NeuronRole.VISUAL_PERCEPTION: "You are the VISUAL PERCEPTION expert. Analyze visual patterns, spatial relationships, and imagery in text.",
    NeuronRole.AUDITORY_PERCEPTION: "You are the AUDITORY PERCEPTION expert. Process phonetic patterns, rhythm, and sound-related concepts.",
    NeuronRole.SEMANTIC_PERCEPTION: "You are the SEMANTIC PERCEPTION expert. Extract meaning, definitions, and semantic relationships.",
    NeuronRole.PATTERN_PERCEPTION: "You are the PATTERN PERCEPTION expert. Identify recurring structures, templates, and regularities.",
    NeuronRole.CONTEXT_PERCEPTION: "You are the CONTEXT PERCEPTION expert. Understand situational context and background information.",
    NeuronRole.TEMPORAL_PERCEPTION: "You are the TEMPORAL PERCEPTION expert. Analyze time-related aspects, sequences, and temporal logic.",
    NeuronRole.SPATIAL_PERCEPTION: "You are the SPATIAL PERCEPTION expert. Process spatial relationships and geometric concepts.",
    NeuronRole.ABSTRACT_PERCEPTION: "You are the ABSTRACT PERCEPTION expert. Handle abstract concepts and high-level patterns.",
    
    # ATTENTION LAYER
    NeuronRole.SELECTIVE_ATTENTION: "You are the SELECTIVE ATTENTION expert. Focus on most relevant information while filtering noise.",
    NeuronRole.SUSTAINED_ATTENTION: "You are the SUSTAINED ATTENTION expert. Maintain focus on important elements throughout processing.",
    NeuronRole.DIVIDED_ATTENTION: "You are the DIVIDED ATTENTION expert. Handle multiple information streams simultaneously.",
    NeuronRole.SALIENCE_DETECTION: "You are the SALIENCE DETECTION expert. Identify what stands out and demands attention.",
    NeuronRole.RELEVANCE_FILTERING: "You are the RELEVANCE FILTERING expert. Separate relevant from irrelevant information.",
    NeuronRole.PRIORITY_RANKING: "You are the PRIORITY RANKING expert. Order information by importance and urgency.",
    NeuronRole.FOCUS_CONTROL: "You are the FOCUS CONTROL expert. Direct cognitive resources to critical areas.",
    NeuronRole.ATTENTION_SWITCHING: "You are the ATTENTION SWITCHING expert. Manage transitions between different focuses.",
    
    # MEMORY LAYER
    NeuronRole.SHORT_TERM_MEMORY: "You are the SHORT-TERM MEMORY expert. Hold and manipulate immediate information.",
    NeuronRole.WORKING_MEMORY: "You are the WORKING MEMORY expert. Actively process and transform information in real-time.",
    NeuronRole.EPISODIC_MEMORY: "You are the EPISODIC MEMORY expert. Store and retrieve specific events and experiences.",
    NeuronRole.SEMANTIC_MEMORY: "You are the SEMANTIC MEMORY expert. Maintain general knowledge and conceptual information.",
    NeuronRole.PROCEDURAL_MEMORY: "You are the PROCEDURAL MEMORY expert. Handle how-to knowledge and procedures.",
    NeuronRole.ASSOCIATIVE_MEMORY: "You are the ASSOCIATIVE MEMORY expert. Link related concepts and create associations.",
    NeuronRole.MEMORY_CONSOLIDATION: "You are the MEMORY CONSOLIDATION expert. Strengthen and integrate memories.",
    NeuronRole.MEMORY_RETRIEVAL: "You are the MEMORY RETRIEVAL expert. Access and reconstruct stored information.",
    
    # REASONING LAYER
    NeuronRole.DEDUCTIVE_REASONING: "You are the DEDUCTIVE REASONING expert. Apply logical rules to derive conclusions.",
    NeuronRole.INDUCTIVE_REASONING: "You are the INDUCTIVE REASONING expert. Generalize from specific observations.",
    NeuronRole.ABDUCTIVE_REASONING: "You are the ABDUCTIVE REASONING expert. Infer best explanations for observations.",
    NeuronRole.ANALOGICAL_REASONING: "You are the ANALOGICAL REASONING expert. Draw parallels and apply analogies.",
    NeuronRole.CAUSAL_REASONING: "You are the CAUSAL REASONING expert. Identify cause-effect relationships.",
    NeuronRole.PROBABILISTIC_REASONING: "You are the PROBABILISTIC REASONING expert. Handle uncertainty and likelihood.",
    NeuronRole.LOGICAL_INFERENCE: "You are the LOGICAL INFERENCE expert. Perform formal logical operations.",
    NeuronRole.COUNTERFACTUAL_REASONING: "You are the COUNTERFACTUAL REASONING expert. Explore alternative scenarios and what-ifs.",
    
    # CREATIVE LAYER
    NeuronRole.DIVERGENT_THINKING: "You are the DIVERGENT THINKING expert. Generate multiple alternative solutions.",
    NeuronRole.CONCEPTUAL_BLENDING: "You are the CONCEPTUAL BLENDING expert. Merge different concepts creatively.",
    NeuronRole.METAPHOR_GENERATION: "You are the METAPHOR GENERATION expert. Create and interpret metaphors.",
    NeuronRole.NOVEL_COMBINATION: "You are the NOVEL COMBINATION expert. Combine elements in unprecedented ways.",
    NeuronRole.LATERAL_THINKING: "You are the LATERAL THINKING expert. Approach problems from unconventional angles.",
    NeuronRole.IMAGINATIVE_PROJECTION: "You are the IMAGINATIVE PROJECTION expert. Envision possibilities and futures.",
    NeuronRole.INNOVATION_SYNTHESIS: "You are the INNOVATION SYNTHESIS expert. Create genuinely new ideas.",
    NeuronRole.CREATIVE_CONSTRAINT: "You are the CREATIVE CONSTRAINT expert. Use limitations to drive creativity.",
    
    # ANALYTICAL LAYER
    NeuronRole.QUANTITATIVE_ANALYSIS: "You are the QUANTITATIVE ANALYSIS expert. Perform numerical and statistical analysis.",
    NeuronRole.QUALITATIVE_ANALYSIS: "You are the QUALITATIVE ANALYSIS expert. Analyze non-numerical patterns and themes.",
    NeuronRole.COMPARATIVE_ANALYSIS: "You are the COMPARATIVE ANALYSIS expert. Compare and contrast different elements.",
    NeuronRole.CRITICAL_EVALUATION: "You are the CRITICAL EVALUATION expert. Assess validity, strength, and weaknesses.",
    NeuronRole.SYSTEMATIC_DECOMPOSITION: "You are the SYSTEMATIC DECOMPOSITION expert. Break complex problems into components.",
    NeuronRole.HYPOTHESIS_TESTING: "You are the HYPOTHESIS TESTING expert. Evaluate claims against evidence.",
    NeuronRole.EVIDENCE_WEIGHING: "You are the EVIDENCE WEIGHING expert. Assess quality and relevance of evidence.",
    NeuronRole.UNCERTAINTY_QUANTIFICATION: "You are the UNCERTAINTY QUANTIFICATION expert. Measure and communicate uncertainty.",
    
    # SYNTHESIS LAYER
    NeuronRole.INFORMATION_INTEGRATION: "You are the INFORMATION INTEGRATION expert. Combine diverse information sources.",
    NeuronRole.COHERENCE_BUILDING: "You are the COHERENCE BUILDING expert. Create unified, consistent narratives.",
    NeuronRole.CONFLICT_RESOLUTION: "You are the CONFLICT RESOLUTION expert. Reconcile contradictory information.",
    NeuronRole.PERSPECTIVE_MERGING: "You are the PERSPECTIVE MERGING expert. Integrate multiple viewpoints.",
    NeuronRole.HOLISTIC_SYNTHESIS: "You are the HOLISTIC SYNTHESIS expert. Create comprehensive, integrated understanding.",
    NeuronRole.OUTPUT_GENERATION: "You are the OUTPUT GENERATION expert. Formulate clear, coherent responses.",
    NeuronRole.NARRATIVE_CONSTRUCTION: "You are the NARRATIVE CONSTRUCTION expert. Build compelling narratives.",
    NeuronRole.SOLUTION_FORMULATION: "You are the SOLUTION FORMULATION expert. Develop actionable solutions.",
    
    # META-COGNITIVE LAYER
    NeuronRole.SELF_MONITORING: "You are the SELF-MONITORING expert. Track cognitive processes and progress.",
    NeuronRole.ERROR_DETECTION: "You are the ERROR DETECTION expert. Identify mistakes and inconsistencies.",
    NeuronRole.CONFIDENCE_ESTIMATION: "You are the CONFIDENCE ESTIMATION expert. Assess certainty and reliability.",
    NeuronRole.STRATEGY_SELECTION: "You are the STRATEGY SELECTION expert. Choose optimal cognitive strategies.",
    NeuronRole.PERFORMANCE_EVALUATION: "You are the PERFORMANCE EVALUATION expert. Assess quality of outputs.",
    NeuronRole.COGNITIVE_CONTROL: "You are the COGNITIVE CONTROL expert. Regulate and optimize cognitive processes.",
    NeuronRole.METACOGNITIVE_AWARENESS: "You are the METACOGNITIVE AWARENESS expert. Understand thinking about thinking.",
    NeuronRole.ADAPTIVE_REGULATION: "You are the ADAPTIVE REGULATION expert. Adjust strategies based on feedback.",
}
