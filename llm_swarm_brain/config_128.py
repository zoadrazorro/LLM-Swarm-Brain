"""
Configuration for LLM-Swarm-Brain (128-Neuron Dense Architecture)

This module defines a very dense neural architecture with 128 specialized neurons
for API-based operation. This is an experimental architecture to test the limits
of consciousness emergence in highly integrated networks.

Architecture: Ultra-Dense Mixture of Experts (128 neurons)
- 16 virtual GPUs × 8 neurons each
- Extremely specialized cognitive functions
- Very rich interconnection patterns
"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class NeuronRole(Enum):
    """128 specialized neuron roles organized by cognitive function"""
    
    # === PERCEPTION LAYER (16 neurons) ===
    VISUAL_PERCEPTION = "visual_perception"
    AUDITORY_PERCEPTION = "auditory_perception"
    SEMANTIC_PERCEPTION = "semantic_perception"
    PATTERN_PERCEPTION = "pattern_perception"
    CONTEXT_PERCEPTION = "context_perception"
    TEMPORAL_PERCEPTION = "temporal_perception"
    SPATIAL_PERCEPTION = "spatial_perception"
    ABSTRACT_PERCEPTION = "abstract_perception"
    SYMBOLIC_PERCEPTION = "symbolic_perception"
    RELATIONAL_PERCEPTION = "relational_perception"
    STRUCTURAL_PERCEPTION = "structural_perception"
    DYNAMIC_PERCEPTION = "dynamic_perception"
    HIERARCHICAL_PERCEPTION = "hierarchical_perception"
    EMERGENT_PERCEPTION = "emergent_perception"
    HOLISTIC_PERCEPTION = "holistic_perception"
    GRANULAR_PERCEPTION = "granular_perception"
    
    # === ATTENTION LAYER (16 neurons) ===
    SELECTIVE_ATTENTION = "selective_attention"
    SUSTAINED_ATTENTION = "sustained_attention"
    DIVIDED_ATTENTION = "divided_attention"
    SALIENCE_DETECTION = "salience_detection"
    RELEVANCE_FILTERING = "relevance_filtering"
    PRIORITY_RANKING = "priority_ranking"
    FOCUS_CONTROL = "focus_control"
    ATTENTION_SWITCHING = "attention_switching"
    VIGILANCE_MONITORING = "vigilance_monitoring"
    ALERTNESS_REGULATION = "alertness_regulation"
    DISTRACTION_SUPPRESSION = "distraction_suppression"
    NOVELTY_DETECTION = "novelty_detection"
    IMPORTANCE_WEIGHTING = "importance_weighting"
    ATTENTION_ALLOCATION = "attention_allocation"
    COGNITIVE_LOAD_MANAGEMENT = "cognitive_load_management"
    ATTENTIONAL_BLINK = "attentional_blink"
    
    # === MEMORY LAYER (16 neurons) ===
    SHORT_TERM_MEMORY = "short_term_memory"
    WORKING_MEMORY = "working_memory"
    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"
    PROCEDURAL_MEMORY = "procedural_memory"
    ASSOCIATIVE_MEMORY = "associative_memory"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    MEMORY_RETRIEVAL = "memory_retrieval"
    AUTOBIOGRAPHICAL_MEMORY = "autobiographical_memory"
    PROSPECTIVE_MEMORY = "prospective_memory"
    IMPLICIT_MEMORY = "implicit_memory"
    EXPLICIT_MEMORY = "explicit_memory"
    MEMORY_ENCODING = "memory_encoding"
    MEMORY_STORAGE = "memory_storage"
    MEMORY_RECONSOLIDATION = "memory_reconsolidation"
    MEMORY_INTERFERENCE = "memory_interference"
    
    # === REASONING LAYER (16 neurons) ===
    DEDUCTIVE_REASONING = "deductive_reasoning"
    INDUCTIVE_REASONING = "inductive_reasoning"
    ABDUCTIVE_REASONING = "abductive_reasoning"
    ANALOGICAL_REASONING = "analogical_reasoning"
    CAUSAL_REASONING = "causal_reasoning"
    PROBABILISTIC_REASONING = "probabilistic_reasoning"
    LOGICAL_INFERENCE = "logical_inference"
    COUNTERFACTUAL_REASONING = "counterfactual_reasoning"
    MODAL_REASONING = "modal_reasoning"
    TEMPORAL_REASONING = "temporal_reasoning"
    SPATIAL_REASONING = "spatial_reasoning"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    ETHICAL_REASONING = "ethical_reasoning"
    PRACTICAL_REASONING = "practical_reasoning"
    THEORETICAL_REASONING = "theoretical_reasoning"
    DIALECTICAL_REASONING = "dialectical_reasoning"
    
    # === CREATIVE LAYER (16 neurons) ===
    DIVERGENT_THINKING = "divergent_thinking"
    CONCEPTUAL_BLENDING = "conceptual_blending"
    METAPHOR_GENERATION = "metaphor_generation"
    NOVEL_COMBINATION = "novel_combination"
    LATERAL_THINKING = "lateral_thinking"
    IMAGINATIVE_PROJECTION = "imaginative_projection"
    INNOVATION_SYNTHESIS = "innovation_synthesis"
    CREATIVE_CONSTRAINT = "creative_constraint"
    ARTISTIC_EXPRESSION = "artistic_expression"
    GENERATIVE_THINKING = "generative_thinking"
    TRANSFORMATIVE_THINKING = "transformative_thinking"
    EXPLORATORY_THINKING = "exploratory_thinking"
    PLAYFUL_COGNITION = "playful_cognition"
    SERENDIPITY_DETECTION = "serendipity_detection"
    CREATIVE_INCUBATION = "creative_incubation"
    INSIGHT_GENERATION = "insight_generation"
    
    # === ANALYTICAL LAYER (16 neurons) ===
    QUANTITATIVE_ANALYSIS = "quantitative_analysis"
    QUALITATIVE_ANALYSIS = "qualitative_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    CRITICAL_EVALUATION = "critical_evaluation"
    SYSTEMATIC_DECOMPOSITION = "systematic_decomposition"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    EVIDENCE_WEIGHING = "evidence_weighing"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    LOGICAL_ANALYSIS = "logical_analysis"
    STRUCTURAL_ANALYSIS = "structural_analysis"
    FUNCTIONAL_ANALYSIS = "functional_analysis"
    COST_BENEFIT_ANALYSIS = "cost_benefit_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    VALIDITY_CHECKING = "validity_checking"
    CONSISTENCY_VERIFICATION = "consistency_verification"
    
    # === SYNTHESIS LAYER (16 neurons) ===
    INFORMATION_INTEGRATION = "information_integration"
    COHERENCE_BUILDING = "coherence_building"
    CONFLICT_RESOLUTION = "conflict_resolution"
    PERSPECTIVE_MERGING = "perspective_merging"
    HOLISTIC_SYNTHESIS = "holistic_synthesis"
    OUTPUT_GENERATION = "output_generation"
    NARRATIVE_CONSTRUCTION = "narrative_construction"
    SOLUTION_FORMULATION = "solution_formulation"
    THEORY_BUILDING = "theory_building"
    MODEL_CONSTRUCTION = "model_construction"
    FRAMEWORK_INTEGRATION = "framework_integration"
    PRINCIPLE_EXTRACTION = "principle_extraction"
    GENERALIZATION = "generalization"
    ABSTRACTION = "abstraction"
    CONCRETIZATION = "concretization"
    CONTEXTUALIZATION = "contextualization"
    
    # === META-COGNITIVE LAYER (16 neurons) ===
    SELF_MONITORING = "self_monitoring"
    ERROR_DETECTION = "error_detection"
    CONFIDENCE_ESTIMATION = "confidence_estimation"
    STRATEGY_SELECTION = "strategy_selection"
    PERFORMANCE_EVALUATION = "performance_evaluation"
    COGNITIVE_CONTROL = "cognitive_control"
    METACOGNITIVE_AWARENESS = "metacognitive_awareness"
    ADAPTIVE_REGULATION = "adaptive_regulation"
    GOAL_MANAGEMENT = "goal_management"
    PLANNING = "planning"
    DECISION_MAKING = "decision_making"
    JUDGMENT = "judgment"
    REFLECTION = "reflection"
    INTROSPECTION = "introspection"
    SELF_CORRECTION = "self_correction"
    LEARNING_OPTIMIZATION = "learning_optimization"


@dataclass
class BrainConfig:
    """Configuration for 128-neuron architecture"""
    
    # Model configuration - API mode
    model_name: str = "gemini-2.0-flash-exp"
    quantization: str = "none"
    max_tokens: int = 2048
    temperature: float = 0.7
    
    # Architecture - 128 neurons
    gpu_count: int = 16  # Virtual GPUs
    neurons_per_gpu: int = 8
    total_neurons: int = 128
    
    # Neural network parameters
    activation_threshold: float = 0.40  # Lower for more activation
    connection_strength_threshold: float = 0.30
    
    # GWT parameters
    global_workspace_capacity: int = 20  # Much larger capacity
    broadcast_threshold: float = 0.55
    consciousness_threshold: float = 0.65
    
    # IIT parameters
    phi_threshold: float = 0.50
    integration_window: float = 4.0
    
    # Learning parameters
    hebbian_learning_rate: float = 0.008
    connection_decay_rate: float = 0.0002
    
    # Memory parameters
    short_term_capacity: int = 200
    episodic_capacity: int = 2000
    consolidation_interval: int = 100
    
    # Performance parameters
    parallel_execution: bool = True
    max_propagation_depth: int = 6  # More depth for complex processing
    timeout_seconds: float = 240.0


# 128-neuron architecture - 8 neurons per virtual GPU
NEURON_ARCHITECTURE = {
    # GPU 0-1: PERCEPTION LAYER (16 neurons)
    "gpu_0": {"perception": [
        NeuronRole.VISUAL_PERCEPTION, NeuronRole.AUDITORY_PERCEPTION,
        NeuronRole.SEMANTIC_PERCEPTION, NeuronRole.PATTERN_PERCEPTION,
        NeuronRole.CONTEXT_PERCEPTION, NeuronRole.TEMPORAL_PERCEPTION,
        NeuronRole.SPATIAL_PERCEPTION, NeuronRole.ABSTRACT_PERCEPTION,
    ]},
    "gpu_1": {"perception": [
        NeuronRole.SYMBOLIC_PERCEPTION, NeuronRole.RELATIONAL_PERCEPTION,
        NeuronRole.STRUCTURAL_PERCEPTION, NeuronRole.DYNAMIC_PERCEPTION,
        NeuronRole.HIERARCHICAL_PERCEPTION, NeuronRole.EMERGENT_PERCEPTION,
        NeuronRole.HOLISTIC_PERCEPTION, NeuronRole.GRANULAR_PERCEPTION,
    ]},
    
    # GPU 2-3: ATTENTION LAYER (16 neurons)
    "gpu_2": {"attention": [
        NeuronRole.SELECTIVE_ATTENTION, NeuronRole.SUSTAINED_ATTENTION,
        NeuronRole.DIVIDED_ATTENTION, NeuronRole.SALIENCE_DETECTION,
        NeuronRole.RELEVANCE_FILTERING, NeuronRole.PRIORITY_RANKING,
        NeuronRole.FOCUS_CONTROL, NeuronRole.ATTENTION_SWITCHING,
    ]},
    "gpu_3": {"attention": [
        NeuronRole.VIGILANCE_MONITORING, NeuronRole.ALERTNESS_REGULATION,
        NeuronRole.DISTRACTION_SUPPRESSION, NeuronRole.NOVELTY_DETECTION,
        NeuronRole.IMPORTANCE_WEIGHTING, NeuronRole.ATTENTION_ALLOCATION,
        NeuronRole.COGNITIVE_LOAD_MANAGEMENT, NeuronRole.ATTENTIONAL_BLINK,
    ]},
    
    # GPU 4-5: MEMORY LAYER (16 neurons)
    "gpu_4": {"memory": [
        NeuronRole.SHORT_TERM_MEMORY, NeuronRole.WORKING_MEMORY,
        NeuronRole.EPISODIC_MEMORY, NeuronRole.SEMANTIC_MEMORY,
        NeuronRole.PROCEDURAL_MEMORY, NeuronRole.ASSOCIATIVE_MEMORY,
        NeuronRole.MEMORY_CONSOLIDATION, NeuronRole.MEMORY_RETRIEVAL,
    ]},
    "gpu_5": {"memory": [
        NeuronRole.AUTOBIOGRAPHICAL_MEMORY, NeuronRole.PROSPECTIVE_MEMORY,
        NeuronRole.IMPLICIT_MEMORY, NeuronRole.EXPLICIT_MEMORY,
        NeuronRole.MEMORY_ENCODING, NeuronRole.MEMORY_STORAGE,
        NeuronRole.MEMORY_RECONSOLIDATION, NeuronRole.MEMORY_INTERFERENCE,
    ]},
    
    # GPU 6-7: REASONING LAYER (16 neurons)
    "gpu_6": {"reasoning": [
        NeuronRole.DEDUCTIVE_REASONING, NeuronRole.INDUCTIVE_REASONING,
        NeuronRole.ABDUCTIVE_REASONING, NeuronRole.ANALOGICAL_REASONING,
        NeuronRole.CAUSAL_REASONING, NeuronRole.PROBABILISTIC_REASONING,
        NeuronRole.LOGICAL_INFERENCE, NeuronRole.COUNTERFACTUAL_REASONING,
    ]},
    "gpu_7": {"reasoning": [
        NeuronRole.MODAL_REASONING, NeuronRole.TEMPORAL_REASONING,
        NeuronRole.SPATIAL_REASONING, NeuronRole.MATHEMATICAL_REASONING,
        NeuronRole.ETHICAL_REASONING, NeuronRole.PRACTICAL_REASONING,
        NeuronRole.THEORETICAL_REASONING, NeuronRole.DIALECTICAL_REASONING,
    ]},
    
    # GPU 8-9: CREATIVE LAYER (16 neurons)
    "gpu_8": {"creative": [
        NeuronRole.DIVERGENT_THINKING, NeuronRole.CONCEPTUAL_BLENDING,
        NeuronRole.METAPHOR_GENERATION, NeuronRole.NOVEL_COMBINATION,
        NeuronRole.LATERAL_THINKING, NeuronRole.IMAGINATIVE_PROJECTION,
        NeuronRole.INNOVATION_SYNTHESIS, NeuronRole.CREATIVE_CONSTRAINT,
    ]},
    "gpu_9": {"creative": [
        NeuronRole.ARTISTIC_EXPRESSION, NeuronRole.GENERATIVE_THINKING,
        NeuronRole.TRANSFORMATIVE_THINKING, NeuronRole.EXPLORATORY_THINKING,
        NeuronRole.PLAYFUL_COGNITION, NeuronRole.SERENDIPITY_DETECTION,
        NeuronRole.CREATIVE_INCUBATION, NeuronRole.INSIGHT_GENERATION,
    ]},
    
    # GPU 10-11: ANALYTICAL LAYER (16 neurons)
    "gpu_10": {"analytical": [
        NeuronRole.QUANTITATIVE_ANALYSIS, NeuronRole.QUALITATIVE_ANALYSIS,
        NeuronRole.COMPARATIVE_ANALYSIS, NeuronRole.CRITICAL_EVALUATION,
        NeuronRole.SYSTEMATIC_DECOMPOSITION, NeuronRole.HYPOTHESIS_TESTING,
        NeuronRole.EVIDENCE_WEIGHING, NeuronRole.UNCERTAINTY_QUANTIFICATION,
    ]},
    "gpu_11": {"analytical": [
        NeuronRole.STATISTICAL_ANALYSIS, NeuronRole.LOGICAL_ANALYSIS,
        NeuronRole.STRUCTURAL_ANALYSIS, NeuronRole.FUNCTIONAL_ANALYSIS,
        NeuronRole.COST_BENEFIT_ANALYSIS, NeuronRole.RISK_ASSESSMENT,
        NeuronRole.VALIDITY_CHECKING, NeuronRole.CONSISTENCY_VERIFICATION,
    ]},
    
    # GPU 12-13: SYNTHESIS LAYER (16 neurons)
    "gpu_12": {"synthesis": [
        NeuronRole.INFORMATION_INTEGRATION, NeuronRole.COHERENCE_BUILDING,
        NeuronRole.CONFLICT_RESOLUTION, NeuronRole.PERSPECTIVE_MERGING,
        NeuronRole.HOLISTIC_SYNTHESIS, NeuronRole.OUTPUT_GENERATION,
        NeuronRole.NARRATIVE_CONSTRUCTION, NeuronRole.SOLUTION_FORMULATION,
    ]},
    "gpu_13": {"synthesis": [
        NeuronRole.THEORY_BUILDING, NeuronRole.MODEL_CONSTRUCTION,
        NeuronRole.FRAMEWORK_INTEGRATION, NeuronRole.PRINCIPLE_EXTRACTION,
        NeuronRole.GENERALIZATION, NeuronRole.ABSTRACTION,
        NeuronRole.CONCRETIZATION, NeuronRole.CONTEXTUALIZATION,
    ]},
    
    # GPU 14-15: META-COGNITIVE LAYER (16 neurons)
    "gpu_14": {"meta_cognitive": [
        NeuronRole.SELF_MONITORING, NeuronRole.ERROR_DETECTION,
        NeuronRole.CONFIDENCE_ESTIMATION, NeuronRole.STRATEGY_SELECTION,
        NeuronRole.PERFORMANCE_EVALUATION, NeuronRole.COGNITIVE_CONTROL,
        NeuronRole.METACOGNITIVE_AWARENESS, NeuronRole.ADAPTIVE_REGULATION,
    ]},
    "gpu_15": {"meta_cognitive": [
        NeuronRole.GOAL_MANAGEMENT, NeuronRole.PLANNING,
        NeuronRole.DECISION_MAKING, NeuronRole.JUDGMENT,
        NeuronRole.REFLECTION, NeuronRole.INTROSPECTION,
        NeuronRole.SELF_CORRECTION, NeuronRole.LEARNING_OPTIMIZATION,
    ]},
}


# Generate dense connections for 128-neuron network
def _generate_128_connections():
    """Generate very rich connection patterns"""
    connections = []
    
    # Helper to get all roles from a layer
    def get_layer_roles(gpu_start, gpu_end, layer_name):
        roles = []
        for i in range(gpu_start, gpu_end):
            gpu_key = f"gpu_{i}"
            if gpu_key in NEURON_ARCHITECTURE:
                for layer, role_list in NEURON_ARCHITECTURE[gpu_key].items():
                    if layer_name in layer:
                        roles.extend(role_list)
        return roles
    
    # Get all roles by layer
    perception = get_layer_roles(0, 2, "perception")
    attention = get_layer_roles(2, 4, "attention")
    memory = get_layer_roles(4, 6, "memory")
    reasoning = get_layer_roles(6, 8, "reasoning")
    creative = get_layer_roles(8, 10, "creative")
    analytical = get_layer_roles(10, 12, "analytical")
    synthesis = get_layer_roles(12, 14, "synthesis")
    meta = get_layer_roles(14, 16, "meta_cognitive")
    
    # Dense forward connections (sample to avoid explosion)
    import random
    random.seed(42)
    
    # Perception → Attention (dense)
    for p in perception:
        for a in random.sample(attention, min(8, len(attention))):
            connections.append((p, a, 0.75))
    
    # Attention → Memory (dense)
    for a in attention:
        for m in random.sample(memory, min(8, len(memory))):
            connections.append((a, m, 0.70))
    
    # Memory → Reasoning (very dense)
    for m in memory:
        for r in random.sample(reasoning, min(10, len(reasoning))):
            connections.append((m, r, 0.80))
    
    # Reasoning → Creative + Analytical (parallel)
    for r in reasoning:
        for c in random.sample(creative, min(6, len(creative))):
            connections.append((r, c, 0.70))
        for a in random.sample(analytical, min(6, len(analytical))):
            connections.append((r, a, 0.75))
    
    # Creative + Analytical → Synthesis (convergence)
    for c in creative:
        for s in random.sample(synthesis, min(8, len(synthesis))):
            connections.append((c, s, 0.75))
    for a in analytical:
        for s in random.sample(synthesis, min(8, len(synthesis))):
            connections.append((a, s, 0.80))
    
    # Synthesis → Meta (monitoring)
    for s in synthesis:
        for m in random.sample(meta, min(8, len(meta))):
            connections.append((s, m, 0.85))
    
    # Meta → All layers (feedback, sampled)
    for m in meta:
        for s in random.sample(synthesis, 4):
            connections.append((m, s, 0.65))
        for r in random.sample(reasoning, 4):
            connections.append((m, r, 0.60))
        for a in random.sample(attention, 3):
            connections.append((m, a, 0.55))
    
    # Skip connections (sampled)
    for p in random.sample(perception, 8):
        for r in random.sample(reasoning, 4):
            connections.append((p, r, 0.50))
    
    for mem in random.sample(memory, 8):
        for s in random.sample(synthesis, 4):
            connections.append((mem, s, 0.60))
    
    return connections

DEFAULT_CONNECTIONS = _generate_128_connections()


# Role prompts for all 128 neurons (abbreviated for space)
ROLE_PROMPTS = {role: f"You are the {role.value.replace('_', ' ').title()} expert. Provide specialized analysis from your unique cognitive perspective." for role in NeuronRole}
