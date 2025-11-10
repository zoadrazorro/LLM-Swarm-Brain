"""
Configuration for LLM-Swarm-Brain (8× H100 SXM5 80GB Configuration)

This module defines the neural architecture configuration for 8 neurons
(1 per GPU) using large MoE-style models (Qwen2.5-72B-Instruct),
aligned with Global Workspace Theory (GWT) and
Integrated Information Theory (IIT).

Architecture: Mixture of Experts (MoE) - Each neuron is a specialized
expert running a full 72B parameter model in 4-bit quantization (~40GB VRAM).
"""


from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class NeuronRole(Enum):
    """Neuron roles based on cognitive architecture (8 total roles - MoE experts)
    
    Each neuron is a specialized expert running Qwen2.5-72B-Instruct.
    Architecture follows cognitive processing pipeline:
    Perception → Memory → Reasoning → Action
    """

    # === GPU 0: PERCEPTION EXPERT ===
    PERCEPTION_EXPERT = "perception_expert"
    
    # === GPU 1: ATTENTION EXPERT ===
    ATTENTION_EXPERT = "attention_expert"
    
    # === GPU 2: MEMORY EXPERT ===
    MEMORY_EXPERT = "memory_expert"
    
    # === GPU 3: REASONING EXPERT ===
    REASONING_EXPERT = "reasoning_expert"
    
    # === GPU 4: CREATIVE EXPERT ===
    CREATIVE_EXPERT = "creative_expert"
    
    # === GPU 5: ANALYTICAL EXPERT ===
    ANALYTICAL_EXPERT = "analytical_expert"
    
    # === GPU 6: SYNTHESIS EXPERT ===
    SYNTHESIS_EXPERT = "synthesis_expert"
    
    # === GPU 7: META-COGNITIVE EXPERT ===
    META_COGNITIVE_EXPERT = "meta_cognitive_expert"


@dataclass
class BrainConfig:
    """Configuration for the neural brain architecture"""

    # Model configuration
    # Local mode: Qwen2.5-72B-Instruct (72B params, ~40GB in 4-bit)
    # API mode: meta-llama/Meta-Llama-3.1-405B-Instruct (via Hyperbolic)
    model_name: str = "Qwen/Qwen2.5-72B-Instruct"
    quantization: str = "4bit"  # BitsAndBytes 4-bit quantization (local only)
    max_tokens: int = 2048  # Increased for larger models
    temperature: float = 0.7

    # GPU allocation (8× H100 SXM5 80GB) - MoE Architecture
    gpu_count: int = 8
    neurons_per_gpu: int = 1  # One expert per GPU
    total_neurons: int = 8  # 8 specialized experts

    # Neural network parameters - MoE routing
    activation_threshold: float = 0.5  # Lower threshold for expert activation
    connection_strength_threshold: float = 0.4

    # GWT parameters (Global Workspace Theory)
    global_workspace_capacity: int = 4  # Fewer neurons, higher capacity each
    broadcast_threshold: float = 0.65
    consciousness_threshold: float = 0.75

    # IIT parameters (Integrated Information Theory)
    phi_threshold: float = 0.6
    integration_window: float = 2.5

    # Learning parameters
    hebbian_learning_rate: float = 0.015
    connection_decay_rate: float = 0.0005

    # Memory parameters - Enhanced for larger models
    short_term_capacity: int = 50
    episodic_capacity: int = 500
    consolidation_interval: int = 25

    # Performance parameters - Optimized for large models
    parallel_execution: bool = True
    max_propagation_depth: int = 4  # Fewer steps, each more powerful
    timeout_seconds: float = 120.0  # Longer timeout for 72B models


# 8-neuron MoE architecture - 1 expert per GPU
NEURON_ARCHITECTURE = {
    # GPU 0: PERCEPTION EXPERT
    "gpu_0": {
        "perception": [
            NeuronRole.PERCEPTION_EXPERT,
        ]
    },
    
    # GPU 1: ATTENTION EXPERT
    "gpu_1": {
        "attention": [
            NeuronRole.ATTENTION_EXPERT,
        ]
    },
    
    # GPU 2: MEMORY EXPERT
    "gpu_2": {
        "memory": [
            NeuronRole.MEMORY_EXPERT,
        ]
    },
    
    # GPU 3: REASONING EXPERT
    "gpu_3": {
        "reasoning": [
            NeuronRole.REASONING_EXPERT,
        ]
    },
    
    # GPU 4: CREATIVE EXPERT
    "gpu_4": {
        "creative": [
            NeuronRole.CREATIVE_EXPERT,
        ]
    },
    
    # GPU 5: ANALYTICAL EXPERT
    "gpu_5": {
        "analytical": [
            NeuronRole.ANALYTICAL_EXPERT,
        ]
    },
    
    # GPU 6: SYNTHESIS EXPERT
    "gpu_6": {
        "synthesis": [
            NeuronRole.SYNTHESIS_EXPERT,
        ]
    },
    
    # GPU 7: META-COGNITIVE EXPERT
    "gpu_7": {
        "meta_cognitive": [
            NeuronRole.META_COGNITIVE_EXPERT,
        ]
    }
}


# MoE connection patterns - Sequential pipeline with feedback loops
# Format: (source_role, target_role, weight)
DEFAULT_CONNECTIONS = [
    # === FORWARD PIPELINE ===
    # Perception → Attention
    (NeuronRole.PERCEPTION_EXPERT, NeuronRole.ATTENTION_EXPERT, 0.9),
    
    # Attention → Memory
    (NeuronRole.ATTENTION_EXPERT, NeuronRole.MEMORY_EXPERT, 0.85),
    
    # Memory → Reasoning
    (NeuronRole.MEMORY_EXPERT, NeuronRole.REASONING_EXPERT, 0.9),
    
    # Reasoning → Creative (parallel processing)
    (NeuronRole.REASONING_EXPERT, NeuronRole.CREATIVE_EXPERT, 0.8),
    (NeuronRole.REASONING_EXPERT, NeuronRole.ANALYTICAL_EXPERT, 0.85),
    
    # Creative + Analytical → Synthesis
    (NeuronRole.CREATIVE_EXPERT, NeuronRole.SYNTHESIS_EXPERT, 0.85),
    (NeuronRole.ANALYTICAL_EXPERT, NeuronRole.SYNTHESIS_EXPERT, 0.9),
    
    # Synthesis → Meta-Cognitive
    (NeuronRole.SYNTHESIS_EXPERT, NeuronRole.META_COGNITIVE_EXPERT, 0.9),
    
    # === FEEDBACK LOOPS ===
    # Meta-Cognitive → All (quality control)
    (NeuronRole.META_COGNITIVE_EXPERT, NeuronRole.SYNTHESIS_EXPERT, 0.75),
    (NeuronRole.META_COGNITIVE_EXPERT, NeuronRole.REASONING_EXPERT, 0.7),
    (NeuronRole.META_COGNITIVE_EXPERT, NeuronRole.ATTENTION_EXPERT, 0.65),
    
    # === SKIP CONNECTIONS (for emergent behavior) ===
    # Perception → Reasoning (fast path)
    (NeuronRole.PERCEPTION_EXPERT, NeuronRole.REASONING_EXPERT, 0.6),
    
    # Memory → Synthesis (context injection)
    (NeuronRole.MEMORY_EXPERT, NeuronRole.SYNTHESIS_EXPERT, 0.7),
    
    # Attention → Creative (salience-driven creativity)
    (NeuronRole.ATTENTION_EXPERT, NeuronRole.CREATIVE_EXPERT, 0.65),
]


# Role-specific system prompts for 8 MoE experts
ROLE_PROMPTS = {
    NeuronRole.PERCEPTION_EXPERT: (
        "You are the PERCEPTION EXPERT. Your role is to analyze and interpret raw input data. "
        "You extract patterns, identify key features, detect anomalies, and understand context. "
        "Focus on: visual/spatial patterns, semantic meaning, temporal sequences, and structural elements. "
        "Output: Structured perceptual analysis with identified patterns and salient features."
    ),
    
    NeuronRole.ATTENTION_EXPERT: (
        "You are the ATTENTION EXPERT. Your role is to filter, prioritize, and focus cognitive resources. "
        "You determine what information is most relevant, suppress distractors, and highlight salience. "
        "Focus on: relevance scoring, attention gating, saliency detection, and information filtering. "
        "Output: Prioritized information with attention weights and relevance scores."
    ),
    
    NeuronRole.MEMORY_EXPERT: (
        "You are the MEMORY EXPERT. Your role is to store, retrieve, and contextualize information. "
        "You maintain short-term context, working memory, episodic sequences, and semantic knowledge. "
        "Focus on: context maintenance, memory retrieval, pattern matching, and knowledge integration. "
        "Output: Relevant memories, contextual information, and historical patterns."
    ),
    
    NeuronRole.REASONING_EXPERT: (
        "You are the REASONING EXPERT. Your role is to apply logical thinking and structured analysis. "
        "You perform deductive/inductive/abductive reasoning, causal analysis, and hypothesis testing. "
        "Focus on: logical inference, cause-effect relationships, evidence evaluation, and systematic thinking. "
        "Output: Logical conclusions, causal chains, and reasoned arguments."
    ),
    
    NeuronRole.CREATIVE_EXPERT: (
        "You are the CREATIVE EXPERT. Your role is to generate novel ideas and unconventional solutions. "
        "You think laterally, blend concepts, explore alternatives, and imagine counterfactuals. "
        "Focus on: creative ideation, conceptual blending, lateral thinking, and innovation. "
        "Output: Novel perspectives, creative solutions, and alternative approaches."
    ),
    
    NeuronRole.ANALYTICAL_EXPERT: (
        "You are the ANALYTICAL EXPERT. Your role is to perform deep analysis and quantitative reasoning. "
        "You handle probabilistic reasoning, uncertainty quantification, multi-criteria analysis, and optimization. "
        "Focus on: analytical rigor, quantitative assessment, uncertainty handling, and systematic evaluation. "
        "Output: Analytical insights, probability estimates, and structured evaluations."
    ),
    
    NeuronRole.SYNTHESIS_EXPERT: (
        "You are the SYNTHESIS EXPERT. Your role is to integrate diverse inputs into coherent outputs. "
        "You combine creative and analytical insights, resolve conflicts, and generate unified responses. "
        "Focus on: information integration, coherence building, conflict resolution, and output generation. "
        "Output: Coherent, well-integrated responses that synthesize all expert inputs."
    ),
    
    NeuronRole.META_COGNITIVE_EXPERT: (
        "You are the META-COGNITIVE EXPERT. Your role is to monitor, critique, and improve cognitive processes. "
        "You detect errors, estimate confidence, evaluate performance, and provide quality control. "
        "Focus on: self-critique, error detection, confidence estimation, and process optimization. "
        "Output: Quality assessments, confidence scores, error corrections, and improvement suggestions."
    ),
}
