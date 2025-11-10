"""
Configuration for Local Training on AMD Radeon RX 7900 XT (20GB VRAM)

This configuration uses Phi-4 models via LM Studio
for efficient local training with enhanced reasoning capabilities.

Hardware: AMD Radeon RX 7900 XT
- VRAM: 20 GB GDDR6
- Architecture: RDNA 3
- Compute: ~51 TFLOPS FP32

Model: microsoft/Phi-4
- Parameters: 14B
- Quantization: Q4_K_M (~8GB per instance)
- Context: 16384 tokens
- Enhanced reasoning and problem-solving capabilities
"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class NeuronRole(Enum):
    """8 specialized neuron roles for local training"""
    
    # Cognitive processing pipeline
    PERCEPTION = "perception"
    ATTENTION = "attention"
    MEMORY = "memory"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    SYNTHESIS = "synthesis"
    META_COGNITIVE = "meta_cognitive"


@dataclass
class LocalBrainConfig:
    """Configuration for local 7900XT training with Phi-4"""
    
    # Model configuration
    model_name: str = "microsoft/Phi-4"
    quantization: str = "Q4_K_M"  # ~8GB per model
    max_tokens: int = 1024  # Phi-4 can handle longer contexts
    temperature: float = 0.7
    
    # LM Studio configuration
    lm_studio_url: str = "http://localhost:1234/v1"
    use_lm_studio: bool = True
    
    # GPU allocation (dual 7900XT setup)
    # Phi-4 is 14B params, Q4 = ~8GB per instance
    # GPU 0: 2 neurons (16GB used)
    # GPU 1: 2 neurons (16GB used)
    # Total: 4 neurons across 2 GPUs
    total_neurons: int = 4
    neurons_per_gpu: int = 2
    vram_per_neuron: float = 8.0  # GB (Q4 quantization)
    total_vram_budget: float = 40.0  # GB (2x 20GB GPUs)
    
    # Neural network parameters
    activation_threshold: float = 0.5
    connection_strength_threshold: float = 0.4
    
    # GWT parameters (optimized for local)
    global_workspace_capacity: int = 3  # Smaller for faster processing
    broadcast_threshold: float = 0.65
    consciousness_threshold: float = 0.75
    
    # IIT parameters
    phi_threshold: float = 0.6
    integration_window: float = 2.0
    
    # Learning parameters
    hebbian_learning_rate: float = 0.015
    connection_decay_rate: float = 0.0005
    
    # Memory parameters (optimized for local)
    short_term_capacity: int = 30
    episodic_capacity: int = 200
    consolidation_interval: int = 15
    
    # Performance parameters (optimized for local)
    parallel_execution: bool = False  # Sequential for stability
    max_propagation_depth: int = 2  # Fewer steps for speed
    timeout_seconds: float = 60.0


# 4-neuron architecture for dual 7900XT (Phi-4 14B model)
# GPU 0: 2 neurons, GPU 1: 2 neurons
# Each neuron handles 2 cognitive functions for balanced processing
NEURON_ARCHITECTURE = {
    # GPU 0 - Neuron 0: Perception & Reasoning
    "neuron_0": {
        "role": [NeuronRole.PERCEPTION, NeuronRole.REASONING],
        "gpu": 0
    },
    # GPU 0 - Neuron 1: Attention & Memory
    "neuron_1": {
        "role": [NeuronRole.ATTENTION, NeuronRole.MEMORY],
        "gpu": 0
    },
    # GPU 1 - Neuron 2: Creative & Analytical
    "neuron_2": {
        "role": [NeuronRole.CREATIVE, NeuronRole.ANALYTICAL],
        "gpu": 1
    },
    # GPU 1 - Neuron 3: Synthesis & Meta-Cognitive
    "neuron_3": {
        "role": [NeuronRole.SYNTHESIS, NeuronRole.META_COGNITIVE],
        "gpu": 1
    },
}


# Connection pattern for 4-neuron dual-GPU Phi-4 architecture
DEFAULT_CONNECTIONS = [
    # Forward pipeline
    (NeuronRole.PERCEPTION, NeuronRole.ATTENTION, 0.9),
    (NeuronRole.ATTENTION, NeuronRole.MEMORY, 0.85),
    (NeuronRole.MEMORY, NeuronRole.REASONING, 0.9),
    (NeuronRole.REASONING, NeuronRole.CREATIVE, 0.85),
    (NeuronRole.REASONING, NeuronRole.ANALYTICAL, 0.85),
    (NeuronRole.CREATIVE, NeuronRole.SYNTHESIS, 0.9),
    (NeuronRole.ANALYTICAL, NeuronRole.SYNTHESIS, 0.9),
    (NeuronRole.SYNTHESIS, NeuronRole.META_COGNITIVE, 0.95),
    
    # Cross-GPU feedback loops
    (NeuronRole.META_COGNITIVE, NeuronRole.PERCEPTION, 0.75),
    (NeuronRole.META_COGNITIVE, NeuronRole.ATTENTION, 0.70),
    (NeuronRole.SYNTHESIS, NeuronRole.MEMORY, 0.65),
]


# Role-specific prompts (same as cloud config)
ROLE_PROMPTS = {
    NeuronRole.PERCEPTION: "You are a perception expert. Analyze and extract key information from the input.",
    NeuronRole.ATTENTION: "You are an attention expert. Focus on the most relevant aspects of the information.",
    NeuronRole.MEMORY: "You are a memory expert. Recall relevant context and integrate it with current information.",
    NeuronRole.REASONING: "You are a reasoning expert. Apply logical analysis and draw inferences.",
    NeuronRole.CREATIVE: "You are a creative expert. Generate novel perspectives and connections.",
    NeuronRole.ANALYTICAL: "You are an analytical expert. Break down complex ideas systematically.",
    NeuronRole.SYNTHESIS: "You are a synthesis expert. Integrate multiple perspectives into coherent insights.",
    NeuronRole.META_COGNITIVE: "You are a meta-cognitive expert. Monitor and optimize the reasoning process.",
}
