"""
LLM-Swarm-Brain: Neural Network of LLM Neurons

A cognitive architecture using Phi-3-mini models as neurons,
implementing Global Workspace Theory, Integrated Information Theory,
and Positronic Dialectical Logic-Gated Coherence Framework.
"""

__version__ = "0.1.0"
__author__ = "LLM-Swarm-Brain Contributors"

from llm_swarm_brain.brain import PhiBrain
from llm_swarm_brain.neuron import Phi3Neuron, NeuronSignal
from llm_swarm_brain.orchestrator import NeuralOrchestrator
from llm_swarm_brain.gw_theory import GlobalWorkspace, ConsciousnessMonitor
from llm_swarm_brain.positronic_framework import PositronicFramework, LogicGate
from llm_swarm_brain.config import BrainConfig, NeuronRole

__all__ = [
    "PhiBrain",
    "Phi3Neuron",
    "NeuronSignal",
    "NeuralOrchestrator",
    "GlobalWorkspace",
    "ConsciousnessMonitor",
    "PositronicFramework",
    "LogicGate",
    "BrainConfig",
    "NeuronRole",
]
