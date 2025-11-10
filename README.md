# LLM-Swarm-Brain

> A neural network architecture using LLMs as individual neurons

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

LLM-Swarm-Brain implements a cognitive architecture where **individual LLMs act as neurons** in a neural network. Using 64 Phi-3-medium models distributed across 8 H100 GPUs, it creates an emergent intelligent system aligned with cognitive science theories.

### Key Features

🧠 **64 Specialized Phi-3 Neurons**
- Distributed across 8× NVIDIA H100 SXM5 GPUs (80GB each)
- 4× scale-up from original architecture
- Each neuron has a specific cognitive role
- Uses Phi-3-Medium model (upgraded from Mini)
- 4-bit quantization for memory efficiency

🌐 **Global Workspace Theory (GWT)**
- Competitive selection for conscious processing
- Global broadcasting of salient information
- Attention mechanisms and consciousness monitoring

🔗 **Integrated Information Theory (IIT)**
- Measures integrated information (Φ - Phi)
- Tracks consciousness levels
- Network integration metrics

🤖 **Positronic Dialectical Framework**
- Dialectical reasoning (thesis-antithesis-synthesis)
- Logic gates for signal processing
- Coherence validation and positronic laws

🧬 **Neural Dynamics**
- Hebbian learning ("neurons that fire together, wire together")
- Synaptic pruning and connection decay
- Emergent specialization and behavior

💾 **Multi-Level Memory**
- Short-term, episodic, and semantic memory
- Memory consolidation
- Context-aware processing

⚡ **Advanced Enhancements (NEW!)**
- **Summarization Neuron**: Compresses verbose outputs to prevent information explosion
- **Attention Windowing**: Selective broadcasting to relevant neurons only
- **Conceptual Thread Tracking**: Tracks how concepts flow and transform through the network
- **Meta-Orchestration**: Dynamically adjusts activation thresholds based on task complexity
- **Coherence Check-ins**: Explicit prompts asking "Does this contradict anything?"

## Architecture (8× H100 SXM5 80GB)

```
┌─────────────────────────────────────────────────────────────┐
│               LLM-SWARM-BRAIN (64 NEURONS)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GPU 0-1: PERCEPTION LAYER (16 neurons)                     │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │  Specialized     │  │   Sensory        │               │
│  │  Perception (8)  │  │ Integration (8)  │               │
│  │  • Visual        │  │ • Multi-sensory  │               │
│  │  • Auditory      │  │ • Feature Bind   │               │
│  │  • Semantic      │  │ • Grouping       │               │
│  │  • Spatial       │  │ • Attention      │               │
│  │  • Temporal      │  │ • Saliency       │               │
│  │  • Pattern       │  │ • Prediction     │               │
│  │  • Anomaly       │  │ • Gating         │               │
│  │  • Context       │  │ • Coherence      │               │
│  └────────┬─────────┘  └────────┬─────────┘               │
│           │                     │                          │
│           └──────────┬──────────┘                          │
│                      ▼                                     │
│  GPU 2-3: MEMORY LAYER (16 neurons, 4× redundancy)         │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ Short-term  (4)  │  │ Working     (4)  │               │
│  │ Episodic    (4)  │  │ Semantic    (4)  │               │
│  └────────┬─────────┘  └────────┬─────────┘               │
│           │                     │                          │
│           └──────────┬──────────┘                          │
│                      ▼                                     │
│  ┌─────────────────────────────────┐                       │
│  │   GLOBAL WORKSPACE (GWT)        │                       │
│  │   • 10 Broadcast Slots          │                       │
│  │   • Consciousness Monitoring    │                       │
│  │   • Attention Windowing         │                       │
│  └──────────────┬──────────────────┘                       │
│                 ▼                                          │
│  ┌─────────────────────────────────┐                       │
│  │ POSITRONIC FRAMEWORK            │                       │
│  │ • Dialectical Reasoning         │                       │
│  │ • Coherence Check-ins           │                       │
│  │ • Summarization                 │                       │
│  └──────────────┬──────────────────┘                       │
│                 ▼                                          │
│  GPU 4-5: REASONING LAYER (16 neurons)                      │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │  Core (8)        │  │  Advanced (8)    │               │
│  │  • Logical       │  │ • Analogical     │               │
│  │  • Deductive     │  │ • Probabilistic  │               │
│  │  • Inductive     │  │ • Hypothesis Gen │               │
│  │  • Abductive     │  │ • Hypothesis Test│               │
│  │  • Creative      │  │ • Inference      │               │
│  │  • Lateral       │  │ • Belief Update  │               │
│  │  • Causal        │  │ • Uncertainty    │               │
│  │  • Counterfactual│  │ • Concept Blend  │               │
│  └────────┬─────────┘  └────────┬─────────┘               │
│           │                     │                          │
│           └──────────┬──────────┘                          │
│                      ▼                                     │
│  GPU 6-7: ACTION/META LAYER (16 neurons)                    │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │  Action/Dec (8)  │  │  Meta-Cog (8)    │               │
│  │  • Action Plan   │  │ • Self-Critique  │               │
│  │  • Hierarchical  │  │ • Error Detect   │               │
│  │  • Decision      │  │ • Error Correct  │               │
│  │  • Multi-Crit    │  │ • Confidence     │               │
│  │  • Synthesis     │  │ • Meta-Learning  │               │
│  │  • Response Gen  │  │ • Strategy Sel   │               │
│  │  • Goal Mgmt     │  │ • Perf Monitor   │               │
│  │  • Exec Monitor  │  │ • Cog Control    │               │
│  └──────────────────┘  └──────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

- Python 3.9+
- **8× NVIDIA H100 SXM5 GPUs (80 GB VRAM each)** or equivalent
- CUDA 12.0+ support
- ~100 GB disk space (for Phi-3-Medium models × 64)

### Setup

```bash
# Clone the repository
git clone https://github.com/zoadrazorro/LLM-Swarm-Brain.git
cd LLM-Swarm-Brain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
from llm_swarm_brain import PhiBrain, BrainConfig

# Create configuration
config = BrainConfig(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    max_tokens=512,
    temperature=0.7,
    activation_threshold=0.6
)

# Initialize the brain (loads all 16 Phi-3 neurons)
brain = PhiBrain(config=config, load_models=True)

# Process input
result = brain.think(
    "What is the nature of consciousness?",
    use_memory=True,
    use_global_workspace=True
)

# View results
print(f"Consciousness Level: {result['consciousness_level']:.3f}")
print(f"\nGlobal Workspace Broadcasts:")
for broadcast in result['global_workspace']['broadcasts']:
    print(f"  [{broadcast['source']}] {broadcast['content'][:100]}...")

# View brain state
print(brain.get_summary())
print(brain.visualize_state())
```

### Testing Without Models

For development/testing without loading full models:

```python
# Initialize without loading models
brain = PhiBrain(load_models=False)

# Brain structure is created, but models aren't loaded
# Useful for testing architecture and connections
print(brain.get_summary())
```

## Examples

See the `examples/` directory for more examples:

- `basic_usage.py` - Simple usage example
- `dialectical_reasoning.py` - Positronic framework demo
- `memory_consolidation.py` - Memory system demo
- `consciousness_monitoring.py` - GWT/IIT metrics

Run an example:

```bash
python examples/basic_usage.py
```

## Key Concepts

### Neurons

Each neuron is a Phi-3-mini instance with a specific role:

```python
from llm_swarm_brain import Phi3Neuron, NeuronRole

neuron = Phi3Neuron(
    role=NeuronRole.LOGICAL_REASONING,
    gpu_id=1,
    neuron_id="reasoning_1",
    activation_threshold=0.6
)
```

### Global Workspace

Implements conscious information broadcasting:

```python
from llm_swarm_brain import GlobalWorkspace

workspace = GlobalWorkspace(
    capacity=5,                    # Max simultaneous broadcasts
    broadcast_threshold=0.7,       # Min salience to broadcast
    consciousness_threshold=0.8    # Threshold for conscious processing
)
```

### Positronic Framework

Dialectical reasoning and coherence validation:

```python
from llm_swarm_brain import PositronicFramework

positronic = PositronicFramework(
    coherence_threshold=0.7,
    enable_dialectical_reasoning=True,
    enforce_positronic_laws=True
)

# Apply dialectical reasoning
triad = positronic.apply_dialectical_reasoning(
    thesis_output="Consciousness emerges from complexity",
    thesis_activation=0.9
)

print(f"Thesis: {triad.thesis}")
print(f"Antithesis: {triad.antithesis}")
print(f"Synthesis: {triad.synthesis}")
```

## Configuration

### Brain Configuration

```python
from llm_swarm_brain import BrainConfig

config = BrainConfig(
    # Model settings
    model_name="microsoft/Phi-3-mini-4k-instruct",
    quantization="Q4_K_M",
    max_tokens=512,
    temperature=0.7,

    # Neural network
    activation_threshold=0.6,
    connection_strength_threshold=0.5,

    # Global Workspace Theory
    global_workspace_capacity=5,
    broadcast_threshold=0.7,
    consciousness_threshold=0.8,

    # Integrated Information Theory
    phi_threshold=0.5,
    integration_window=2.0,

    # Learning
    hebbian_learning_rate=0.01,
    connection_decay_rate=0.001,

    # Memory
    short_term_capacity=10,
    episodic_capacity=100,
    consolidation_interval=50
)
```

## Performance (8× H100 SXM5 Configuration)

### Resource Usage

- **GPU 0-1** (Perception): 16 neurons × ~4 GB = ~64 GB VRAM
- **GPU 2-3** (Memory): 16 neurons × ~4 GB = ~64 GB VRAM
- **GPU 4-5** (Reasoning): 16 neurons × ~4 GB = ~64 GB VRAM
- **GPU 6-7** (Action/Meta): 16 neurons × ~4 GB = ~64 GB VRAM
- **System RAM**: ~16-32 GB (embeddings, context, queues, conceptual tracking)

### Throughput

- **Per neuron**: 100-200 tokens/sec (Phi-3-Medium on H100)
- **Parallel processing**: All 64 neurons run simultaneously
- **End-to-end latency**: 1-3 seconds (depends on activation pattern and task complexity)
- **Max propagation depth**: 6 layers (up from 4)

## Documentation

- [Architecture Documentation](README_ARCHITECTURE.md) - Detailed technical documentation
- [API Reference](docs/API.md) - Complete API documentation
- [Cognitive Science Background](docs/COGNITIVE_SCIENCE.md) - Theory and references

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black llm_swarm_brain/
flake8 llm_swarm_brain/
```

## Roadmap

- [ ] Dynamic network topology
- [ ] Neurogenesis (add neurons at runtime)
- [ ] Multi-modal processing (vision, audio)
- [ ] Distributed scaling across machines
- [ ] Meta-learning and architecture optimization
- [ ] Integration with robotic systems
- [ ] Web interface for visualization

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{llm_swarm_brain,
  title={LLM-Swarm-Brain: Neural Network of LLM Neurons},
  author={LLM-Swarm-Brain Contributors},
  year={2025},
  url={https://github.com/zoadrazorro/LLM-Swarm-Brain}
}
```

## References

1. Baars, B. J. (1988). A cognitive theory of consciousness. Cambridge University Press.
2. Tononi, G. (2004). An information integration theory of consciousness. BMC Neuroscience.
3. Hebb, D. O. (1949). The Organization of Behavior. Wiley.
4. Hegel, G. W. F. (1807). Phenomenology of Spirit.
5. Asimov, I. (1950). I, Robot. Gnome Press.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by cognitive science theories of consciousness
- Built on Microsoft's Phi-3-mini model
- Community contributions and feedback

---

**Status**: Alpha - Under active development

For questions and support, please open an issue on GitHub.
