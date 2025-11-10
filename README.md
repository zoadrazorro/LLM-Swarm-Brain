# LLM-Swarm-Brain

> A cognitive architecture where individual LLMs act as specialized neurons in a neural network

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![API Mode](https://img.shields.io/badge/API-Hyperbolic-green.svg)](https://hyperbolic.xyz/)

## Overview

LLM-Swarm-Brain implements a **cognitive architecture** where **individual LLMs act as specialized neurons** in a neural network. Each neuron has a unique cognitive function (perception, memory, reasoning, etc.) and communicates through a rich network of weighted connections.

**Two Deployment Modes:**
- **🖥️ Local Mode**: 8 neurons using Qwen2.5-72B on 8× H100 GPUs (8 neurons per GPU)
- **☁️ API Mode**: 8 or 64 neurons using Llama 3.1 405B via Hyperbolic API (no GPU required!)

The system creates emergent intelligent behavior aligned with cognitive science theories including Global Workspace Theory (GWT) and Integrated Information Theory (IIT).

### Key Features

🧠 **Flexible Architecture**
- **8-Neuron MoE**: 8 specialized experts (Perception, Attention, Memory, Reasoning, Creative, Analytical, Synthesis, Meta-Cognitive)
- **64-Neuron Dense**: 64 highly specialized neurons across 8 cognitive layers
- **Local**: Qwen2.5-72B (72B params, 4-bit quantized)
- **API**: Llama 3.1 405B (405B params via Hyperbolic)
- Rich interconnection patterns (~14 connections for 8-neuron, ~656 for 64-neuron)

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

⚡ **Advanced Enhancements**
- **Summarization Neuron**: Compresses verbose outputs to prevent information explosion
- **Attention Windowing**: Selective broadcasting to relevant neurons only
- **Conceptual Thread Tracking**: Tracks how concepts flow and transform through the network
- **Meta-Orchestration**: Dynamically adjusts activation thresholds based on task complexity
- **Coherence Check-ins**: Explicit prompts asking "Does this contradict anything?"

📊 **Philosophy Test Suite**
- **40-Question Test**: 8 complexity levels testing philosophical reasoning
- **100-Question Expanded Test**: 10 levels from foundational concepts to cutting-edge philosophy
- Comprehensive evaluation of reasoning, creativity, synthesis, and meta-cognition

## Architecture

### 8-Neuron MoE (Mixture of Experts)

```
┌────────────────────────────────────────────────────────┐
│         LLM-SWARM-BRAIN (8-NEURON MoE)                 │
│    Qwen2.5-72B (Local) or Llama 3.1 405B (API)        │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────┐      ┌──────────────┐              │
│  │ PERCEPTION   │─────→│  ATTENTION   │              │
│  │   EXPERT     │      │    EXPERT    │              │
│  └──────┬───────┘      └──────┬───────┘              │
│         │                     │                       │
│         └──────────┬──────────┘                       │
│                    ▼                                  │
│            ┌──────────────┐                           │
│            │   MEMORY     │                           │
│            │   EXPERT     │                           │
│            └──────┬───────┘                           │
│                   │                                   │
│                   ▼                                   │
│            ┌──────────────┐                           │
│            │  REASONING   │                           │
│            │   EXPERT     │                           │
│            └──────┬───────┘                           │
│                   │                                   │
│         ┌─────────┴─────────┐                         │
│         ▼                   ▼                         │
│  ┌──────────────┐    ┌──────────────┐                │
│  │  CREATIVE    │    │ ANALYTICAL   │                │
│  │   EXPERT     │    │   EXPERT     │                │
│  └──────┬───────┘    └──────┬───────┘                │
│         │                   │                         │
│         └─────────┬─────────┘                         │
│                   ▼                                   │
│            ┌──────────────┐                           │
│            │  SYNTHESIS   │                           │
│            │   EXPERT     │                           │
│            └──────┬───────┘                           │
│                   │                                   │
│                   ▼                                   │
│            ┌──────────────┐                           │
│            │META-COGNITIVE│                           │
│            │   EXPERT     │                           │
│            └──────────────┘                           │
│                                                        │
│  14 Connections | Global Workspace | Hebbian Learning │
└────────────────────────────────────────────────────────┘
```

### 64-Neuron Dense Architecture

```
┌────────────────────────────────────────────────────────┐
│      LLM-SWARM-BRAIN (64-NEURON DENSE)                 │
│           Llama 3.1 405B via API                       │
├────────────────────────────────────────────────────────┤
│                                                        │
│  LAYER 1: PERCEPTION (8 neurons)                       │
│  Visual | Auditory | Semantic | Pattern | Context     │
│  Temporal | Spatial | Abstract                        │
│                    ↓                                   │
│  LAYER 2: ATTENTION (8 neurons)                        │
│  Selective | Sustained | Divided | Salience           │
│  Relevance | Priority | Focus | Switching             │
│                    ↓                                   │
│  LAYER 3: MEMORY (8 neurons)                           │
│  Short-term | Working | Episodic | Semantic           │
│  Procedural | Associative | Consolidation | Retrieval │
│                    ↓                                   │
│  LAYER 4: REASONING (8 neurons)                        │
│  Deductive | Inductive | Abductive | Analogical       │
│  Causal | Probabilistic | Logical | Counterfactual    │
│                    ↓                                   │
│  LAYER 5: CREATIVE (8 neurons)                         │
│  Divergent | Blending | Metaphor | Novel              │
│  Lateral | Imaginative | Innovation | Constraint      │
│                    ↓                                   │
│  LAYER 6: ANALYTICAL (8 neurons)                       │
│  Quantitative | Qualitative | Comparative | Critical  │
│  Decomposition | Hypothesis | Evidence | Uncertainty  │
│                    ↓                                   │
│  LAYER 7: SYNTHESIS (8 neurons)                        │
│  Integration | Coherence | Conflict | Perspective     │
│  Holistic | Output | Narrative | Solution             │
│                    ↓                                   │
│  LAYER 8: META-COGNITIVE (8 neurons)                   │
│  Monitoring | Error | Confidence | Strategy           │
│  Performance | Control | Awareness | Regulation       │
│                                                        │
│  656 Connections | Rich Feedback Loops | Skip Paths   │
└────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

**For API Mode (Recommended):**
- Python 3.9+
- Internet connection
- Hyperbolic API key ([get one here](https://app.hyperbolic.xyz/))
- ~2 GB disk space

**For Local Mode:**
- Python 3.9+
- **8× NVIDIA H100 SXM5 GPUs (80 GB VRAM each)** or equivalent
- CUDA 12.0+ support
- ~80 GB disk space (for Qwen2.5-72B model)

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

# For API mode, create .env file
cp .env.example .env
# Edit .env and add your Hyperbolic API key:
# HYPERBOLIC_API_KEY=your_key_here
```

## Quick Start

### API Mode (8-Neuron MoE)

```python
from llm_swarm_brain import PhiBrain, BrainConfig

# Initialize with API mode (Llama 3.1 405B)
brain = PhiBrain(
    use_api=True,
    api_key="your_hyperbolic_api_key"  # Or set HYPERBOLIC_API_KEY env var
)

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
```

### API Mode (64-Neuron Dense)

```python
from llm_swarm_brain import PhiBrain

# Initialize with 64-neuron architecture
brain = PhiBrain(
    use_api=True,
    use_64_neurons=True,
    api_key="your_hyperbolic_api_key"
)

# Process complex philosophical question
result = brain.think(
    "If consciousness is substrate-independent, what implications "
    "does this have for personal identity and ethics?",
    max_steps=5,
    use_memory=True,
    use_global_workspace=True
)

print(f"Answer: {result['global_workspace']['conscious_summary']}")
```

### Local Mode (8-Neuron MoE)

```python
from llm_swarm_brain import PhiBrain, BrainConfig

# Initialize with local models (requires 8× H100 GPUs)
config = BrainConfig(
    model_name="Qwen/Qwen2.5-72B-Instruct",
    max_tokens=2048,
    temperature=0.7
)

brain = PhiBrain(config=config, load_models=True)

# Process input
result = brain.think("Explain quantum entanglement")
```

### Philosophy Test Suite

```bash
# Quick test (10 questions, simulation mode)
python expanded_inference_test_philosophy.py --quick

# Full 100-question test with API (8-neuron)
python expanded_inference_test_philosophy.py --use-api

# Full 100-question test with 64-neuron architecture
python expanded_inference_test_philosophy.py --use-api --use-64-neurons

# Sample test (30 questions)
python expanded_inference_test_philosophy.py --use-api --sample
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

## Performance

### API Mode (Recommended)

**8-Neuron MoE:**
- **Cost**: ~$0.002-0.004 per query (~$0.20 for 100 questions)
- **Latency**: ~5-15 seconds per query
- **Model**: Llama 3.1 405B (405 billion parameters)
- **No GPU required**: Runs on any machine with internet

**64-Neuron Dense:**
- **Cost**: ~$0.015-0.030 per query (~$1.50-2.00 for 100 questions)
- **Latency**: ~30-90 seconds per query
- **Model**: Llama 3.1 405B (405 billion parameters)
- **Rich processing**: 656 connections, 8 cognitive layers

### Local Mode

**8-Neuron MoE (8× H100 GPUs):**
- **VRAM**: 8 neurons × ~40 GB = ~320 GB total
- **Model**: Qwen2.5-72B (72 billion parameters, 4-bit quantized)
- **Latency**: ~2-5 seconds per query
- **Throughput**: ~100-200 tokens/sec per neuron
- **Cost**: GPU rental ~$20-40/hour

### Comparison

| Metric | API (8N) | API (64N) | Local (8N) |
|--------|----------|-----------|------------|
| Model Size | 405B | 405B | 72B |
| Cost/Query | $0.003 | $0.02 | $0.01* |
| Latency | 10s | 60s | 3s |
| GPU Required | No | No | Yes (8×) |
| Neurons | 8 | 64 | 8 |
| Connections | 14 | 656 | 14 |

*Amortized GPU rental cost

## Documentation

- **[API Mode Guide](API_MODE.md)** - Complete guide to using Hyperbolic API
- **[MoE Architecture](MOE_ARCHITECTURE.md)** - 8-neuron Mixture of Experts design
- **[Philosophy Test Guide](PHILOSOPHY_TEST.md)** - 40-question test suite documentation
- [Architecture Documentation](README_ARCHITECTURE.md) - Detailed technical documentation
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

- [x] API mode with Llama 3.1 405B
- [x] 64-neuron dense architecture
- [x] Philosophy test suite (100 questions)
- [x] MoE architecture (8 specialized experts)
- [ ] Multi-provider API support (OpenAI, Anthropic, etc.)
- [ ] Async API calls for parallel execution
- [ ] Dynamic network topology
- [ ] Neurogenesis (add neurons at runtime)
- [ ] Multi-modal processing (vision, audio)
- [ ] Web interface for visualization and monitoring
- [ ] Fine-tuning for specialized domains

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
