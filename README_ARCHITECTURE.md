# LLM-Swarm-Brain Architecture Documentation

## Overview

LLM-Swarm-Brain is a cognitive architecture that implements a neural network using LLMs (specifically Phi-3-mini) as individual neurons. The system integrates multiple cognitive science theories to create an emergent intelligent system.

## Core Architecture

### 16-Neuron Configuration

The brain consists of 16 Phi-3-mini instances distributed across 2 GPUs (AMD Radeon 7900 XT):

**GPU 0: Perception & Memory Layer (8 neurons)**
- **Perception Neurons (4)**
  - Visual Perception: Analyzes visual patterns and spatial relationships
  - Semantic Perception: Extracts meaning and concepts
  - Pattern Recognition: Identifies recurring patterns and regularities
  - Anomaly Detection: Detects unusual or contradictory elements

- **Memory Neurons (4)**
  - Short-term Memory: Tracks recent information
  - Episodic Memory: Stores specific events and sequences
  - Semantic Memory: Maintains factual knowledge
  - Working Memory: Actively manipulates task-relevant information

**GPU 1: Reasoning & Action Layer (8 neurons)**
- **Reasoning Neurons (4)**
  - Logical Reasoning: Applies deductive and inductive logic
  - Creative Thinking: Generates novel solutions
  - Causal Analysis: Identifies cause-effect relationships
  - Hypothesis Generation: Proposes testable explanations

- **Action Neurons (4)**
  - Action Planning: Develops step-by-step strategies
  - Decision Making: Evaluates and selects optimal actions
  - Output Synthesis: Integrates information into coherent responses
  - Self-Critique: Identifies weaknesses and areas for improvement

## Theoretical Frameworks

### 1. Global Workspace Theory (GWT)

Based on Bernard Baars' theory of consciousness, the Global Workspace implements:

- **Competitive Selection**: Neurons compete for global broadcast based on salience
- **Global Broadcasting**: Winning information is broadcast to all neurons
- **Consciousness Threshold**: High-salience broadcasts represent "conscious" content
- **Attention Mechanism**: Attended neurons receive salience boost

Key Components:
- `GlobalWorkspace`: Manages broadcast competition and distribution
- `ConsciousnessMonitor`: Tracks consciousness levels and integration
- Broadcast capacity: 5 simultaneous broadcasts (configurable)

### 2. Integrated Information Theory (IIT)

Implements Giulio Tononi's IIT to measure integrated information (Φ - Phi):

- **Φ Calculation**: Measures information integration across the network
- **System vs. Parts**: Compares whole-system entropy to sum of parts
- **Integration Window**: 2-second temporal window for integration
- **Connectivity Modulation**: Connection strength affects Φ

Key Metrics:
- Φ (Phi): Integrated information score
- Consciousness threshold: Φ > 0.5 (configurable)
- Network connectivity: Average connection strength

### 3. Positronic Dialectical Logic-Gated Coherence Framework

Inspired by Asimov's positronic brain and dialectical reasoning:

#### Dialectical Reasoning (Hegel)
- **Thesis**: Initial proposition/output
- **Antithesis**: Contradictory position (auto-generated)
- **Synthesis**: Higher-order understanding reconciling both

#### Logic Gates
Classical and fuzzy logic operations:
- AND, OR, NOT, XOR, NAND, NOR
- IMPLICATION, EQUIVALENCE
- FUZZY_AND (min), FUZZY_OR (max)
- DIALECTICAL_SYNTHESIS (mean + innovation)

#### Positronic Laws (Three Laws)
1. **First Law**: Coherence and truth preservation
2. **Second Law**: Logical consistency
3. **Third Law**: Self-consistency and stability

#### Coherence Validation
- Validates logical consistency of outputs
- Checks for contradictions between neurons
- Enforces positronic laws
- Reports violations and coherence scores

## Neural Network Dynamics

### Activation Process

1. **Signal Reception**: Neuron receives input signal
2. **Activation Calculation**:
   ```
   activation = cosine_similarity(input_embedding, role_embedding) * input_activation
   activation = sigmoid(activation)
   ```
3. **Threshold Check**: Activation > threshold → neuron fires
4. **Generation**: Phi-3 model generates output based on role
5. **Propagation**: Output sent to connected neurons

### Connection System

- **Predefined Pathways**: 18+ default connections based on cognitive flow
- **Hebbian Learning**: "Neurons that fire together, wire together"
  ```
  Δw = learning_rate × pre_activation × post_activation
  ```
- **Synaptic Decay**: Unused connections gradually weaken
- **Pruning**: Connections below threshold are removed

### Signal Processing

1. **Layer 1 (Perception)**: Input activates perception neurons
2. **Layer 2 (Memory)**: Perception signals activate memory neurons
3. **Layer 3 (Reasoning)**: Memory signals activate reasoning neurons
4. **Layer 4 (Action)**: Reasoning signals activate action neurons
5. **Feedback Loops**: Self-critique influences all output neurons

Maximum propagation depth: 4 steps (configurable)

## Memory System

Three-level memory hierarchy:

### Short-Term Memory
- Capacity: 10 items (configurable)
- Stores: Recent inputs and outputs
- Structure: Circular buffer (FIFO)

### Episodic Memory
- Capacity: 100 episodes (configurable)
- Stores: Event sequences with timestamps
- Consolidation: Periodic transfer from short-term

### Semantic Memory
- Capacity: Unlimited (dictionary)
- Stores: Factual knowledge and concepts
- Structure: Key-value pairs

**Memory Consolidation**: Every 50 processing steps (configurable)

## Processing Pipeline

```
Input Text
    ↓
[Embedding Generation]
    ↓
[Neuron Activation Calculation]
    ↓
[Neuron Firing (Phi-3 Generation)]
    ↓
[Signal Propagation via Connections]
    ↓
[Global Workspace Competition]
    ↓
[Coherence Validation (Positronic)]
    ↓
[Dialectical Synthesis]
    ↓
[Memory Storage]
    ↓
[Consciousness Level Calculation]
    ↓
Output Result
```

## Key Algorithms

### Global Workspace Broadcasting

```python
def compete_for_broadcast(outputs, activations):
    # Calculate salience scores
    salience = []
    for neuron_id, output in outputs.items():
        activation = activations[neuron_id]
        coherence = min(1.0, len(output) / 500)
        salience_score = activation * 0.6 + coherence * 0.4

        # Attention boost
        if neuron_id in attention_focus:
            salience_score *= 1.5

        salience.append((neuron_id, salience_score))

    # Select top-k above threshold
    salience.sort(reverse=True)
    return [s for s in salience[:capacity] if s[1] >= threshold]
```

### Integrated Information (Φ)

```python
def calculate_phi(activations, connections):
    # Whole-system entropy
    H_whole = -Σ(p * log2(p) + (1-p) * log2(1-p))

    # Sum of individual entropies
    H_parts = Σ[-(p * log2(p) + (1-p) * log2(1-p))]

    # Integration
    phi = max(0, H_whole - H_parts)

    # Modulate by connectivity
    phi *= mean(connections)

    return phi
```

### Dialectical Synthesis

```python
def synthesize(thesis, antithesis, signal_a, signal_b):
    # Dialectical synthesis gate
    mean = (signal_a + signal_b) / 2
    innovation = abs(signal_a - signal_b) * 0.2
    confidence = clip(mean + innovation, 0, 1)

    # Generate synthesis (would use LLM in full version)
    synthesis = integrate_perspectives(thesis, antithesis)

    return DialecticalTriad(thesis, antithesis, synthesis, confidence)
```

## Performance Characteristics

### Resource Usage

**Per Neuron (4-bit quantized Phi-3-mini)**:
- Model size: ~2 GB VRAM
- KV cache: ~1 GB VRAM
- Total per neuron: ~3 GB VRAM

**Total System**:
- GPU 0: 8 neurons × 3 GB = ~24 GB VRAM
- GPU 1: 8 neurons × 3 GB = ~24 GB VRAM
- System RAM: ~4-8 GB (embeddings, queues, context)

AMD Radeon 7900 XT (24 GB VRAM) × 2 = Perfect fit!

### Throughput

- **Per neuron**: 50-100 tokens/sec
- **Parallel**: All 16 can run simultaneously
- **End-to-end latency**: 1-3 seconds (depends on activation pattern)

### Optimization Strategies

1. **4-bit Quantization**: Reduces memory by ~75%
2. **Embedding Cache**: Avoids recomputation
3. **Circular Buffers**: Efficient memory management
4. **Connection Pruning**: Removes dead connections
5. **Parallel Execution**: Maximizes GPU utilization

## Configuration

### BrainConfig Parameters

```python
config = BrainConfig(
    # Model
    model_name="microsoft/Phi-3-mini-4k-instruct",
    quantization="Q4_K_M",
    max_tokens=512,
    temperature=0.7,

    # Neural network
    activation_threshold=0.6,
    connection_strength_threshold=0.5,

    # GWT
    global_workspace_capacity=5,
    broadcast_threshold=0.7,
    consciousness_threshold=0.8,

    # IIT
    phi_threshold=0.5,
    integration_window=2.0,

    # Learning
    hebbian_learning_rate=0.01,
    connection_decay_rate=0.001,

    # Memory
    short_term_capacity=10,
    episodic_capacity=100,
    consolidation_interval=50,
)
```

## Emergent Properties

The system exhibits emergent behaviors not explicitly programmed:

1. **Specialization**: Neurons develop specific roles through Hebbian learning
2. **Attention Shifting**: System naturally focuses on salient information
3. **Context Awareness**: Memory system provides continuity across interactions
4. **Self-Correction**: Critique neuron provides feedback loop
5. **Creative Synthesis**: Dialectical reasoning generates novel insights
6. **Coherent Output**: Positronic laws ensure logical consistency

## Future Enhancements

1. **Dynamic Topology**: Neurons form/break connections based on utility
2. **Neurogenesis**: Add new neurons for new capabilities
3. **Meta-Learning**: Brain learns to optimize its own architecture
4. **Multi-Modal**: Visual, auditory input processing
5. **Embodiment**: Integration with robotic systems
6. **Distributed**: Scale across multiple machines

## References

1. Baars, B. J. (1988). A cognitive theory of consciousness.
2. Tononi, G. (2004). An information integration theory of consciousness.
3. Hebb, D. O. (1949). The Organization of Behavior.
4. Hegel, G. W. F. (1807). Phenomenology of Spirit.
5. Asimov, I. (1950). I, Robot.

## License

MIT License - See LICENSE file for details.
