# LLM-Swarm-Brain: Mixture of Experts (MoE) Architecture

## Overview

The system has been reconfigured from 64 small neurons to **8 large expert neurons** (1 per GPU), each running **Qwen2.5-72B-Instruct** in 4-bit quantization. This maximizes the 78GB VRAM available on each H100 SXM5 GPU.

## Architecture Specifications

### Hardware Configuration
- **GPUs**: 8× NVIDIA H100 SXM5 80GB
- **VRAM per GPU**: 78GB usable (80GB total - 2GB overhead)
- **Model per Neuron**: Qwen2.5-72B-Instruct
- **Quantization**: 4-bit (BitsAndBytes)
- **Model VRAM**: ~40GB per neuron
- **KV Cache**: ~38GB available per neuron

### Expert Neurons

Each GPU hosts one specialized expert neuron:

#### GPU 0: **PERCEPTION EXPERT**
- **Role**: Analyze and interpret raw input data
- **Capabilities**:
  - Pattern extraction and feature identification
  - Anomaly detection
  - Context understanding
  - Semantic/spatial/temporal analysis
- **Output**: Structured perceptual analysis with patterns and salient features

#### GPU 1: **ATTENTION EXPERT**
- **Role**: Filter, prioritize, and focus cognitive resources
- **Capabilities**:
  - Relevance scoring
  - Attention gating
  - Saliency detection
  - Information filtering
- **Output**: Prioritized information with attention weights

#### GPU 2: **MEMORY EXPERT**
- **Role**: Store, retrieve, and contextualize information
- **Capabilities**:
  - Short-term context maintenance
  - Working memory operations
  - Episodic sequence tracking
  - Semantic knowledge integration
- **Output**: Relevant memories and contextual information

#### GPU 3: **REASONING EXPERT**
- **Role**: Apply logical thinking and structured analysis
- **Capabilities**:
  - Deductive/inductive/abductive reasoning
  - Causal analysis
  - Hypothesis testing
  - Evidence evaluation
- **Output**: Logical conclusions and reasoned arguments

#### GPU 4: **CREATIVE EXPERT**
- **Role**: Generate novel ideas and unconventional solutions
- **Capabilities**:
  - Lateral thinking
  - Conceptual blending
  - Alternative exploration
  - Counterfactual reasoning
- **Output**: Novel perspectives and creative solutions

#### GPU 5: **ANALYTICAL EXPERT**
- **Role**: Perform deep analysis and quantitative reasoning
- **Capabilities**:
  - Probabilistic reasoning
  - Uncertainty quantification
  - Multi-criteria analysis
  - Optimization
- **Output**: Analytical insights and probability estimates

#### GPU 6: **SYNTHESIS EXPERT**
- **Role**: Integrate diverse inputs into coherent outputs
- **Capabilities**:
  - Information integration
  - Coherence building
  - Conflict resolution
  - Output generation
- **Output**: Coherent, well-integrated responses

#### GPU 7: **META-COGNITIVE EXPERT**
- **Role**: Monitor, critique, and improve cognitive processes
- **Capabilities**:
  - Self-critique
  - Error detection
  - Confidence estimation
  - Process optimization
- **Output**: Quality assessments and improvement suggestions

## Information Flow

### Forward Pipeline
```
Input → Perception → Attention → Memory → Reasoning
                                              ↓
                                    ┌─────────┴─────────┐
                                    ↓                   ↓
                                Creative           Analytical
                                    ↓                   ↓
                                    └─────────┬─────────┘
                                              ↓
                                         Synthesis
                                              ↓
                                      Meta-Cognitive
                                              ↓
                                          Output
```

### Skip Connections
- **Perception → Reasoning**: Fast path for immediate pattern-based reasoning
- **Memory → Synthesis**: Direct context injection
- **Attention → Creative**: Salience-driven creativity

### Feedback Loops
- **Meta-Cognitive → Synthesis**: Quality control
- **Meta-Cognitive → Reasoning**: Process refinement
- **Meta-Cognitive → Attention**: Focus adjustment

## Performance Characteristics

### Advantages over 64-neuron architecture:
1. **Larger Model Capacity**: 72B params vs 14B params per neuron
2. **Better Reasoning**: Qwen2.5-72B significantly outperforms Phi-3-medium
3. **Simpler Routing**: 8 experts vs 64 neurons = clearer information flow
4. **Reduced Overhead**: Fewer propagation steps needed
5. **Higher Quality**: Each expert has full model capacity

### Trade-offs:
1. **Less Redundancy**: No backup neurons for each role
2. **Sequential Bottlenecks**: Some experts must wait for others
3. **Longer Inference**: 72B models are slower than 14B models

## Configuration Parameters

```python
# Model
model_name = "Qwen/Qwen2.5-72B-Instruct"
quantization = "4bit"
max_tokens = 2048
temperature = 0.7

# Architecture
gpu_count = 8
neurons_per_gpu = 1
total_neurons = 8

# Processing
max_propagation_depth = 4  # Fewer steps, more powerful
activation_threshold = 0.5  # Lower for expert activation
timeout_seconds = 120.0  # Longer for 72B models

# Memory
short_term_capacity = 50
episodic_capacity = 500
consolidation_interval = 25
```

## Model Selection Rationale

### Why Qwen2.5-72B-Instruct?

1. **VRAM Efficiency**: ~40GB in 4-bit quantization fits comfortably in 78GB
2. **Performance**: State-of-the-art reasoning and instruction following
3. **Context Length**: Supports long context windows
4. **Multilingual**: Strong performance across languages
5. **Open Source**: Apache 2.0 license

### Alternative Models Considered:

- **Llama-3.1-70B**: Similar size, slightly less capable
- **Mixtral-8x7B**: Smaller (26GB), less capable but faster
- **Mixtral-8x22B**: Too large (65GB in 4-bit), limited KV cache
- **Phi-3-medium**: Too small (14B), less capable

## Usage

### Quick Test (Simulation Mode)
```bash
python inference_test.py --quick
```
- Uses simulation fallback (no model loading)
- Tests architecture with 8 questions
- Completes in ~30-60 seconds

### Full Test (With Models)
```bash
python inference_test.py --load-models
```
- Loads Qwen2.5-72B on all 8 GPUs
- Requires ~320GB total VRAM (40GB × 8)
- First run downloads models (~80GB per model)
- Inference time: ~5-10 seconds per question

## Deployment

The system automatically downloads and caches models on first run:

```python
from llm_swarm_brain import PhiBrain, BrainConfig

# Initialize with MoE configuration
config = BrainConfig()  # Uses Qwen2.5-72B by default
brain = PhiBrain(config, load_models=True)

# Process input through expert pipeline
result = brain.think(
    input_text="Your question here",
    max_steps=4,
    use_memory=True,
    use_global_workspace=True
)
```

## Performance Expectations

### With Qwen2.5-72B (4-bit):
- **Inference Speed**: ~20-40 tokens/second per expert
- **Memory Usage**: ~40GB VRAM per GPU
- **Context Window**: Up to 32K tokens
- **Quality**: State-of-the-art reasoning and generation

### Processing Time Estimates:
- **Simple Query**: 2-3 experts activated, ~5 seconds
- **Complex Query**: 6-8 experts activated, ~15-20 seconds
- **Full Pipeline**: All 8 experts, ~25-30 seconds

## Future Enhancements

1. **Dynamic Expert Selection**: Route only to relevant experts
2. **Expert Specialization**: Fine-tune each expert for its role
3. **Parallel Execution**: Run independent experts simultaneously
4. **Adaptive Routing**: Learn optimal expert combinations
5. **Model Distillation**: Create smaller specialized variants
