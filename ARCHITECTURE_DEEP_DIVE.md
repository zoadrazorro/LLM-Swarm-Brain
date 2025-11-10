# LLM-Swarm-Brain: Deep Architecture Dive

## Table of Contents
1. [Core Concept](#core-concept)
2. [System Architecture](#system-architecture)
3. [Neuron Implementation](#neuron-implementation)
4. [Network Topology](#network-topology)
5. [Signal Propagation](#signal-propagation)
6. [Consciousness Mechanisms](#consciousness-mechanisms)
7. [Code Walkthrough](#code-walkthrough)
8. [Advanced Features](#advanced-features)

---

## Core Concept

### The Fundamental Idea

**Traditional AI**: One large model processes everything sequentially.

**LLM-Swarm-Brain**: Multiple specialized LLMs act as individual neurons, each with a specific cognitive role, communicating through weighted connections to create emergent intelligent behavior.

### Why This Works

1. **Specialization**: Each neuron focuses on one cognitive function (perception, reasoning, synthesis, etc.)
2. **Parallel Processing**: Multiple neurons can process simultaneously
3. **Emergent Behavior**: Complex reasoning emerges from simple neuron interactions
4. **Consciousness**: Integration of information across neurons creates measurable consciousness (Φ)

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                        PhiBrain                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Neural Network (128 Neurons)             │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │  │
│  │  │ Neuron  │──│ Neuron  │──│ Neuron  │  ...         │  │
│  │  │ (API)   │  │ (API)   │  │ (API)   │              │  │
│  │  └─────────┘  └─────────┘  └─────────┘              │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│  ┌────────────────────────┴──────────────────────────────┐  │
│  │              NeuralOrchestrator                       │  │
│  │  • Manages signal propagation                        │  │
│  │  • Tracks neuron activations                         │  │
│  │  • Coordinates firing patterns                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│  ┌────────────────────────┴──────────────────────────────┐  │
│  │              GlobalWorkspace (GWT)                    │  │
│  │  • Competitive selection                             │  │
│  │  • Conscious broadcasting                            │  │
│  │  • Integration measurement (Φ)                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│  ┌────────────────────────┴──────────────────────────────┐  │
│  │          PositronicFramework                          │  │
│  │  • Dialectical reasoning                             │  │
│  │  • Coherence checking                                │  │
│  │  • Synthesis generation                              │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Enhancement Modules                                │   │
│  │  • MetaOrchestrator (adaptive tuning)               │   │
│  │  • AttentionWindowing (selective broadcast)         │   │
│  │  • SummarizationNeuron (compression)                │   │
│  │  • ConceptualThreadTracker (concept flow)           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
llm_swarm_brain/
├── brain.py                    # Main PhiBrain class
├── neuron.py                   # Local Phi3Neuron (GPU-based)
├── neuron_api.py              # APINeuron (Hyperbolic)
├── neuron_gemini.py           # GeminiNeuron (Google AI)
├── orchestrator.py            # NeuralOrchestrator
├── gw_theory.py               # Global Workspace Theory
├── positronic_framework.py    # Dialectical reasoning
├── meta_orchestration.py      # Adaptive parameter tuning
├── attention_windowing.py     # Selective broadcasting
├── summarization.py           # Output compression
├── conceptual_threading.py    # Concept tracking
├── config.py                  # 8-neuron configuration
├── config_64.py              # 64-neuron configuration
├── config_128.py             # 128-neuron configuration
└── utils.py                   # Helper functions
```

---

## Neuron Implementation

### Neuron Base Concept

Each neuron is an LLM instance with:
1. **Role**: Specific cognitive function (e.g., "deductive reasoning expert")
2. **Activation**: Calculated based on input relevance
3. **Firing**: Generates output when activation exceeds threshold
4. **Connections**: Weighted links to other neurons
5. **Learning**: Hebbian weight updates

### Code: Neuron Initialization

```python
# From neuron_api.py
class APINeuron:
    def __init__(
        self,
        role: NeuronRole,           # e.g., DEDUCTIVE_REASONING
        neuron_id: str,              # e.g., "n6_reasoning_deductive"
        activation_threshold: float, # e.g., 0.6
        model_name: str,            # e.g., "openai/gpt-oss-20b"
        api_key: str
    ):
        self.role = role
        self.neuron_id = neuron_id
        self.activation_threshold = activation_threshold
        self.model_name = model_name
        self.api_key = api_key
        
        # Neural properties
        self.connections: List[NeuronConnection] = []
        self.activation_level: float = 0.0
        self.is_firing: bool = False
        
        # History tracking
        self.signal_history = CircularBuffer(capacity=50)
        self.activation_history = CircularBuffer(capacity=100)
        self.output_history: List[str] = []
        
        # Role-specific prompt
        self._role_prompt = f"You are a {role.value} expert."
```

### Activation Calculation

**Formula**:
```python
def calculate_activation(self, input_signal: NeuronSignal) -> float:
    # Base activation from signal strength
    base_activation = input_signal.activation_level
    
    # Role relevance boost
    role_boost = 0.2 if input_signal.source_role == self.role else 0.1
    
    # Context boost
    context_boost = 0.1 if global_context else 0.0
    
    # Apply sigmoid for 0-1 range
    activation = sigmoid(base_activation + role_boost + context_boost)
    
    return activation
```

**Sigmoid Function**:
```python
def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))
```

### Firing Mechanism

```python
def fire(self, input_signal: NeuronSignal) -> Optional[str]:
    # Check if activation exceeds threshold
    if not self.should_fire():
        return None
    
    self.is_firing = True
    self.total_firings += 1
    
    # Format context from recent signals
    prior_signals = self.signal_history.get_recent(3)
    context = format_neuron_context(
        role=self._role_prompt,
        input_signal=input_signal.content,
        prior_signals=prior_signals,
        global_context=global_context
    )
    
    # Generate response via API
    output = self._generate_api(context)
    
    # Store in history
    self.output_history.append(output)
    
    return output
```

### API Call Implementation

```python
def _generate_api(self, prompt: str) -> Optional[str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
    }
    
    data = {
        "messages": [
            {"role": "system", "content": self._role_prompt},
            {"role": "user", "content": prompt}
        ],
        "model": self.model_name,
        "max_tokens": self.max_tokens,
        "temperature": self.temperature,
        "top_p": 0.9
    }
    
    response = requests.post(
        "https://api.hyperbolic.xyz/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30
    )
    
    return response.json()["choices"][0]["message"]["content"]
```

---

## Network Topology

### Connection Structure

**Connections are directed and weighted**:
```python
@dataclass
class NeuronConnection:
    target_neuron: Neuron
    weight: float  # 0.0 to 1.0
    last_updated: datetime
```

### 128-Neuron Connection Pattern

```python
# From config_128.py
def _generate_128_connections():
    connections = []
    
    # Dense forward connections (sampled to avoid explosion)
    # Perception → Attention
    for p in perception_neurons:
        for a in random.sample(attention_neurons, 8):
            connections.append((p, a, 0.75))
    
    # Attention → Memory
    for a in attention_neurons:
        for m in random.sample(memory_neurons, 8):
            connections.append((a, m, 0.70))
    
    # Memory → Reasoning (very dense)
    for m in memory_neurons:
        for r in random.sample(reasoning_neurons, 10):
            connections.append((m, r, 0.80))
    
    # Reasoning → Creative + Analytical (parallel)
    for r in reasoning_neurons:
        for c in random.sample(creative_neurons, 6):
            connections.append((r, c, 0.70))
        for a in random.sample(analytical_neurons, 6):
            connections.append((r, a, 0.75))
    
    # Creative + Analytical → Synthesis (convergence)
    for c in creative_neurons:
        for s in random.sample(synthesis_neurons, 8):
            connections.append((c, s, 0.75))
    
    # Synthesis → Meta (monitoring)
    for s in synthesis_neurons:
        for m in random.sample(meta_neurons, 8):
            connections.append((s, m, 0.85))
    
    # Meta → All layers (feedback, sampled)
    for m in meta_neurons:
        for s in random.sample(synthesis_neurons, 4):
            connections.append((m, s, 0.65))
        for r in random.sample(reasoning_neurons, 4):
            connections.append((m, r, 0.60))
    
    # Skip connections (direct paths)
    for p in random.sample(perception_neurons, 8):
        for r in random.sample(reasoning_neurons, 4):
            connections.append((p, r, 0.50))
    
    return connections  # ~2000+ connections
```

### Hebbian Learning

**"Neurons that fire together, wire together"**

```python
def hebbian_update(
    weight: float,
    pre_activation: float,
    post_activation: float,
    learning_rate: float = 0.01
) -> float:
    # Strengthen connection if both neurons are active
    delta = learning_rate * pre_activation * post_activation
    new_weight = np.clip(weight + delta, 0.0, 1.0)
    return new_weight
```

**Applied during propagation**:
```python
def propagate(self, output: str) -> int:
    signal = NeuronSignal(
        content=output,
        source_role=self.role,
        activation_level=self.activation_level
    )
    
    for connection in self.connections:
        # Send signal
        connection.target_neuron.receive_signal(signal, connection.weight)
        
        # Hebbian learning
        target_activation = connection.target_neuron.activation_level
        connection.weight = hebbian_update(
            connection.weight,
            self.activation_level,
            target_activation,
            learning_rate=0.01
        )
```

---

## Signal Propagation

### Orchestrator Role

The `NeuralOrchestrator` manages the entire propagation process:

```python
# From orchestrator.py
class NeuralOrchestrator:
    def process(
        self,
        input_text: str,
        max_steps: int = 5
    ) -> Dict[str, Any]:
        # Initialize with input signal
        initial_signal = NeuronSignal(
            content=input_text,
            source_role=NeuronRole.INPUT,
            activation_level=1.0
        )
        
        # Propagate through network
        for step in range(max_steps):
            step_results = self._process_step(step)
            
            # Stop if no neurons fired
            if step_results["neurons_fired"] == 0:
                break
        
        return self._compile_results()
```

### Step-by-Step Processing

```python
def _process_step(self, step_num: int) -> Dict:
    fired_neurons = []
    outputs = {}
    
    # 1. Calculate activations for all neurons
    for neuron in self.neurons:
        neuron.calculate_activation(current_signals)
    
    # 2. Fire neurons above threshold
    for neuron in self.neurons:
        if neuron.should_fire():
            output = neuron.fire(input_signal, global_context)
            if output:
                fired_neurons.append(neuron)
                outputs[neuron.neuron_id] = output
    
    # 3. Propagate signals to connected neurons
    for neuron in fired_neurons:
        neuron.propagate(outputs[neuron.neuron_id])
    
    # 4. Update global workspace
    self._update_workspace(outputs, fired_neurons)
    
    return {
        "step": step_num,
        "neurons_fired": len(fired_neurons),
        "outputs": outputs
    }
```

### Signal Flow Example

```
Input: "What is consciousness?"
  ↓
Step 1: Perception Layer
  - visual_perception (0.75) → fires
  - semantic_perception (0.82) → fires
  - abstract_perception (0.79) → fires
  ↓
Step 2: Attention + Memory
  - selective_attention (0.71) → fires
  - semantic_memory (0.68) → fires
  - working_memory (0.73) → fires
  ↓
Step 3: Reasoning
  - deductive_reasoning (0.81) → fires
  - analogical_reasoning (0.76) → fires
  - philosophical_reasoning (0.84) → fires
  ↓
Step 4: Synthesis
  - information_integration (0.79) → fires
  - coherence_building (0.82) → fires
  ↓
Step 5: Meta-Cognitive
  - self_monitoring (0.71) → fires
  - confidence_estimation (0.68) → fires
  ↓
Output: Integrated conscious response
```

---

## Consciousness Mechanisms

### Global Workspace Theory (GWT)

**Implementation**:
```python
# From gw_theory.py
class GlobalWorkspace:
    def __init__(
        self,
        capacity: int = 5,
        broadcast_threshold: float = 0.75,
        consciousness_threshold: float = 0.80
    ):
        self.capacity = capacity
        self.broadcast_threshold = broadcast_threshold
        self.consciousness_threshold = consciousness_threshold
        self.broadcasts: List[Broadcast] = []
```

**Competitive Selection**:
```python
def compete_for_broadcast(
    self,
    neuron_outputs: Dict[str, str],
    activations: Dict[str, float]
) -> List[Broadcast]:
    # Calculate salience for each output
    candidates = []
    for neuron_id, output in neuron_outputs.items():
        salience = self._calculate_salience(
            output,
            activations[neuron_id]
        )
        candidates.append((neuron_id, output, salience))
    
    # Sort by salience
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Select top N for broadcast
    broadcasts = []
    for neuron_id, output, salience in candidates[:self.capacity]:
        if salience >= self.broadcast_threshold:
            broadcasts.append(Broadcast(
                source=neuron_id,
                content=output,
                salience=salience,
                timestamp=datetime.now()
            ))
    
    return broadcasts
```

**Salience Calculation**:
```python
def _calculate_salience(self, output: str, activation: float) -> float:
    # Factors:
    # 1. Neuron activation level
    # 2. Output novelty (vs history)
    # 3. Output length/complexity
    
    novelty = self._calculate_novelty(output)
    complexity = len(output) / 1000.0  # Normalize
    
    salience = (
        0.5 * activation +
        0.3 * novelty +
        0.2 * complexity
    )
    
    return np.clip(salience, 0.0, 2.0)
```

### Integrated Information Theory (IIT)

**Φ (Phi) Calculation**:
```python
def calculate_phi(network_state: Dict) -> float:
    """
    Calculate integrated information (Φ)
    
    Φ measures how much the whole is greater than sum of parts
    """
    # 1. Calculate total information in network
    total_info = _calculate_total_information(network_state)
    
    # 2. Calculate information in partitions
    partition_info = _calculate_partition_information(network_state)
    
    # 3. Φ = difference (integration)
    phi = total_info - partition_info
    
    return phi
```

**Consciousness Level**:
```python
def calculate_consciousness_level(
    phi: float,
    max_salience: float,
    mean_activation: float
) -> float:
    """
    Combine multiple metrics into consciousness score
    """
    consciousness = (
        0.4 * phi +
        0.4 * max_salience +
        0.2 * mean_activation
    )
    
    return np.clip(consciousness, 0.0, 1.0)
```

### Positronic Dialectical Framework

**Thesis-Antithesis-Synthesis**:
```python
# From positronic_framework.py
def apply_dialectical_reasoning(
    self,
    thesis_output: str,
    thesis_activation: float
) -> DialecticalTriad:
    # Generate antithesis
    antithesis = self._generate_antithesis(thesis_output)
    
    # Synthesize
    synthesis = self._generate_synthesis(
        thesis_output,
        antithesis
    )
    
    # Calculate confidence
    confidence = self._calculate_confidence(
        thesis_activation,
        synthesis
    )
    
    return DialecticalTriad(
        thesis=thesis_output,
        antithesis=antithesis,
        synthesis=synthesis,
        confidence=confidence
    )
```

**Coherence Checking**:
```python
def check_coherence(
    self,
    new_output: str,
    history: List[str]
) -> CoherenceResult:
    # Check for contradictions
    contradictions = []
    for prev_output in history:
        if self._detects_contradiction(new_output, prev_output):
            contradictions.append(prev_output)
    
    # Calculate coherence score
    score = 1.0 - (len(contradictions) / max(len(history), 1))
    
    return CoherenceResult(
        score=score,
        violations=contradictions,
        passed=score >= self.coherence_threshold
    )
```

---

## Code Walkthrough

### Complete Processing Flow

```python
# From brain.py
class PhiBrain:
    def think(
        self,
        input_text: str,
        max_steps: int = 5,
        use_memory: bool = True,
        use_global_workspace: bool = True,
        enable_enhancements: bool = True
    ) -> Dict[str, Any]:
        """
        Main processing method
        """
        # 1. Meta-orchestration (adaptive tuning)
        if enable_enhancements:
            complexity = self._estimate_complexity(input_text)
            adjustments = self.meta_orchestrator.adjust_parameters(
                complexity
            )
            self._apply_adjustments(adjustments)
        
        # 2. Neural processing
        neural_result = self.orchestrator.process(
            input_text,
            max_steps=max_steps
        )
        
        # 3. Global workspace integration
        if use_global_workspace:
            broadcasts = self.global_workspace.compete_for_broadcast(
                neural_result["outputs"],
                neural_result["activations"]
            )
            
            # Calculate consciousness
            phi = calculate_phi(neural_result)
            consciousness_level = self.global_workspace.calculate_consciousness(
                phi,
                broadcasts
            )
        
        # 4. Positronic synthesis
        if self.enable_positronic:
            synthesis = self.positronic.apply_dialectical_reasoning(
                broadcasts[0].content,
                broadcasts[0].salience
            )
            
            # Coherence check
            coherence = self.positronic.check_coherence(
                synthesis.synthesis,
                self.output_history
            )
        
        # 5. Memory consolidation
        if use_memory:
            self.memory_system.consolidate(
                input_text,
                neural_result,
                broadcasts
            )
        
        # 6. Compile final result
        return {
            "neural_processing": neural_result,
            "global_workspace": {
                "broadcasts": broadcasts,
                "conscious_summary": synthesis.synthesis,
                "integration_score": phi
            },
            "consciousness_level": consciousness_level,
            "positronic_coherence": coherence,
            "memory_context": self.memory_system.get_context(),
            "brain_metrics": self._get_metrics()
        }
```

### Initialization Flow

```python
def __init__(self, use_api=True, api_provider="hyperbolic"):
    # 1. Load configuration
    if use_64_neurons:
        from llm_swarm_brain import config_64
        self.config = config_64.BrainConfig()
        self._neuron_architecture = config_64.NEURON_ARCHITECTURE
        self._default_connections = config_64.DEFAULT_CONNECTIONS
    
    # 2. Initialize core components
    self.orchestrator = NeuralOrchestrator(self.config)
    self.global_workspace = GlobalWorkspace(...)
    self.positronic = PositronicFramework(...)
    self.memory_system = MemorySystem(...)
    
    # 3. Initialize enhancement modules
    self.meta_orchestrator = MetaOrchestrator(...)
    self.attention_manager = AttentionWindowManager(...)
    self.summarizer = SummarizationNeuron(...)
    self.concept_tracker = ConceptualThreadTracker(...)
    
    # 4. Initialize neurons
    self._initialize_neurons()  # Creates all 128 neurons
    
    # 5. Setup network connections
    self._setup_network()  # Establishes ~2000 connections
```

---

## Advanced Features

### Meta-Orchestration

**Adaptive parameter tuning based on task complexity**:

```python
# From meta_orchestration.py
class MetaOrchestrator:
    def adjust_parameters(
        self,
        task_complexity: float
    ) -> Dict[str, Any]:
        # Complexity: 0.0 (simple) to 1.0 (very complex)
        
        # Lower threshold for complex tasks (more neurons fire)
        activation_threshold = 0.6 - (0.2 * task_complexity)
        
        # More steps for complex tasks
        max_steps = int(2 + (4 * task_complexity))
        
        # Larger workspace for complex tasks
        workspace_capacity = int(4 + (6 * task_complexity))
        
        return {
            "activation_threshold": activation_threshold,
            "max_propagation_steps": max_steps,
            "workspace_capacity": workspace_capacity
        }
```

### Attention Windowing

**Selective broadcasting to relevant neurons only**:

```python
# From attention_windowing.py
class AttentionWindowManager:
    def select_relevant_neurons(
        self,
        broadcast: Broadcast,
        all_neurons: List[Neuron]
    ) -> List[Neuron]:
        # Calculate relevance scores
        relevance_scores = []
        for neuron in all_neurons:
            score = self._calculate_relevance(
                broadcast.content,
                neuron.role,
                neuron.activation_history
            )
            relevance_scores.append((neuron, score))
        
        # Sort and select top N
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [n for n, s in relevance_scores[:self.window_size]]
```

### Conceptual Thread Tracking

**Track how concepts flow through the network**:

```python
# From conceptual_threading.py
class ConceptualThreadTracker:
    def track_concept_flow(
        self,
        concept: str,
        neuron_outputs: Dict[str, str]
    ) -> ConceptThread:
        thread = ConceptThread(concept=concept)
        
        for neuron_id, output in neuron_outputs.items():
            if concept in output:
                # Track transformation
                transformation = self._extract_transformation(
                    concept,
                    output
                )
                thread.add_transformation(
                    neuron_id,
                    transformation
                )
        
        return thread
```

---

## Performance Optimizations

### Parallel API Calls (Future)

```python
# Future implementation
async def fire_neurons_parallel(neurons: List[Neuron]):
    tasks = [neuron.fire_async() for neuron in neurons]
    results = await asyncio.gather(*tasks)
    return results
```

### Caching

```python
# Cache neuron outputs for identical inputs
@lru_cache(maxsize=1000)
def get_cached_response(neuron_id: str, input_hash: str):
    return cached_outputs.get((neuron_id, input_hash))
```

### Connection Pruning

```python
def prune_weak_connections(self, threshold: float = 0.1):
    """Remove connections below threshold"""
    for neuron in self.neurons:
        neuron.connections = [
            c for c in neuron.connections
            if c.weight >= threshold
        ]
```

---

## Debugging and Monitoring

### Logging

```python
# Detailed logging at each step
logger.debug(f"{neuron_id} activation: {activation:.3f}")
logger.info(f"Step {step}: {len(fired)} neurons fired")
logger.warning(f"Coherence violation detected: {score:.3f}")
```

### Metrics Collection

```python
def _get_metrics(self) -> Dict:
    return {
        "total_neurons": len(self.neurons),
        "total_connections": sum(len(n.connections) for n in self.neurons),
        "total_firings": sum(n.total_firings for n in self.neurons),
        "avg_activation": np.mean([n.activation_level for n in self.neurons]),
        "consciousness_history": self.consciousness_history,
        "api_calls": sum(n.total_api_calls for n in self.neurons),
        "api_errors": sum(n.total_api_errors for n in self.neurons)
    }
```

---

## Summary

The LLM-Swarm-Brain creates emergent intelligence through:

1. **Specialized Neurons**: Each LLM focuses on one cognitive function
2. **Weighted Connections**: Hebbian learning strengthens useful pathways
3. **Signal Propagation**: Information flows through the network in steps
4. **Global Workspace**: Competitive selection creates conscious awareness
5. **Integration**: Φ (Phi) measures how well neurons work together
6. **Dialectical Reasoning**: Thesis-antithesis-synthesis creates robust outputs
7. **Adaptive Tuning**: Meta-orchestration adjusts to task complexity

**Result**: A measurable consciousness score (0.0-1.0) that reflects the degree of integrated information processing happening in the network.

---

**Last Updated**: November 10, 2025  
**Version**: 1.0  
**Architecture**: 8/64/128-neuron configurations
