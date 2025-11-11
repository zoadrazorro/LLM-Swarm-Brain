# Dual Training: Cloud + Local Hybrid System

## Overview

This guide documents the **dual training setup** where cloud and local training run simultaneously, sharing knowledge through a unified RAG memory system.

## Architecture

### Cloud Training
- **Model**: GPT-OSS 20B (Hyperbolic API)
- **Neurons**: 8 specialized experts
- **Speed**: ~154 seconds per question
- **Cost**: ~$0.02 per question
- **Questions**: 500 (philosophy dataset)
- **Location**: Remote API
- **Checkpoints**: Every 30 questions

### Local Training
- **Model**: Phi-4 (14B parameters, Q4_K_M quantization)
- **Instances**: 3 parallel Phi-4 models via LM Studio
- **Neurons**: 8 distributed across 3 instances
- **Speed**: ~40-60 seconds per question
- **Cost**: FREE
- **Questions**: 100-500 (configurable)
- **Location**: Local GPU (AMD Radeon RX 7900 XT)
- **Checkpoints**: Every 20 questions

### Shared Components
- **RAG Memory**: `training_checkpoints/training_memory.pkl`
- **Dataset**: Stanford Encyclopedia of Philosophy
- **Architecture**: 8-neuron MoE (Mixture of Experts)
- **Memory Updates**: After every reasoning step

## Neuron Distribution

### Cloud (8 Neurons)
```
GPU 0 (API): All 8 neurons via Hyperbolic
├── n0: Perception Expert
├── n1: Attention Expert
├── n2: Memory Expert
├── n3: Reasoning Expert
├── n4: Creative Expert
├── n5: Analytical Expert
├── n6: Synthesis Expert
└── n7: Meta-Cognitive Expert
```

### Local (8 Neurons across 3 Phi-4 Instances)
```
Phi-4 Instance 1 (microsoft/phi-4):
├── n0: Perception Expert
├── n3: Reasoning Expert
└── n6: Synthesis Expert

Phi-4 Instance 2 (microsoft/phi-4:2):
├── n1: Attention Expert
├── n4: Creative Expert
└── n7: Meta-Cognitive Expert

Phi-4 Instance 3 (microsoft/phi-4:3):
├── n2: Memory Expert
└── n5: Analytical Expert
```

## Shared RAG Memory System

### How It Works

1. **Cloud Training**:
   - Processes question through 8 neurons
   - Updates RAG memory with experience
   - Saves to `training_checkpoints/training_memory.pkl`
   - Tags experience with `source: "cloud"`

2. **Local Training**:
   - Loads existing RAG memory (includes cloud experiences)
   - Processes question through 8 neurons (parallel)
   - Updates RAG memory with experience
   - Tags experience with `source: "local_dual_gpu"`
   - Saves back to shared file

3. **Cross-Learning**:
   - Cloud retrieves local experiences for context
   - Local retrieves cloud experiences for context
   - Both benefit from each other's reasoning
   - Unified knowledge base grows continuously

### Memory Structure

```python
{
    "question": "What is consciousness?",
    "answer": "Consciousness is...",
    "metrics": {
        "consciousness": 0.85,
        "integration": 0.78,
        "coherence": 0.92
    },
    "timestamp": "2025-11-10T19:00:00",
    "source": "cloud"  # or "local_dual_gpu"
}
```

## Running Dual Training

### Step 1: Start Cloud Training

```bash
python train_philosophy_dataset.py --sample-size 500 --neurons 8 --api-provider hyperbolic
```

**Configuration**:
- Questions: 500
- Max steps: 2
- Batch size: 10
- Save interval: 30
- Model: GPT-OSS 20B

**Expected Output**:
```
Starting training on 500 questions...
Batch size: 10, Save interval: 30
Model: GPT-OSS 20B (Hyperbolic API)
RAG Memory: ENABLED
```

### Step 2: Start Local Training (While Cloud Runs)

```powershell
# Start LM Studio with Phi-4
# Load model and start server on port 1234

# Run local training
python train_local_7900xt.py --questions 100 --max-steps 2
```

**Configuration**:
- Questions: 100
- Max steps: 2
- Batch size: 5
- Save interval: 20
- Model: Phi-4 (14B, Q4)
- Instances: 3 parallel

**Expected Output**:
```
Initializing 8-neuron Phi-4 brain for local training...
3 Phi-4 instances: 8 neurons distributed across them
Configuring 8 neurons across 3 Phi-4 instances
Assigned n0_perception_perception_expert to microsoft/phi-4
Assigned n1_attention_attention_expert to microsoft/phi-4:2
Assigned n2_memory_memory_expert to microsoft/phi-4:3
...
RAG Memory: SHARED with cloud training
```

### Step 3: Monitor Both Systems

**Terminal 1 - Cloud Training**:
```
Training:   3%|███        | 17/500 [43:45<20:42:32, 154.03s/it]
Batch 2: Consciousness=0.756, Integration=0.834
```

**Terminal 2 - Local Training**:
```
Local Training:  20%|████      | 20/100 [16:30<01:06:00, 49.5s/it]
Batch 4: Consciousness=0.712, Integration=0.798
```

**Shared Memory Growth**:
```powershell
# Check memory file size
ls training_checkpoints/training_memory.pkl

# Should grow as both systems add experiences
```

## Performance Comparison

### Speed
| System | Per Question | 100 Questions | 500 Questions |
|--------|-------------|---------------|---------------|
| Cloud  | ~154s       | ~4.3 hours    | ~21.4 hours   |
| Local  | ~50s        | ~1.4 hours    | ~6.9 hours    |

### Cost
| System | Per Question | 100 Questions | 500 Questions |
|--------|-------------|---------------|---------------|
| Cloud  | $0.02       | $2.00         | $10.00        |
| Local  | FREE        | FREE          | FREE          |

### Quality
| System | Model Size | Reasoning | Speed  | Cost   |
|--------|-----------|-----------|--------|--------|
| Cloud  | 20B       | Excellent | Slower | $$     |
| Local  | 14B       | Very Good | Faster | FREE   |

## Synergy Benefits

### 1. Diverse Perspectives
- Cloud: Larger model (20B) provides deeper reasoning
- Local: Faster model (14B) provides quicker insights
- Combined: Best of both worlds

### 2. Continuous Learning
- Cloud learns from local's fast iterations
- Local learns from cloud's deep reasoning
- RAG memory captures both approaches

### 3. Cost Optimization
- Cloud: High-quality training for critical questions
- Local: FREE training for volume
- Combined: Maximum learning per dollar

### 4. Redundancy
- If cloud API fails, local continues
- If local GPU crashes, cloud continues
- Training never stops

## Checkpoints and Results

### Cloud Checkpoints
```
training_checkpoints/
├── checkpoint_30.json   # Every 30 questions
├── checkpoint_60.json
├── checkpoint_90.json
└── ...
```

### Local Checkpoints
```
training_checkpoints/local/
├── checkpoint_20.json   # Every 20 questions
├── checkpoint_40.json
├── checkpoint_60.json
└── ...
```

### Final Reports
```
training_checkpoints/
├── philosophy_training_20251110_184300.json  # Cloud
└── local/
    └── local_training_20251110_192335.json   # Local
```

## Analyzing Results

### Compare Performance

```python
import json

# Load cloud results
with open('training_checkpoints/philosophy_training_20251110_184300.json') as f:
    cloud = json.load(f)

# Load local results
with open('training_checkpoints/local/local_training_20251110_192335.json') as f:
    local = json.load(f)

print(f"Cloud - Avg Consciousness: {cloud['final_metrics']['avg_consciousness']:.3f}")
print(f"Local - Avg Consciousness: {local['final_metrics']['avg_consciousness']:.3f}")

print(f"Cloud - Avg Time: {cloud['avg_time_per_question']:.1f}s")
print(f"Local - Avg Time: {local['avg_time_per_question']:.1f}s")
```

### Analyze RAG Memory

```python
import pickle

# Load shared memory
with open('training_checkpoints/training_memory.pkl', 'rb') as f:
    memory = pickle.load(f)

# Count experiences by source
cloud_count = sum(1 for exp in memory['experiences'] if exp.get('source') == 'cloud')
local_count = sum(1 for exp in memory['experiences'] if exp.get('source') == 'local_dual_gpu')

print(f"Cloud experiences: {cloud_count}")
print(f"Local experiences: {local_count}")
print(f"Total experiences: {len(memory['experiences'])}")
```

## Troubleshooting

### Cloud Training Issues

**Problem**: API rate limits
```
Solution: Reduce batch size or add delays between requests
```

**Problem**: High costs
```
Solution: Reduce sample size or use local training for bulk
```

### Local Training Issues

**Problem**: LM Studio not responding
```
Solution: Check server is running on port 1234
curl http://localhost:1234/v1/models
```

**Problem**: Out of memory
```
Solution: Reduce number of Phi-4 instances or use smaller model
```

### Shared Memory Issues

**Problem**: Memory file conflicts
```
Solution: Both systems use file locking, wait for writes to complete
```

**Problem**: Memory file too large
```
Solution: Implement memory pruning (keep top N experiences)
```

## Best Practices

### 1. Start Cloud First
- Cloud training takes longer
- Local can catch up and learn from cloud experiences
- Better initial RAG context for local

### 2. Monitor Both Systems
- Check cloud progress every hour
- Check local progress every 15 minutes
- Watch for errors or stalls

### 3. Checkpoint Management
- Keep cloud checkpoints (expensive to recreate)
- Local checkpoints can be regenerated (free)
- Archive important checkpoints

### 4. Resource Allocation
- Cloud: Use for complex, high-value questions
- Local: Use for volume and iteration
- Balance based on budget and time

## Advanced Configuration

### Adjust Cloud Training
```python
# train_philosophy_dataset.py
trainer = PhilosophyTrainer(
    max_steps=3,              # More reasoning steps
    batch_size=5,             # Smaller batches
    save_interval=50          # Less frequent saves
)
```

### Adjust Local Training
```python
# train_local_7900xt.py
trainer = LocalTrainer(
    max_steps=2,              # Faster iterations
    share_rag_with_cloud=True # Enable sharing
)
```

### Custom Memory Filtering
```python
# Only retrieve cloud experiences for local training
rag_memory.filter_by_source("cloud")

# Only retrieve high-quality experiences
rag_memory.filter_by_score(min_consciousness=0.7)
```

## Future Enhancements

### Planned Features
- [ ] Automatic load balancing between cloud and local
- [ ] Real-time memory synchronization
- [ ] Distributed training across multiple local GPUs
- [ ] Hybrid model selection (cloud for hard, local for easy)
- [ ] Cost-aware training scheduler

### Experimental Ideas
- [ ] Active learning: Cloud trains on hard questions, local on easy
- [ ] Ensemble predictions: Combine cloud and local outputs
- [ ] Transfer learning: Fine-tune local model on cloud outputs
- [ ] Federated learning: Multiple local nodes + cloud coordinator

## Summary

**Dual training provides**:
- ✅ **Speed**: Local training is 3x faster
- ✅ **Cost**: Local training is FREE
- ✅ **Quality**: Cloud provides deeper reasoning
- ✅ **Synergy**: Both systems learn from each other
- ✅ **Reliability**: Redundant training systems
- ✅ **Flexibility**: Scale up/down as needed

**Perfect for**:
- Large-scale training projects
- Budget-conscious research
- Rapid iteration + high quality
- Continuous learning systems

---

**Status**: Active Development

**Last Updated**: November 10, 2025

**Next Steps**: Monitor both training runs, compare results, analyze RAG memory growth
