# Dual 7900XT Setup Guide - Phi-4 with Shared RAG

Complete guide for running PhiBrain on dual AMD Radeon RX 7900 XT GPUs with Phi-4, sharing RAG memory with cloud training.

## Hardware Configuration

**Dual AMD Radeon RX 7900 XT**
- GPU 0: 20 GB GDDR6 → 2 Phi-4 neurons
- GPU 1: 20 GB GDDR6 → 2 Phi-4 neurons
- **Total**: 40 GB VRAM, 4 powerful neurons

## Architecture

### Neuron Distribution

**GPU 0 (2 neurons)**:
- Neuron 0: Perception & Reasoning
- Neuron 1: Attention & Memory

**GPU 1 (2 neurons)**:
- Neuron 2: Creative & Analytical  
- Neuron 3: Synthesis & Meta-Cognitive

### Cross-GPU Communication

Neurons communicate across GPUs through the connection matrix, enabling:
- Forward reasoning pipeline (GPU 0 → GPU 1)
- Feedback loops (GPU 1 → GPU 0)
- Integrated consciousness across both GPUs

## RAG Memory Sharing

### How It Works

1. **Cloud training** (Hyperbolic API):
   - Saves experiences to `training_checkpoints/training_memory.pkl`
   - Updates every 30 questions

2. **Local training** (Dual 7900XT):
   - Loads same `training_memory.pkl` on startup
   - Adds new experiences with `source: "local_dual_gpu"` tag
   - Saves every 10 questions to share back

3. **Continuous Learning**:
   - Both systems learn from each other's experiences
   - Cloud benefits from local GPU insights
   - Local benefits from cloud's larger model reasoning

### Benefits

✅ **Unified Knowledge**: Single shared memory pool
✅ **Faster Learning**: Both systems contribute
✅ **Cost Efficient**: Local training is FREE
✅ **Quality Boost**: Cloud uses 20B model, local uses 14B

## LM Studio Setup

### 1. Download Phi-4 Model

In LM Studio:
1. Search for "Phi-4"
2. Select **Q4_K_M** quantization (~8GB)
3. Download the model

Recommended:
```
microsoft/Phi-4-GGUF
Quantization: Q4_K_M
Size: ~8GB
```

### 2. Configure Multi-GPU

LM Studio doesn't natively support multi-GPU for multiple instances. You'll need to run **2 separate LM Studio instances**:

**Option A: Two LM Studio Instances (Recommended)**

Terminal 1 (GPU 0):
```bash
# Set environment variable to use GPU 0
$env:CUDA_VISIBLE_DEVICES="0"
# Start LM Studio on port 1234
lmstudio server start --port 1234 --gpu 0
```

Terminal 2 (GPU 1):
```bash
# Set environment variable to use GPU 1
$env:CUDA_VISIBLE_DEVICES="1"
# Start LM Studio on port 1235
lmstudio server start --port 1235 --gpu 1
```

**Option B: Use llama.cpp Directly**

For better GPU control, use llama.cpp:

```bash
# GPU 0 - Port 1234
llama-server.exe --model phi-4-q4.gguf --port 1234 --gpu-layers 99 --main-gpu 0

# GPU 1 - Port 1235
llama-server.exe --model phi-4-q4.gguf --port 1235 --gpu-layers 99 --main-gpu 1
```

### 3. Verify Both Servers

```bash
# Check GPU 0
curl http://localhost:1234/v1/models

# Check GPU 1
curl http://localhost:1235/v1/models
```

## Running Training

### Start Local Training

```bash
# 100 questions, shares RAG with cloud
python train_local_7900xt.py --questions 100 --max-steps 2
```

### Monitor Both Systems

**Terminal 1** - Cloud training (already running):
```
Training: 3/500 [00:08<21:25, 154.21s/it]
```

**Terminal 2** - Local training (new):
```
Local Training: 5/100 [00:02<00:38, 2.47it/s]
```

### Check Shared RAG Memory

```python
import pickle

# Load shared memory
with open('training_checkpoints/training_memory.pkl', 'rb') as f:
    memory = pickle.load(f)

# See experiences from both systems
for exp in memory['training_history'][-10:]:
    source = exp.get('metadata', {}).get('source', 'cloud')
    print(f"{source}: {exp['question'][:50]}...")
```

## Performance Expectations

### Speed

**Per Question** (4 neurons, 2 steps):
- Inference: ~10-15s per neuron
- Total: ~40-60s per question
- **Faster than cloud** (cloud = 150s/question)

### Training Duration

| Questions | Duration | Cost |
|-----------|----------|------|
| 10 | ~10 min | FREE |
| 50 | ~50 min | FREE |
| 100 | ~1.5 hours | FREE |
| 500 | ~8 hours | FREE |

### VRAM Usage

```
GPU 0: 2 neurons × 8GB = 16GB used (4GB free)
GPU 1: 2 neurons × 8GB = 16GB used (4GB free)
Total: 32GB / 40GB (80% utilization)
```

## Comparison: Local vs Cloud

| Metric | Dual 7900XT | Cloud (Hyperbolic) |
|--------|-------------|-------------------|
| **Cost** | FREE 🎉 | $0.02/question |
| **Speed** | 40-60s/question | 150s/question |
| **Model** | Phi-4 (14B) | GPT-OSS (20B) |
| **Neurons** | 4 (2 per GPU) | 8 |
| **Quality** | Excellent | Better |
| **RAG** | SHARED | SHARED |

## Hybrid Training Strategy

### Recommended Approach

1. **Start cloud training** (already running):
   - 500 questions on Hyperbolic
   - Builds initial RAG memory
   - Cost: ~$10

2. **Start local training** (parallel):
   - 100-200 questions on dual 7900XT
   - Adds diverse experiences to shared RAG
   - Cost: FREE

3. **Both systems learn from each other**:
   - Cloud gets local GPU insights
   - Local gets cloud's advanced reasoning
   - Unified knowledge base

### Timeline

**Cloud** (500 questions):
- Started: 6:30 PM
- ETA: Tuesday 8:00 PM (~25 hours)

**Local** (100 questions):
- Start now: 6:40 PM
- ETA: 8:10 PM (~1.5 hours)
- **Finishes first!**

## Troubleshooting

### GPU Not Detected

```bash
# Check AMD GPU status
rocm-smi

# Verify both GPUs visible
rocm-smi --showid
```

### Port Conflicts

If ports 1234/1235 are in use:
```bash
# Use different ports
python train_local_7900xt.py --lm-studio-port 8080
```

### Memory Sync Issues

```bash
# Manually check memory file
ls -lh training_checkpoints/training_memory.pkl

# Should update every 10 local questions
# Should update every 30 cloud questions
```

### One GPU Not Working

If only one GPU works, reduce to 2 neurons:
```python
# In config_local_7900xt.py
total_neurons: int = 2  # Instead of 4
```

## Advanced: Monitor GPU Usage

```bash
# Watch GPU usage in real-time
watch -n 1 rocm-smi

# Expected output:
# GPU 0: 16GB / 20GB, 80% utilization
# GPU 1: 16GB / 20GB, 80% utilization
```

## Files Created

- `config_local_7900xt.py` - Dual-GPU configuration
- `train_local_7900xt.py` - Local training script with RAG sharing
- `training_checkpoints/training_memory.pkl` - **SHARED** RAG memory
- `training_checkpoints/local/` - Local-specific checkpoints

## Next Steps

1. **Set up LM Studio** with Phi-4 on both GPUs
2. **Start local training** while cloud runs
3. **Monitor shared RAG** growth
4. **Compare results** after both complete

---

**Ready for dual-GPU training!** 🚀🚀

Start with a quick test:
```bash
python train_local_7900xt.py --questions 10 --max-steps 2
```

Watch both systems learn together! 🧠💡
