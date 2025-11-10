# Quick Start - Local Training with Phi-4

Your Phi-4 model is downloading! Here's what to do once it's ready.

## Step-by-Step Setup (5 minutes)

### 1. Load Phi-4 in LM Studio

Once download completes:

1. **Go to "Local Server" tab** in LM Studio
2. **Select the model**: Phi-4 (Q4_K_M) you just downloaded
3. **Configure**:
   - Port: `1234`
   - Context Length: `16384` (or max available)
   - GPU Layers: `99` (use full GPU acceleration)
4. **Click "Start Server"**

### 2. Verify Server is Running

Open PowerShell and test:
```powershell
curl http://localhost:1234/v1/models
```

Should return model information.

### 3. Start Training

**Option A: Use Quick-Start Script**
```powershell
.\start_local_training.ps1
```

**Option B: Manual Command**
```powershell
# Quick test (10 questions, ~10 minutes)
python train_local_7900xt.py --questions 10 --max-steps 2

# Full training (100 questions, ~1.5 hours)
python train_local_7900xt.py --questions 100 --max-steps 2
```

## What Happens During Training

### First Question
You'll see:
```
Initializing 4-neuron Phi-4 brain for dual-GPU training...
GPU 0: 2 neurons | GPU 1: 2 neurons
RAG Memory: SHARED with cloud training
Starting LOCAL training on 100 questions...
```

### Progress Updates
Every 5 questions:
```
Batch 1: Consciousness=0.756, Integration=0.834, Avg Time=45.2s/question
```

### Checkpoints
Every 20 questions:
```
Checkpoint saved: training_checkpoints/local/checkpoint_20.json
```

## Monitor Both Systems

### Terminal 1 - Cloud Training
```
Training: 5/500 [14:25<23:15:42, 169.12s/it]
```

### Terminal 2 - Local Training
```
Local Training: 15/100 [11:18<01:03:45, 2.21it/s]
```

### Shared RAG Memory
Both systems learn from each other! Check growth:
```powershell
# View shared memory size
ls training_checkpoints/training_memory.pkl
```

## Expected Performance

### Speed
- **Per question**: 40-60 seconds
- **10 questions**: ~10 minutes
- **100 questions**: ~1.5 hours

### VRAM Usage
Monitor with AMD tools:
- GPU 0: ~16GB / 20GB (80%)
- GPU 1: ~16GB / 20GB (80%)

### Quality
- Phi-4 (14B) provides excellent reasoning
- Slightly lower than cloud's GPT-OSS (20B)
- But **FREE** and **faster**!

## Troubleshooting

### "LM Studio server not detected"
- Make sure server is started in LM Studio
- Check port 1234 is not blocked
- Verify with: `curl http://localhost:1234/v1/models`

### Slow Performance
- Ensure GPU acceleration is enabled in LM Studio
- Check GPU layers set to 99
- Close other GPU-intensive applications

### Out of Memory
- Reduce to 2 neurons (edit `config_local_7900xt.py`)
- Or use smaller context length in LM Studio

## Next Steps

Once training starts:
1. **Let it run** - it will save checkpoints automatically
2. **Monitor progress** - watch the console output
3. **Compare with cloud** - see how local vs cloud performs
4. **Analyze results** - check `training_checkpoints/local/` when done

---

**Ready to start!** 🚀

As soon as Phi-4 download completes, just run:
```powershell
.\start_local_training.ps1
```
