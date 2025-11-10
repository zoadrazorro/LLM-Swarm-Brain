# Local Training Guide - AMD Radeon RX 7900 XT

Complete guide for running PhiBrain training locally on your 7900XT using LM Studio and Phi-3-mini models.

## Hardware Specifications

**AMD Radeon RX 7900 XT**
- VRAM: 20 GB GDDR6
- Memory Bandwidth: 800 GB/s
- Compute: ~51 TFLOPS (FP32)
- Architecture: RDNA 3

## Model Configuration

**Phi-3-mini-4k-instruct**
- Parameters: 3.8B
- Context Length: 4096 tokens
- Quantization: Q4_K_M (~2GB per instance)
- **8 concurrent instances** (16GB used, 4GB buffer)

## Setup Instructions

### 1. Install LM Studio

1. Download LM Studio from https://lmstudio.ai/
2. Install and launch LM Studio
3. Go to the "Discover" tab

### 2. Download Phi-3-mini Model

In LM Studio:
1. Search for "Phi-3-mini-4k-instruct"
2. Select the **Q4_K_M** quantization variant
3. Download the model (~2.3GB)

Recommended model:
```
microsoft/Phi-3-mini-4k-instruct-gguf
Quantization: Q4_K_M
```

### 3. Start LM Studio Server

1. Go to the "Local Server" tab in LM Studio
2. Load the Phi-3-mini-4k-instruct (Q4_K_M) model
3. Configure server settings:
   - **Port**: 1234 (default)
   - **Context Length**: 4096
   - **GPU Layers**: Auto (will use all available)
4. Click "Start Server"

Verify server is running:
```bash
curl http://localhost:1234/v1/models
```

### 4. Run Local Training

Basic training (100 questions):
```bash
python train_local_7900xt.py --questions 100 --max-steps 2
```

Extended training (500 questions):
```bash
python train_local_7900xt.py --questions 500 --max-steps 2 --batch-size 10 --save-interval 50
```

Quick test (10 questions):
```bash
python train_local_7900xt.py --questions 10 --max-steps 1
```

## Performance Expectations

### Speed Estimates

**Per Question** (8 neurons, 2 steps):
- Inference time: ~5-10 seconds per neuron
- Total per question: ~40-80 seconds
- With overhead: ~60-120 seconds

**Training Duration**:
- 10 questions: ~10-20 minutes
- 100 questions: ~1.5-3 hours
- 500 questions: ~8-15 hours

### VRAM Usage

```
8 neurons × 2GB (Q4) = 16GB
System overhead = 2-3GB
Total usage = 18-19GB / 20GB available
```

**Safe and stable** - leaves buffer for system processes.

### Cost

**FREE!** 🎉
- No API costs
- No cloud fees
- Unlimited training on your hardware

## Configuration Options

### Reduce Neurons (Faster, Less Capable)

Edit `config_local_7900xt.py`:
```python
total_neurons: int = 4  # Instead of 8
```

### Increase Quality (Slower, Better Results)

Use Q8 quantization instead of Q4:
```python
quantization: str = "Q8_0"  # ~4GB per model
total_neurons: int = 4  # Reduce to fit in VRAM
```

### Adjust Reasoning Steps

```bash
# Faster training (1 step)
python train_local_7900xt.py --questions 100 --max-steps 1

# Deeper reasoning (3 steps)
python train_local_7900xt.py --questions 100 --max-steps 3
```

## Monitoring

### Check VRAM Usage

**Windows** (AMD):
```powershell
# Use AMD Radeon Software
# Or GPU-Z for detailed monitoring
```

**Linux**:
```bash
radeontop
```

### View Training Progress

Training logs show:
```
Batch 1: Consciousness=0.654, Integration=0.823, Avg Time=45.2s/question
Checkpoint saved: training_checkpoints/local/checkpoint_20.json
```

### Check Checkpoints

```bash
# View latest checkpoint
cat training_checkpoints/local/checkpoint_*.json | tail -1
```

## Comparison: Local vs Cloud

| Metric | Local (7900XT) | Cloud (Hyperbolic) |
|--------|----------------|-------------------|
| **Cost** | FREE | $0.02/question |
| **Speed** | 60-120s/question | 150-180s/question |
| **Quality** | Good (3.8B model) | Better (20B model) |
| **Neurons** | 8 | 8 or 64 |
| **VRAM** | 20GB (yours) | Unlimited (cloud) |
| **Privacy** | 100% local | Data sent to API |
| **Availability** | Always | Requires internet |

## Advantages of Local Training

✅ **Zero Cost**: No API fees, unlimited training
✅ **Privacy**: All data stays on your machine
✅ **Speed**: Faster inference with local GPU
✅ **Control**: Full control over model and parameters
✅ **Offline**: No internet required
✅ **Experimentation**: Try different configs freely

## Troubleshooting

### LM Studio Server Not Starting

1. Check if port 1234 is already in use
2. Try restarting LM Studio
3. Verify model is fully downloaded

### Out of Memory Errors

1. Reduce neurons to 4 or 6
2. Use Q4 quantization (not Q8 or FP16)
3. Close other GPU-intensive applications
4. Reduce context length to 2048

### Slow Performance

1. Verify GPU is being used (check LM Studio logs)
2. Update AMD drivers
3. Reduce max_steps to 1
4. Use smaller batch sizes

### Connection Errors

```python
# Verify LM Studio is running
curl http://localhost:1234/v1/models

# Check if model is loaded
# Should return model information
```

## Next Steps

After local training completes:

1. **Compare Results**: Check `training_checkpoints/local/` for metrics
2. **Run Deep Reasoning Test**: Test the trained brain
3. **Analyze Performance**: Compare local vs cloud training
4. **Scale Up**: Try 500+ questions for deeper training

## Advanced: Hybrid Training

Run **both** local and cloud training simultaneously:

**Terminal 1** (Cloud - ongoing):
```bash
# Already running: 500 questions on Hyperbolic
```

**Terminal 2** (Local - new):
```bash
# Start local training in parallel
python train_local_7900xt.py --questions 100
```

Compare results to see which approach works better for your use case!

## Files Created

- `config_local_7900xt.py` - Local configuration
- `train_local_7900xt.py` - Local training script
- `training_checkpoints/local/` - Local checkpoints
- `training_checkpoints/local_memory.pkl` - Local persistent memory

## Support

For issues or questions:
1. Check LM Studio logs
2. Verify AMD drivers are up to date
3. Monitor VRAM usage during training
4. Review checkpoint files for progress

---

**Ready to train locally!** 🚀

Start with a small test:
```bash
python train_local_7900xt.py --questions 10 --max-steps 2
```
