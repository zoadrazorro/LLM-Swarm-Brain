# Dual-GPU Quick Start Guide

## Step-by-Step Setup (5 minutes)

### 1. Start First LM Studio Instance (GPU 0)

```powershell
.\start_lmstudio_gpu0.bat
```

In LM Studio GUI:
- Load **Phi-4 Q4_K_M** model
- Go to **"Local Server"** tab
- Set port to **1234**
- Click **"Start Server"**

### 2. Start Second LM Studio Instance (GPU 1)

```powershell
.\start_lmstudio_gpu1.bat
```

In second LM Studio GUI:
- Load **Phi-4 Q4_K_M** model
- Go to **"Local Server"** tab
- Set port to **1235**
- Click **"Start Server"**

### 3. Verify Both Servers

```powershell
.\verify_dual_gpu.ps1
```

Should show:
```
✓ GPU 0 server is running
✓ GPU 1 server is running
✓ BOTH GPUS READY!
```

### 4. Start Training

```powershell
python train_local_7900xt.py --questions 100 --max-steps 2
```

## Neuron Distribution

**GPU 0 (Port 1234)** - 2 neurons:
- Neuron 0: Perception & Reasoning
- Neuron 1: Attention & Memory

**GPU 1 (Port 1235)** - 2 neurons:
- Neuron 2: Creative & Analytical
- Neuron 3: Synthesis & Meta-Cognitive

## Quick Commands

**Setup**:
```powershell
.\setup_dual_gpu.ps1          # Interactive setup guide
.\start_lmstudio_gpu0.bat     # Start GPU 0 server
.\start_lmstudio_gpu1.bat     # Start GPU 1 server
.\verify_dual_gpu.ps1         # Check both servers
```

**Training**:
```powershell
# Quick test (10 questions)
python train_local_7900xt.py --questions 10

# Full training (100 questions)
python train_local_7900xt.py --questions 100

# Extended training (500 questions)
python train_local_7900xt.py --questions 500
```

**Verification**:
```powershell
# Test GPU 0
curl http://localhost:1234/v1/models

# Test GPU 1
curl http://localhost:1235/v1/models
```

## Troubleshooting

### "Port already in use"
- Close other LM Studio instances
- Check Task Manager for orphaned processes

### "Only one GPU detected"
- Make sure you ran both .bat files
- Check that HIP_VISIBLE_DEVICES is set correctly
- Verify in AMD Radeon Software that both GPUs are active

### "Server not responding"
- Wait 10-20 seconds after starting LM Studio
- Check that model is fully loaded in GUI
- Verify "Server Running" indicator is green

## Performance

**Expected Speed**:
- Per question: 40-60 seconds
- 100 questions: ~1.5 hours
- Both GPUs at ~80% utilization

**VRAM Usage**:
- GPU 0: ~8GB (Phi-4 Q4)
- GPU 1: ~8GB (Phi-4 Q4)
- Total: ~16GB / 40GB available

## Files Created

- `start_lmstudio_gpu0.bat` - Launch GPU 0 server
- `start_lmstudio_gpu1.bat` - Launch GPU 1 server
- `setup_dual_gpu.ps1` - Interactive setup
- `verify_dual_gpu.ps1` - Verify both servers
- `train_local_7900xt.py` - Training script (auto-configured for dual-GPU)

---

**You're ready!** Just run the .bat files and start training! 🚀🚀
