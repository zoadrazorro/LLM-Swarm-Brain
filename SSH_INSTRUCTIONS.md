# SSH Setup and Inference Instructions

## Quick Start

### 1. SSH into the Server

```bash
ssh ubuntu@147.185.41.15
```

If you need to use a specific SSH key:
```bash
ssh -i /path/to/your/key.pem ubuntu@147.185.41.15
```

### 2. Run the Setup Script

Once connected, run the automated setup:

```bash
# Download and run the setup script
wget https://raw.githubusercontent.com/zoadrazorro/LLM-Swarm-Brain/main/remote_setup.sh
bash remote_setup.sh
```

Or if you prefer to clone first:

```bash
git clone https://github.com/zoadrazorro/LLM-Swarm-Brain.git
cd LLM-Swarm-Brain
bash remote_setup.sh
```

### 3. Run Inference Tests

After setup completes:

```bash
# Activate the virtual environment
source venv/bin/activate

# Quick test (no model loading, 1 question per level)
python inference_test.py --quick

# Full test without loading models (architecture test)
python inference_test.py

# Full test with actual models (requires GPUs)
python inference_test.py --load-models

# Custom test (2 questions per level, with models)
python inference_test.py --load-models --max-per-level 2
```

## Detailed Instructions

### Manual Setup (Alternative to Script)

If you prefer manual setup:

```bash
# 1. Update system
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv python3-pip git

# 2. Clone repository
git clone https://github.com/zoadrazorro/LLM-Swarm-Brain.git
cd LLM-Swarm-Brain

# 3. Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Check GPU Availability

```bash
# Check NVIDIA GPUs
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch GPU support (after activating venv)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Inference Test Options

The `inference_test.py` script tests the model with 8 levels of complexity:

1. **Level 1 - Basic Comprehension**: Simple factual questions
2. **Level 2 - Simple Reasoning**: Basic logical deduction
3. **Level 3 - Pattern Recognition**: Sequence completion
4. **Level 4 - Abstract Reasoning**: Conceptual understanding
5. **Level 5 - Counterfactual Thinking**: Hypothetical scenarios
6. **Level 6 - Meta-Cognitive Reasoning**: Self-reflection
7. **Level 7 - Philosophical & Existential**: Deep questions
8. **Level 8 - Multi-Step Problem Solving**: Complex puzzles

#### Command Line Options:

```bash
# Quick test (recommended first run)
python inference_test.py --quick

# Test without loading models (tests architecture only)
python inference_test.py

# Test with actual models (requires 8x H100 GPUs or equivalent)
python inference_test.py --load-models

# Limit questions per level
python inference_test.py --max-per-level 2

# Full test with models and limited questions
python inference_test.py --load-models --max-per-level 1
```

### View Results

Results are automatically saved to JSON files with timestamps:

```bash
# List result files
ls -lh inference_results_*.json

# View latest results
cat $(ls -t inference_results_*.json | head -1) | python -m json.tool | less

# Quick summary
python -c "import json; data=json.load(open('$(ls -t inference_results_*.json | head -1)')); print(f\"Total Levels: {len(data['levels'])}\"); print(f\"Timestamp: {data['test_info']['timestamp']}\")"
```

### Run Other Examples

```bash
# Basic usage example
python examples/basic_usage.py

# Enhancements demo
python examples/enhancements_demo.py
```

## Troubleshooting

### SSH Connection Issues

```bash
# Test connection
ping 147.185.41.15

# Verbose SSH for debugging
ssh -v ubuntu@147.185.41.15

# Check if port 22 is open
nc -zv 147.185.41.15 22
```

### GPU/CUDA Issues

```bash
# Check if NVIDIA drivers are installed
lsmod | grep nvidia

# Install NVIDIA drivers (if needed)
sudo apt-get install -y nvidia-driver-535

# Reboot after driver installation
sudo reboot
```

### Memory Issues

If you encounter OOM (Out of Memory) errors:

```bash
# Check available memory
free -h

# Check GPU memory
nvidia-smi

# Run with fewer questions
python inference_test.py --load-models --max-per-level 1

# Or test without loading models
python inference_test.py --quick
```

### Python/Dependency Issues

```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.9+
```

## System Requirements

### Minimum (Architecture Testing)
- Python 3.9+
- 8 GB RAM
- No GPU required (use `--quick` or without `--load-models`)

### Recommended (Full Inference)
- Python 3.9+
- 8× NVIDIA H100 SXM5 GPUs (80 GB VRAM each)
- 64 GB System RAM
- CUDA 12.0+
- ~100 GB disk space

### Alternative GPU Configurations
- 8× A100 (80 GB)
- 16× A100 (40 GB)
- Other high-VRAM GPU combinations

## Performance Expectations

### Without Models (`--quick` or no `--load-models`)
- Tests architecture and connectivity
- Fast execution (~1-2 minutes)
- No GPU required
- Useful for development/testing

### With Models (`--load-models`)
- Full inference with 64 Phi-3 neurons
- Requires significant GPU resources
- Execution time: 5-15 minutes (depends on questions)
- Real reasoning and consciousness metrics

## Next Steps

After successful inference:

1. **Analyze Results**: Review the JSON output files
2. **Tune Parameters**: Modify `BrainConfig` in `inference_test.py`
3. **Custom Questions**: Add your own questions to test specific capabilities
4. **Explore Examples**: Check out other examples in the `examples/` directory
5. **Monitor Performance**: Use `nvidia-smi -l 1` to watch GPU usage during inference

## Support

For issues or questions:
- GitHub Issues: https://github.com/zoadrazorro/LLM-Swarm-Brain/issues
- Check README.md for detailed architecture information
- Review examples/ directory for usage patterns
