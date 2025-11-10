# Gemini Model Reference

## Available Gemini Models for LLM-Swarm-Brain

### Latest Models (December 2024)

| Model Name | Description | Context | Speed | Cost |
|------------|-------------|---------|-------|------|
| `gemini-exp-1206` | **Latest Experimental** (Dec 2024) | 2M tokens | Fast | Low |
| `gemini-2.0-flash-exp` | Gemini 2.0 Flash Experimental | 1M tokens | Very Fast | Very Low |
| `gemini-1.5-pro` | Gemini 1.5 Pro (Stable) | 2M tokens | Medium | Medium |
| `gemini-1.5-flash` | Gemini 1.5 Flash (Stable) | 1M tokens | Fast | Low |

### Recommended Models

**For Deep Reasoning (128-neuron):**
```bash
python deep_reasoning_test.py --gemini-model gemini-exp-1206 --neurons 128
```

**For Fast Testing (8-neuron):**
```bash
python expanded_inference_test_philosophy.py --use-api --api-provider gemini --gemini-model gemini-2.0-flash-exp
```

**For Production (Stable):**
```bash
python deep_reasoning_test.py --gemini-model gemini-1.5-pro --neurons 64
```

## Usage

### Command Line

```bash
# Use latest experimental model (default)
python deep_reasoning_test.py --api-provider gemini

# Specify custom model
python deep_reasoning_test.py --api-provider gemini --gemini-model gemini-2.0-flash-exp

# Use stable production model
python deep_reasoning_test.py --api-provider gemini --gemini-model gemini-1.5-pro
```

### Python API

```python
from llm_swarm_brain import PhiBrain, BrainConfig

# Method 1: Set via config
config = BrainConfig()
config.gemini_model_name = "gemini-exp-1206"

brain = PhiBrain(
    config=config,
    use_api=True,
    api_provider="gemini"
)

# Method 2: Use default (gemini-exp-1206)
brain = PhiBrain(
    use_api=True,
    api_provider="gemini"
)
```

## Model Comparison

### Performance Characteristics

**gemini-exp-1206** (Recommended for 128-neuron)
- ✅ Latest capabilities
- ✅ Best reasoning quality
- ✅ 2M token context
- ⚠️ Experimental (may change)

**gemini-2.0-flash-exp** (Recommended for 8/64-neuron)
- ✅ Very fast responses
- ✅ Low cost
- ✅ Good for high-volume testing
- ⚠️ Experimental

**gemini-1.5-pro** (Recommended for production)
- ✅ Stable API
- ✅ Excellent reasoning
- ✅ 2M token context
- ⚠️ Higher cost than Flash

**gemini-1.5-flash** (Recommended for budget)
- ✅ Stable API
- ✅ Very low cost
- ✅ Fast responses
- ⚠️ Less capable than Pro

## Cost Estimates

### Per 100-Question Philosophy Test

| Model | 8-Neuron | 64-Neuron | 128-Neuron |
|-------|----------|-----------|------------|
| gemini-exp-1206 | $0.10 | $0.80 | $1.60 |
| gemini-2.0-flash-exp | $0.08 | $0.65 | $1.30 |
| gemini-1.5-pro | $0.35 | $2.80 | $5.60 |
| gemini-1.5-flash | $0.08 | $0.65 | $1.30 |

### Per Deep Reasoning Test (2 Questions, 5 Steps)

| Model | 8-Neuron | 64-Neuron | 128-Neuron |
|-------|----------|-----------|------------|
| gemini-exp-1206 | $0.05 | $0.30 | $0.80 |
| gemini-2.0-flash-exp | $0.04 | $0.25 | $0.65 |
| gemini-1.5-pro | $0.15 | $1.20 | $3.20 |
| gemini-1.5-flash | $0.04 | $0.25 | $0.65 |

## Troubleshooting

### Model Not Found Error

If you get an error like:
```
google.api_core.exceptions.InvalidArgument: 400 Model not found
```

**Solutions:**
1. Check the model name spelling
2. Ensure you have the latest `google-generativeai` package:
   ```bash
   pip install --upgrade google-generativeai
   ```
3. Try a stable model like `gemini-1.5-pro`
4. Check [Google AI Studio](https://aistudio.google.com/) for available models

### API Key Issues

Make sure your `GOOGLE_API_KEY` is set in `.env`:
```bash
GOOGLE_API_KEY=your_actual_api_key_here
```

Get your API key from: https://aistudio.google.com/app/apikey

## Future Models

When Gemini 2.5 Pro is officially released, you can use it by specifying:
```bash
python deep_reasoning_test.py --gemini-model gemini-2.5-pro
```

The system is designed to work with any Gemini model name, so new models will work automatically!
