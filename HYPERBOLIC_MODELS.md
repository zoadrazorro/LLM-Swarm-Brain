# Hyperbolic API Model Reference

## Available Models for LLM-Swarm-Brain

### Fast Mode (Recommended for 128-Neuron)

| Model Name | Description | Speed | Cost | Best For |
|------------|-------------|-------|------|----------|
| `openai/gpt-oss-20b` | **GPT-OSS 20B** | Very Fast | Very Low | **128-neuron tests** ✅ |
| `meta-llama/Meta-Llama-3.1-70B-Instruct` | Llama 3.1 70B | Fast | Low | 64-neuron tests |
| `meta-llama/Meta-Llama-3.1-405B-Instruct` | Llama 3.1 405B | Slow | Medium | Maximum quality |

## Recommended Configurations

### 128-Neuron Deep Reasoning (Fast Mode)

```bash
# Use GPT-OSS 20B for fast, cost-effective 128-neuron processing
python deep_reasoning_test.py \
  --neurons 128 \
  --api-provider hyperbolic \
  --hyperbolic-model openai/gpt-oss-20b \
  --max-steps 5
```

**Why GPT-OSS for 128 neurons?**
- ⚡ **Very Fast**: ~1-2 seconds per API call
- 💰 **Very Cheap**: ~$0.10 per 1M tokens (10× cheaper than 405B)
- 🧠 **Good Quality**: 20B parameters is sufficient for specialized neuron roles
- 📊 **Scalable**: Can handle 128 neurons × 5 steps without timeout

**Estimated Cost:**
- 128 neurons × 5 steps × 2 questions = ~1,280 API calls
- ~1,280 calls × 1,000 tokens = 1.28M tokens
- Cost: **~$0.13** (vs $2.00+ with Llama 405B)

### 64-Neuron Tests (Balanced)

```bash
# Use Llama 3.1 70B for balanced performance
python expanded_inference_test_philosophy.py \
  --use-api \
  --api-provider hyperbolic \
  --use-64-neurons \
  --hyperbolic-model meta-llama/Meta-Llama-3.1-70B-Instruct
```

### 8-Neuron Tests (Maximum Quality)

```bash
# Use Llama 3.1 405B for best quality
python expanded_inference_test_philosophy.py \
  --use-api \
  --api-provider hyperbolic \
  --hyperbolic-model meta-llama/Meta-Llama-3.1-405B-Instruct
```

## Model Comparison

### Performance Characteristics

**openai/gpt-oss-20b** ✅ **Recommended for 128-neuron**
- Size: 20 billion parameters
- Speed: Very fast (~1-2s per call)
- Cost: Very low (~$0.10/1M tokens)
- Quality: Good for specialized tasks
- Best for: High-volume, multi-neuron architectures

**meta-llama/Meta-Llama-3.1-70B-Instruct**
- Size: 70 billion parameters
- Speed: Fast (~2-3s per call)
- Cost: Low (~$0.20/1M tokens)
- Quality: Very good
- Best for: 64-neuron balanced performance

**meta-llama/Meta-Llama-3.1-405B-Instruct**
- Size: 405 billion parameters
- Speed: Slow (~5-8s per call)
- Cost: Medium (~$0.40/1M tokens)
- Quality: Excellent
- Best for: 8-neuron maximum quality

## Cost Estimates

### 128-Neuron Deep Reasoning Test (2 Questions, 5 Steps)

| Model | API Calls | Est. Tokens | Est. Cost | Duration |
|-------|-----------|-------------|-----------|----------|
| **GPT-OSS 20B** | ~1,280 | 1.28M | **$0.13** | 30-45 min |
| Llama 70B | ~1,280 | 1.28M | $0.26 | 45-60 min |
| Llama 405B | ~1,280 | 1.28M | $0.51 | 90-120 min |

### 64-Neuron Philosophy Test (100 Questions)

| Model | API Calls | Est. Tokens | Est. Cost | Duration |
|-------|-----------|-------------|-----------|----------|
| **GPT-OSS 20B** | ~3,000 | 3M | **$0.30** | 1-2 hours |
| Llama 70B | ~3,000 | 3M | $0.60 | 2-3 hours |
| Llama 405B | ~3,000 | 3M | $1.20 | 4-6 hours |

### 8-Neuron Philosophy Test (100 Questions)

| Model | API Calls | Est. Tokens | Est. Cost | Duration |
|-------|-----------|-------------|-----------|----------|
| GPT-OSS 20B | ~500 | 500K | $0.05 | 15-20 min |
| Llama 70B | ~500 | 500K | $0.10 | 20-30 min |
| **Llama 405B** | ~500 | 500K | **$0.20** | 40-60 min |

## Usage Examples

### Python API

```python
from llm_swarm_brain import PhiBrain
from llm_swarm_brain import config_128

# 128-neuron with GPT-OSS (fast mode)
config = config_128.BrainConfig()
config.hyperbolic_model_name = "openai/gpt-oss-20b"

brain = PhiBrain(
    config=config,
    use_api=True,
    api_provider="hyperbolic",
    api_key="your_hyperbolic_api_key"
)

# Process complex question
result = brain.think(
    "Explain the hard problem of consciousness",
    max_steps=5
)
```

### Command Line

```bash
# Fast 128-neuron test
python deep_reasoning_test.py \
  --neurons 128 \
  --api-provider hyperbolic \
  --hyperbolic-model openai/gpt-oss-20b

# Quality 8-neuron test
python expanded_inference_test_philosophy.py \
  --use-api \
  --api-provider hyperbolic \
  --hyperbolic-model meta-llama/Meta-Llama-3.1-405B-Instruct
```

## API Request Format

### GPT-OSS 20B Example

```python
import requests

url = "https://api.hyperbolic.xyz/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}
data = {
    "messages": [{
        "role": "user",
        "content": "What is consciousness?"
    }],
    "model": "openai/gpt-oss-20b",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.8
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

## Troubleshooting

### 502 Bad Gateway Errors

If you encounter 502 errors:
1. **Retry**: The system automatically retries failed requests
2. **Switch models**: Try GPT-OSS instead of Llama 405B
3. **Reduce load**: Lower max_steps or use fewer neurons
4. **Check status**: Visit [Hyperbolic Status](https://status.hyperbolic.xyz/)

### Timeout Errors

For timeout issues:
1. **Use faster model**: Switch to GPT-OSS or Llama 70B
2. **Reduce max_tokens**: Lower from 2048 to 1024
3. **Increase timeout**: Set longer timeout in neuron config

### Rate Limiting

If you hit rate limits:
1. **Add delays**: Implement exponential backoff
2. **Reduce concurrency**: Process fewer neurons simultaneously
3. **Upgrade plan**: Contact Hyperbolic for higher limits

## Recommendations

### For 128-Neuron Architecture

**Use GPT-OSS 20B** ✅
- Fast enough to complete in reasonable time
- Cheap enough for experimentation
- Good enough quality for specialized neuron roles
- Reliable (fewer 502 errors than 405B)

### For 64-Neuron Architecture

**Use Llama 70B** ✅
- Good balance of speed and quality
- Reasonable cost
- Better than GPT-OSS for complex reasoning

### For 8-Neuron Architecture

**Use Llama 405B** ✅
- Maximum quality
- Acceptable speed with only 8 neurons
- Worth the cost for best results

## Future Models

Hyperbolic frequently adds new models. Check their [API documentation](https://docs.hyperbolic.xyz/) for the latest available models.

---

**Last Updated**: November 10, 2025  
**API Endpoint**: https://api.hyperbolic.xyz/v1/chat/completions  
**Documentation**: https://docs.hyperbolic.xyz/
