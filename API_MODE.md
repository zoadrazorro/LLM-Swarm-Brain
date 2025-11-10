# API Mode: Running LLM-Swarm-Brain with Hyperbolic API

Instead of running models locally on GPUs, you can use the Hyperbolic API to run the MoE architecture with **Llama 3.1 405B** - a much larger and more capable model than what fits on local hardware.

## Benefits of API Mode

### Advantages:
- **Larger Model**: Llama 3.1 405B (405 billion parameters) vs Qwen2.5-72B (72 billion)
- **No GPU Required**: Runs on any machine with internet connection
- **No Model Downloads**: No need to download 80GB+ models
- **Instant Start**: No model loading time (~4-8 minutes saved)
- **Cost Effective**: Pay per use instead of GPU rental
- **Latest Models**: Access to cutting-edge models via API

### Trade-offs:
- **API Latency**: Network round-trip adds ~1-3 seconds per request
- **API Costs**: ~$0.40 per 1M tokens (check Hyperbolic pricing)
- **Rate Limits**: Subject to API rate limiting
- **Internet Required**: Must have stable internet connection

## Setup

### 1. Get API Key

Sign up at [Hyperbolic](https://app.hyperbolic.xyz/) and get your API key.

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
HYPERBOLIC_API_KEY=your_actual_api_key_here
```

### 3. Install Dependencies

The API mode requires the `requests` library (already in requirements.txt):

```bash
pip install requests python-dotenv
```

## Usage

### Philosophy Test with API

```bash
# Run with API (Llama 3.1 405B)
python inference_test_philosophy.py --use-api

# Quick test with API
python inference_test_philosophy.py --use-api --quick

# Specify API key directly (optional)
python inference_test_philosophy.py --use-api --api-key your_key_here

# Custom test with API
python inference_test_philosophy.py --use-api --start-level 5 --end-level 8 --questions-per-level 3
```

### Python Code

```python
from llm_swarm_brain import PhiBrain, BrainConfig

# Initialize with API mode
config = BrainConfig()
config.model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct"

brain = PhiBrain(
    config=config,
    use_api=True,
    api_key="your_api_key_here"  # Or set HYPERBOLIC_API_KEY env var
)

# Process input
result = brain.think(
    input_text="What is the nature of consciousness?",
    max_steps=4,
    use_memory=True,
    use_global_workspace=True
)

print(result["final_output"])
```

## Architecture

### API-Based MoE

Each of the 8 expert neurons makes independent API calls to Hyperbolic:

```
┌─────────────────────────────────────────────────┐
│           LLM-Swarm-Brain (Local)               │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Expert 0 │  │ Expert 1 │  │ Expert 2 │ ... │
│  │Perception│  │Attention │  │ Memory   │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
│       │             │             │            │
└───────┼─────────────┼─────────────┼────────────┘
        │             │             │
        └─────────────┴─────────────┘
                      │
                 [Internet]
                      │
        ┌─────────────┴─────────────┐
        │   Hyperbolic API Endpoint  │
        │  Llama 3.1 405B-Instruct   │
        └────────────────────────────┘
```

### Request Format

Each neuron sends requests in this format:

```python
{
    "messages": [
        {
            "role": "system",
            "content": "You are the PERCEPTION EXPERT. Your role is to analyze and interpret raw input data..."
        },
        {
            "role": "user",
            "content": "What is the nature of consciousness?"
        }
    ],
    "model": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9
}
```

## Performance Expectations

### Timing Estimates (API Mode)

| Metric | Local (Qwen 72B) | API (Llama 405B) |
|--------|------------------|------------------|
| Model Loading | 4-8 minutes | 0 seconds |
| Per Expert Call | 2-5 seconds | 3-8 seconds |
| Simple Query (3 experts) | ~10 seconds | ~15-20 seconds |
| Complex Query (8 experts) | ~25 seconds | ~40-60 seconds |
| Full 40Q Test | ~15-30 min | ~30-50 min |

### Cost Estimates

Assuming $0.40 per 1M tokens (check current Hyperbolic pricing):

| Test Type | Tokens Used | Estimated Cost |
|-----------|-------------|----------------|
| Quick Test (8Q) | ~50K tokens | ~$0.02 |
| Full Test (40Q) | ~250K tokens | ~$0.10 |
| Single Query | ~5-10K tokens | ~$0.002-0.004 |

**Note**: Llama 3.1 405B typically produces higher quality responses than smaller models, potentially requiring fewer retries.

## Error Handling

The API neurons include automatic fallback:

1. **API Call Fails**: Falls back to simulation mode
2. **Timeout**: 60-second timeout per request
3. **Rate Limit**: Logged as error, returns simulation response
4. **No API Key**: Automatically uses simulation mode

### Monitoring API Usage

Check neuron statistics to monitor API performance:

```python
# Get stats for all neurons
for neuron in brain.orchestrator.neurons.values():
    stats = neuron.get_stats()
    print(f"{stats['neuron_id']}:")
    print(f"  API Calls: {stats['total_api_calls']}")
    print(f"  API Errors: {stats['total_api_errors']}")
    print(f"  Success Rate: {stats['api_success_rate']:.1%}")
```

## Comparison: Local vs API

### Local Mode (Qwen2.5-72B)

**Pros:**
- Faster inference (no network latency)
- No per-use costs
- No rate limits
- Works offline

**Cons:**
- Requires 8× H100 GPUs (~$20-40/hour)
- 80GB model download
- 4-8 minute loading time
- Limited to 72B model size

### API Mode (Llama 3.1 405B)

**Pros:**
- 5.6× larger model (405B vs 72B)
- No GPU required
- Instant start
- Pay-per-use pricing
- Access to latest models

**Cons:**
- Network latency (~1-3s per call)
- API costs (~$0.002-0.004 per query)
- Requires internet
- Subject to rate limits

## Best Practices

### 1. Batch Processing
Process multiple questions in one session to amortize initialization overhead.

### 2. Error Handling
Always check API success rates in neuron statistics.

### 3. Cost Management
- Use `--quick` mode for testing
- Limit `--questions-per-level` for cost control
- Monitor token usage

### 4. Performance Optimization
- Reduce `max_tokens` if responses are too long
- Adjust `temperature` for more focused responses
- Use `max_steps` to limit propagation depth

### 5. Fallback Strategy
Keep simulation mode as fallback:
```python
brain = PhiBrain(use_api=True, api_key=api_key)
# If API fails, neurons automatically fall back to simulation
```

## Troubleshooting

### "No API key provided"
- Set `HYPERBOLIC_API_KEY` environment variable
- Or pass `api_key` parameter to `PhiBrain()`

### "API request timed out"
- Check internet connection
- Increase timeout in `neuron_api.py` (default: 60s)
- Try again (may be temporary API issue)

### "API request failed: 429"
- Rate limit exceeded
- Wait a few seconds and retry
- Reduce concurrent requests

### High API costs
- Use `--quick` mode for testing
- Reduce `max_tokens` in config
- Limit number of questions per test

## Environment Variables

```bash
# Required for API mode
HYPERBOLIC_API_KEY=your_api_key_here

# Optional: Override API endpoint
HYPERBOLIC_API_URL=https://api.hyperbolic.xyz/v1/chat/completions
```

## Migration Guide

### From Local to API

```python
# Before (Local)
brain = PhiBrain(
    config=config,
    load_models=True
)

# After (API)
brain = PhiBrain(
    config=config,
    use_api=True,
    api_key="your_key"
)
```

### From API to Local

```python
# Before (API)
brain = PhiBrain(
    config=config,
    use_api=True
)

# After (Local)
brain = PhiBrain(
    config=config,
    load_models=True
)
```

## Future Enhancements

- [ ] Support for multiple API providers (OpenAI, Anthropic, etc.)
- [ ] Automatic retry with exponential backoff
- [ ] Request batching for cost optimization
- [ ] Caching for repeated queries
- [ ] Async API calls for parallel expert execution
- [ ] Cost tracking and budgeting
- [ ] Model selection per expert (different models for different roles)
