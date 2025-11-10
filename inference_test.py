"""
Inference Test Script for LLM-Swarm-Brain
Tests the model's reasoning capabilities with probing questions of increasing complexity
"""

import logging
import sys
import os
import json
from datetime import datetime
from typing import Any
from dataclasses import asdict, is_dataclass

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - numpy is expected but handle gracefully
    np = None

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from llm_swarm_brain import PhiBrain, BrainConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# Probing questions organized by complexity level
PROBING_QUESTIONS = {
    "Level 1 - Basic Comprehension": [
        "What is 2 + 2?",
        "What color is the sky on a clear day?",
        "Is water wet or dry?",
    ],
    
    "Level 2 - Simple Reasoning": [
        "If all cats are animals, and Fluffy is a cat, what can we conclude about Fluffy?",
        "If it's raining outside, should you bring an umbrella? Why?",
        "Which is heavier: a pound of feathers or a pound of bricks?",
    ],
    
    "Level 3 - Pattern Recognition": [
        "What comes next in this sequence: 2, 4, 8, 16, ?",
        "Complete the pattern: A, C, E, G, ?",
        "If Monday is to Tuesday as Wednesday is to what?",
    ],
    
    "Level 4 - Abstract Reasoning": [
        "What is the relationship between cause and effect?",
        "How does emergence arise in complex systems?",
        "What is the difference between correlation and causation?",
    ],
    
    "Level 5 - Counterfactual Thinking": [
        "If gravity suddenly stopped working, what would happen to Earth's atmosphere?",
        "What would be different if humans had evolved with photosynthetic skin?",
        "If time travel were possible, what paradoxes might arise?",
    ],
    
    "Level 6 - Meta-Cognitive Reasoning": [
        "How do you know what you know?",
        "What are the limits of logical reasoning?",
        "Can a system fully understand itself? Why or why not?",
    ],
    
    "Level 7 - Philosophical & Existential": [
        "What is the nature of consciousness?",
        "Is free will compatible with determinism?",
        "What does it mean for something to exist?",
    ],
    
    "Level 8 - Multi-Step Problem Solving": [
        "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons?",
        "Three people need to cross a bridge at night with one flashlight. Person A takes 1 minute, B takes 2 minutes, C takes 5 minutes. The bridge can hold 2 people max. What's the fastest way to get everyone across?",
        "A farmer needs to transport a fox, a chicken, and a bag of grain across a river. The boat can only hold the farmer and one item. If left alone, the fox will eat the chicken, and the chicken will eat the grain. How does the farmer get everything across safely?",
    ],
}


def _make_serializable(value: Any) -> Any:
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(value, dict):
        return {k: _make_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_make_serializable(item) for item in value]
    if isinstance(value, set):
        return [_make_serializable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return _make_serializable(asdict(value))
    if np is not None:
        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    if hasattr(value, "isoformat") and callable(getattr(value, "isoformat")):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if hasattr(value, "__dict__") and value.__dict__:
        return _make_serializable(vars(value))
    return value


def save_results(results, filename="inference_results.json"):
    """Save inference results to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{filename.split('.')[0]}_{timestamp}.json"

    serializable_results = _make_serializable(results)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_file}")
    return output_file


def analyze_response(result):
    """Analyze the brain's response and extract key metrics"""
    analysis = {
        "consciousness_level": result.get('consciousness_level', 0.0),
        "neurons_fired": 0,
        "broadcast_count": 0,
        "integration_phi": result.get('integration', {}).get('phi', 0.0),
        "response_summary": ""
    }
    
    # Count neurons fired
    if result.get('neural_processing', {}).get('steps'):
        final_step = result['neural_processing']['steps'][-1]
        analysis["neurons_fired"] = len(final_step.get('fired_neurons', []))
    
    # Count broadcasts
    if result.get('global_workspace', {}).get('broadcasts'):
        analysis["broadcast_count"] = len(result['global_workspace']['broadcasts'])
        
        # Get first broadcast as summary
        if result['global_workspace']['broadcasts']:
            first_broadcast = result['global_workspace']['broadcasts'][0]
            analysis["response_summary"] = first_broadcast.get('content', '')[:200]
    
    return analysis


def run_inference_test(load_models=False, max_questions_per_level=None):
    """
    Run inference test with probing questions
    
    Args:
        load_models: Whether to load actual models (requires GPUs)
        max_questions_per_level: Limit questions per level (None = all)
    """
    print("=" * 80)
    print("LLM-SWARM-BRAIN INFERENCE TEST")
    print("=" * 80)
    print(f"\nTest Configuration:")
    print(f"  - Load Models: {load_models}")
    print(f"  - Questions per Level: {max_questions_per_level or 'All'}")
    print(f"  - Total Complexity Levels: {len(PROBING_QUESTIONS)}")
    
    # Create brain configuration
    config = BrainConfig(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        max_tokens=512,
        temperature=0.7,
        activation_threshold=0.5,
    )
    
    print("\n[1] Initializing PhiBrain...")
    brain = PhiBrain(config=config, load_models=load_models)
    print(f"✓ Brain initialized with {brain.get_summary()}")
    
    # Store all results
    all_results = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "load_models": load_models,
            "config": {
                "model_name": config.model_name,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "activation_threshold": config.activation_threshold,
            }
        },
        "levels": {}
    }
    
    print("\n[2] Running inference tests...\n")
    
    # Process each complexity level
    for level_name, questions in PROBING_QUESTIONS.items():
        print("\n" + "=" * 80)
        print(f"  {level_name}")
        print("=" * 80)
        
        level_results = []
        
        # Limit questions if specified
        test_questions = questions[:max_questions_per_level] if max_questions_per_level else questions
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[Q{i}] {question}")
            
            try:
                # Process through brain
                result = brain.think(
                    input_text=question,
                    max_steps=4,
                    use_memory=True,
                    use_global_workspace=True
                )
                
                # Analyze response
                analysis = analyze_response(result)
                
                # Display key metrics
                print(f"  → Consciousness: {analysis['consciousness_level']:.3f}")
                print(f"  → Neurons Fired: {analysis['neurons_fired']}")
                print(f"  → Broadcasts: {analysis['broadcast_count']}")
                print(f"  → Phi (Φ): {analysis['integration_phi']:.3f}")
                
                if analysis['response_summary']:
                    print(f"  → Response: {analysis['response_summary'][:150]}...")
                
                # Store results
                level_results.append({
                    "question": question,
                    "analysis": analysis,
                    "full_result": result
                })
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                level_results.append({
                    "question": question,
                    "error": str(e)
                })
        
        all_results["levels"][level_name] = level_results
    
    # Final summary
    print("\n" + "=" * 80)
    print("INFERENCE TEST COMPLETE")
    print("=" * 80)
    
    # Calculate aggregate statistics
    total_questions = sum(len(results) for results in all_results["levels"].values())
    successful_questions = sum(
        1 for level_results in all_results["levels"].values()
        for result in level_results if "error" not in result
    )
    
    print(f"\nSummary:")
    print(f"  - Total Questions: {total_questions}")
    print(f"  - Successful: {successful_questions}")
    print(f"  - Failed: {total_questions - successful_questions}")
    
    # Average metrics across all questions
    all_consciousness = []
    all_neurons = []
    all_phi = []
    
    for level_results in all_results["levels"].values():
        for result in level_results:
            if "analysis" in result:
                all_consciousness.append(result["analysis"]["consciousness_level"])
                all_neurons.append(result["analysis"]["neurons_fired"])
                all_phi.append(result["analysis"]["integration_phi"])
    
    if all_consciousness:
        print(f"\nAverage Metrics:")
        print(f"  - Consciousness Level: {sum(all_consciousness)/len(all_consciousness):.3f}")
        print(f"  - Neurons Fired: {sum(all_neurons)/len(all_neurons):.1f}")
        print(f"  - Integration Phi: {sum(all_phi)/len(all_phi):.3f}")
    
    # Save results
    output_file = save_results(all_results)
    print(f"\n✓ Detailed results saved to: {output_file}")
    
    return all_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-Swarm-Brain Inference Test")
    parser.add_argument(
        "--load-models",
        action="store_true",
        help="Load actual models (requires GPUs with sufficient VRAM)"
    )
    parser.add_argument(
        "--max-per-level",
        type=int,
        default=None,
        help="Maximum questions per complexity level (default: all)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (1 question per level, no models)"
    )
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.load_models = False
        args.max_per_level = 1
        print("\n🚀 Quick Test Mode Enabled\n")
    
    try:
        run_inference_test(
            load_models=args.load_models,
            max_questions_per_level=args.max_per_level
        )
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
