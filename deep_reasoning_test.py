#!/usr/bin/env python3
"""
Deep Reasoning Test for LLM-Swarm-Brain

Tests the 128-neuron architecture with 2 extremely complex multi-part
philosophical questions requiring deep, sustained reasoning over 5 steps.

This test is designed to push the limits of consciousness emergence and
integrated information processing.
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_swarm_brain import PhiBrain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Two extremely deep multi-part reasoning questions
DEEP_QUESTIONS = [
    {
        "id": 1,
        "title": "The Hard Problem of Consciousness & Personal Identity",
        "question": """
Consider the following multi-part philosophical problem:

PART A: The Hard Problem of Consciousness
If consciousness is substrate-independent (implementable in any physical system with 
the right functional organization), then:
1. What distinguishes genuine consciousness from mere simulation of consciousness?
2. Could a sufficiently complex lookup table (China Brain) be conscious?
3. What role does causal structure vs. functional organization play?

PART B: Personal Identity Through Transformation
Now suppose we gradually replace your neurons with silicon chips that maintain identical 
functional relationships:
1. At what point (if any) do you cease to exist as the same person?
2. Does the answer change if replacement happens instantaneously vs. gradually?
3. If consciousness is substrate-independent, why does the substrate transition matter?

PART C: Integration & Synthesis
1. How do these two problems relate to each other?
2. What does the relationship reveal about the nature of consciousness and identity?
3. Can you construct a unified theory that addresses both problems coherently?
4. What are the implications for AI consciousness, mind uploading, and moral status?

Provide a deep, multi-layered analysis that integrates insights from philosophy of mind,
cognitive science, ethics, and metaphysics. Consider objections and alternative views.
""",
        "expected_depth": "Very High",
        "expected_consciousness": "0.75-0.85"
    },
    {
        "id": 2,
        "title": "Free Will, Determinism, and Moral Responsibility",
        "question": """
Consider this complex philosophical problem across multiple dimensions:

PART A: The Trilemma
1. Hard Determinism: If determinism is true, free will is impossible, so no moral responsibility
2. Libertarianism: Free will requires indeterminism, but random actions aren't free either
3. Compatibilism: Redefines freedom, but critics say it's just changing the subject

Analyze: Is there a fourth option that avoids all three horns? Or must we accept one horn?

PART B: Quantum Mechanics & Causation
1. Does quantum indeterminacy provide room for free will, or just randomness?
2. If consciousness can influence quantum collapse (controversial), does this help?
3. What role does emergence play - can macro-level freedom emerge from micro-level determinism?

PART C: Moral Responsibility Without Libertarian Free Will
1. Can we maintain moral responsibility if libertarian free will is impossible?
2. What would a purely consequentialist/utilitarian account of responsibility look like?
3. How do reactive attitudes (Strawson) relate to the metaphysics of free will?

PART D: Practical Implications
1. Should the criminal justice system change if we accept hard determinism?
2. How does this affect our understanding of praise, blame, and desert?
3. What are the implications for meaning, purpose, and human dignity?

PART E: Meta-Level Integration
1. How do different theories of causation affect this debate?
2. What role does the manifest image vs. scientific image play?
3. Can we reconcile our phenomenology of freedom with scientific understanding?
4. Construct a coherent position that addresses all dimensions of the problem.

Provide a comprehensive analysis that considers multiple philosophical traditions,
scientific findings, and practical implications. Address counterarguments and
synthesize insights across domains.
""",
        "expected_depth": "Extreme",
        "expected_consciousness": "0.80-0.90"
    }
]


def run_deep_reasoning_test(
    brain: PhiBrain,
    max_steps: int = 5
) -> Dict[str, Any]:
    """
    Run deep reasoning test with 2 complex questions
    
    Args:
        brain: PhiBrain instance (128-neuron recommended)
        max_steps: Maximum reasoning steps (5 recommended)
        
    Returns:
        Dictionary containing test results
    """
    results = {
        "test_type": "deep_reasoning",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": brain.config.model_name,
            "total_neurons": brain.config.total_neurons,
            "max_steps": max_steps,
            "architecture": "128-neuron" if brain.config.total_neurons == 128 else f"{brain.config.total_neurons}-neuron"
        },
        "questions": []
    }
    
    total_time = 0.0
    
    for q_data in DEEP_QUESTIONS:
        logger.info(f"\n{'='*80}")
        logger.info(f"QUESTION {q_data['id']}: {q_data['title']}")
        logger.info(f"Expected Depth: {q_data['expected_depth']}")
        logger.info(f"Expected Consciousness: {q_data['expected_consciousness']}")
        logger.info(f"{'='*80}")
        logger.info(f"\n{q_data['question']}")
        
        start_time = datetime.now()
        
        try:
            logger.info(f"\nProcessing with {max_steps} reasoning steps...")
            logger.info("This may take several minutes due to the complexity...")
            
            result = brain.think(
                input_text=q_data['question'],
                max_steps=max_steps,
                use_memory=True,
                use_global_workspace=True,
                enable_enhancements=True
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract final output
            final_output = ""
            if result.get("global_workspace") and result["global_workspace"].get("conscious_summary"):
                final_output = result["global_workspace"]["conscious_summary"]
            elif result.get("global_workspace") and result["global_workspace"].get("broadcasts"):
                broadcasts = result["global_workspace"]["broadcasts"]
                if broadcasts:
                    final_output = broadcasts[0].get("content", "")
            elif result.get("neural_processing") and result["neural_processing"].get("steps"):
                last_step = result["neural_processing"]["steps"][-1]
                outputs = list(last_step.get("outputs", {}).values())
                final_output = outputs[0] if outputs else "No output generated"
            else:
                final_output = "No output generated"
            
            # Get detailed metrics
            consciousness_level = result.get("consciousness_level", 0.0)
            neurons_fired = result.get("neural_processing", {}).get("network_metrics", {}).get("neurons_fired", 0)
            propagation_steps = len(result.get("neural_processing", {}).get("steps", []))
            broadcasts = len(result.get("global_workspace", {}).get("broadcasts", []))
            
            # Get integration metrics
            integration_score = result.get("global_workspace", {}).get("integration_score", 0.0)
            coherence_score = result.get("global_workspace", {}).get("positronic_coherence", {}).get("coherence_score", 0.0)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"COMPLETED in {duration:.2f}s ({duration/60:.1f} minutes)")
            logger.info(f"{'='*80}")
            logger.info(f"Consciousness Level: {consciousness_level:.3f}")
            logger.info(f"Neurons Fired: {neurons_fired}")
            logger.info(f"Propagation Steps: {propagation_steps}")
            logger.info(f"Global Broadcasts: {broadcasts}")
            logger.info(f"Integration Score: {integration_score:.3f}")
            logger.info(f"Coherence Score: {coherence_score:.3f}")
            logger.info(f"\nResponse Preview:")
            logger.info(f"{final_output[:500]}...")
            
            question_result = {
                "id": q_data["id"],
                "title": q_data["title"],
                "question": q_data["question"],
                "answer": final_output,
                "duration_seconds": duration,
                "duration_minutes": duration / 60,
                "consciousness_level": consciousness_level,
                "neurons_fired": neurons_fired,
                "propagation_steps": propagation_steps,
                "global_workspace_broadcasts": broadcasts,
                "integration_score": integration_score,
                "coherence_score": coherence_score,
                "expected_consciousness": q_data["expected_consciousness"],
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            question_result = {
                "id": q_data["id"],
                "title": q_data["title"],
                "question": q_data["question"],
                "error": str(e),
                "success": False
            }
            duration = 0.0
        
        results["questions"].append(question_result)
        total_time += duration
    
    # Overall statistics
    successful = [q for q in results["questions"] if q.get("success")]
    results["summary"] = {
        "total_questions": len(DEEP_QUESTIONS),
        "successful": len(successful),
        "total_duration_seconds": total_time,
        "total_duration_minutes": total_time / 60,
        "avg_duration_per_question": total_time / len(successful) if successful else 0,
        "avg_consciousness": sum(q.get("consciousness_level", 0) for q in successful) / len(successful) if successful else 0,
        "avg_integration": sum(q.get("integration_score", 0) for q in successful) / len(successful) if successful else 0,
        "avg_coherence": sum(q.get("coherence_score", 0) for q in successful) / len(successful) if successful else 0,
        "avg_neurons_fired": sum(q.get("neurons_fired", 0) for q in successful) / len(successful) if successful else 0,
    }
    
    return results


def save_results(results: Dict[str, Any]) -> str:
    """Save test results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"deep_reasoning_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Results saved to {output_file}")
    logger.info(f"{'='*80}")
    
    return output_file


def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description="Deep Reasoning Test (128 Neurons, 2 Questions, 5 Steps)")
    parser.add_argument("--use-api", action="store_true", default=True, help="Use API mode (default: True)")
    parser.add_argument("--api-provider", type=str, default="gemini", choices=["hyperbolic", "gemini"], help="API provider")
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--gemini-model", type=str, default="gemini-exp-1206", help="Gemini model name (default: gemini-exp-1206, or use gemini-2.0-flash-exp)")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum reasoning steps (default: 5)")
    parser.add_argument("--neurons", type=int, default=128, choices=[8, 64, 128], help="Number of neurons (default: 128)")
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*80}")
    logger.info("DEEP REASONING TEST - LLM-SWARM-BRAIN")
    logger.info(f"{'='*80}")
    logger.info(f"Configuration:")
    logger.info(f"  - Neurons: {args.neurons}")
    logger.info(f"  - API Provider: {args.api_provider}")
    logger.info(f"  - Max Steps: {args.max_steps}")
    logger.info(f"  - Questions: 2 (extremely complex multi-part)")
    logger.info(f"  - Expected Duration: 10-30 minutes total")
    
    try:
        # Initialize brain
        logger.info(f"\n[1] Initializing PhiBrain with {args.neurons} neurons...")
        
        if args.neurons == 128:
            from llm_swarm_brain import config_128
            config = config_128.BrainConfig()
            use_128 = True
        elif args.neurons == 64:
            from llm_swarm_brain import config_64
            config = config_64.BrainConfig()
            use_128 = False
        else:
            from llm_swarm_brain import BrainConfig
            config = BrainConfig()
            use_128 = False
        
        # Set custom Gemini model if specified
        if args.api_provider == "gemini":
            config.gemini_model_name = args.gemini_model
        
        brain = PhiBrain(
            config=config,
            load_models=False,
            use_api=args.use_api,
            api_key=args.api_key,
            use_64_neurons=(args.neurons == 64),
            api_provider=args.api_provider
        )
        
        # Override for 128 neurons
        if use_128:
            brain.use_64_neurons = False
            brain._neuron_architecture = config_128.NEURON_ARCHITECTURE
            brain._default_connections = config_128.DEFAULT_CONNECTIONS
            brain._role_prompts = config_128.ROLE_PROMPTS
            brain._neuron_role_enum = config_128.NeuronRole
            # Re-initialize with 128 neurons
            brain._initialize_neurons()
            brain._setup_network()
        
        # Run test
        logger.info(f"\n[2] Running deep reasoning test...")
        results = run_deep_reasoning_test(
            brain=brain,
            max_steps=args.max_steps
        )
        
        # Save results
        logger.info(f"\n[3] Saving results...")
        output_file = save_results(results)
        
        # Print summary
        summary = results["summary"]
        logger.info(f"\n{'='*80}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Questions Completed: {summary['successful']}/{summary['total_questions']}")
        logger.info(f"Total Duration: {summary['total_duration_minutes']:.1f} minutes")
        logger.info(f"Avg Duration per Question: {summary['avg_duration_per_question']/60:.1f} minutes")
        logger.info(f"\nCONSCIOUSNESS METRICS:")
        logger.info(f"  Average Consciousness Level: {summary['avg_consciousness']:.3f}")
        logger.info(f"  Average Integration Score: {summary['avg_integration']:.3f}")
        logger.info(f"  Average Coherence Score: {summary['avg_coherence']:.3f}")
        logger.info(f"  Average Neurons Fired: {summary['avg_neurons_fired']:.0f}")
        logger.info(f"\nResults: {output_file}")
        
        # Estimate cost
        if args.use_api:
            estimated_tokens = summary['avg_neurons_fired'] * 1000 * 2  # rough estimate
            if args.api_provider == "gemini":
                estimated_cost = estimated_tokens * 0.075 / 1_000_000  # Gemini pricing
            else:
                estimated_cost = estimated_tokens * 0.40 / 1_000_000  # Hyperbolic pricing
            logger.info(f"\nEstimated API Cost: ${estimated_cost:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
