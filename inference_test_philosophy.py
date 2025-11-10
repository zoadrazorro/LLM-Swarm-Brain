#!/usr/bin/env python3
"""
Philosophy-Focused Inference Test for LLM-Swarm-Brain

Tests the neural network's ability to handle philosophical reasoning
across 8 complexity levels, from basic comprehension to complex
multi-step problem solving.

Total: 40 questions across 8 levels (5 questions per level)
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

from llm_swarm_brain import PhiBrain, BrainConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Philosophical Test Questions by Complexity Level
PHILOSOPHY_QUESTIONS = {
    1: {
        "name": "Basic Comprehension",
        "description": "Ancient & Foundational Concepts",
        "questions": [
            "What is Socrates' definition of knowledge in Plato's Theaetetus?",
            "What does Aristotle mean by 'the good life' (eudaimonia)?",
            "What is Descartes' famous statement about certainty?",
            "Is the statement 'All bachelors are unmarried' true by definition?",
            "What does 'tabula rasa' mean in Locke's philosophy?"
        ]
    },
    2: {
        "name": "Simple Reasoning",
        "description": "Basic Philosophical Arguments",
        "questions": [
            "If pleasure is good, and virtue produces pleasure, does that make virtue good? (Epicurean reasoning)",
            "If we can doubt everything, can we doubt that we are doubting? (Cartesian doubt)",
            "If all swans we've observed are white, must all swans be white? (Induction problem)",
            "If an action maximizes happiness, is it therefore morally right? (Basic utilitarianism)",
            "If nothing changes, can time exist? (Pre-Socratic paradox)"
        ]
    },
    3: {
        "name": "Pattern Recognition",
        "description": "Philosophical Patterns & Structures",
        "questions": [
            "What pattern connects Plato's Forms, Kant's noumena, and thing-in-itself concepts?",
            "How does Hegel's dialectic (thesis-antithesis-synthesis) apply to historical change?",
            "What connects Descartes' dualism, Berkeley's idealism, and materialism as responses to the mind-body problem?",
            "What pattern underlies skepticism from Pyrrho to Hume to modern epistemology?",
            "How do Aristotle's four causes form a complete explanatory pattern?"
        ]
    },
    4: {
        "name": "Abstract Reasoning",
        "description": "Complex Conceptual Analysis",
        "questions": [
            "What is the relationship between essence and existence in existentialist thought?",
            "How does Kant reconcile empiricism and rationalism in his Critique of Pure Reason?",
            "What does Wittgenstein mean by 'the limits of my language are the limits of my world'?",
            "How does Quine's web of belief challenge the analytic-synthetic distinction?",
            "What is the difference between ethical relativism and moral subjectivism?"
        ]
    },
    5: {
        "name": "Counterfactual Thinking",
        "description": "Philosophical Thought Experiments",
        "questions": [
            "If personal identity depends on memory (Locke), what happens if you have false memories?",
            "What would ethics look like if Kant's categorical imperative were the only moral principle?",
            "If Nietzsche's eternal recurrence were true, how would it change moral decision-making?",
            "What if Leibniz is right and this is the best of all possible worlds despite apparent evil?",
            "If phenomenology is correct that consciousness is always 'consciousness of something,' can there be unconscious mental states?"
        ]
    },
    6: {
        "name": "Meta-Cognitive Reasoning",
        "description": "Philosophy of Philosophy",
        "questions": [
            "Can philosophy make progress, or does it merely reformulate eternal questions?",
            "What are the limits of rational inquiry in addressing philosophical problems?",
            "How do we know which philosophical method (analytic, phenomenological, pragmatic) is appropriate for which questions?",
            "Is there a fact of the matter about philosophical disagreements, or are they merely conceptual?",
            "Can thought experiments reveal genuine metaphysical truths or only conceptual relations?"
        ]
    },
    7: {
        "name": "Philosophical & Existential",
        "description": "Deep Metaphysical & Existential Questions",
        "questions": [
            "Why is there something rather than nothing? (Leibnizian cosmological question)",
            "Is consciousness fundamental to reality or emergent from physical processes?",
            "Does the hard problem of consciousness (Chalmers) show that physicalism is false?",
            "What grounds the laws of logic themselves? (Meta-logical foundation)",
            "Is authentic existence (Heidegger/Sartre) possible in a deterministic universe?"
        ]
    },
    8: {
        "name": "Multi-Step Problem Solving",
        "description": "Complex Philosophical Puzzles",
        "questions": [
            "Gettier Problem Chain: If justified true belief isn't sufficient for knowledge (Gettier cases), and we add a 'no false lemmas' condition, does that solve it? What if the justification is itself only accidentally true? How many conditions are needed?",
            "Modal Metaphysics: If possible worlds are real (Lewis), how do we have knowledge of them? If they're abstract (Plantinga), how do they ground modal truths? If they're linguistic (Carnap), how do they capture metaphysical necessity? Which view is most defensible?",
            "Personal Identity Spectrum: You're slowly replaced neuron-by-neuron with silicon chips that maintain the same functional relations. At what point (if any) do you cease to exist? Does the answer change if it happens instantaneously? What does this reveal about the nature of personal identity?",
            "Free Will Trilemma: Hard determinism denies free will. Libertarianism requires causally inexplicable actions. Compatibilism redefines freedom. Each has serious problems. Can you construct a fourth position that avoids all three horns, or show why one horn is actually acceptable?",
            "Semantic Holism Puzzle: If Quine is right that meanings are determined holistically by entire theories, and theories change when we adopt new beliefs, then meaning changes constantly. But if meaning changes, can we say we're changing our minds about the 'same thing'? How can radical translation or theory change be possible? Resolve this paradox."
        ]
    }
}


def run_philosophy_test(
    brain: PhiBrain,
    max_questions_per_level: int = 5,
    start_level: int = 1,
    end_level: int = 8
) -> Dict[str, Any]:
    """
    Run philosophical inference test across complexity levels
    
    Args:
        brain: PhiBrain instance
        max_questions_per_level: Maximum questions to test per level
        start_level: Starting complexity level (1-8)
        end_level: Ending complexity level (1-8)
        
    Returns:
        Dictionary containing test results
    """
    results = {
        "test_type": "philosophy",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": brain.config.model_name,
            "total_neurons": brain.config.total_neurons,
            "max_questions_per_level": max_questions_per_level,
            "levels_tested": f"{start_level}-{end_level}"
        },
        "levels": {}
    }
    
    total_questions = 0
    total_time = 0.0
    
    for level in range(start_level, end_level + 1):
        level_data = PHILOSOPHY_QUESTIONS[level]
        logger.info(f"\n{'='*80}")
        logger.info(f"LEVEL {level}: {level_data['name']}")
        logger.info(f"Description: {level_data['description']}")
        logger.info(f"{'='*80}")
        
        level_results = {
            "name": level_data["name"],
            "description": level_data["description"],
            "questions": []
        }
        
        questions = level_data["questions"][:max_questions_per_level]
        
        for i, question in enumerate(questions, 1):
            logger.info(f"\n[Q{i}/{len(questions)}] {question}")
            
            start_time = datetime.now()
            
            # Adjust max_steps based on complexity level
            max_steps = min(2 + (level // 2), 4)  # 2-4 steps based on level
            
            try:
                result = brain.think(
                    input_text=question,
                    max_steps=max_steps,
                    use_memory=True,
                    use_global_workspace=True,
                    enable_enhancements=True
                )
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                logger.info(f"✓ Response generated in {duration:.2f}s")
                logger.info(f"Answer: {result['final_output'][:200]}...")
                
                question_result = {
                    "question": question,
                    "answer": result["final_output"],
                    "duration_seconds": duration,
                    "neurons_fired": result.get("neurons_fired", 0),
                    "propagation_steps": result.get("propagation_steps", 0),
                    "global_workspace_broadcasts": len(result.get("global_workspace_history", [])),
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"✗ Error processing question: {e}")
                question_result = {
                    "question": question,
                    "error": str(e),
                    "success": False
                }
                duration = 0.0
            
            level_results["questions"].append(question_result)
            total_questions += 1
            total_time += duration
        
        # Calculate level statistics
        successful = [q for q in level_results["questions"] if q.get("success")]
        level_results["statistics"] = {
            "total_questions": len(questions),
            "successful": len(successful),
            "failed": len(questions) - len(successful),
            "avg_duration": sum(q.get("duration_seconds", 0) for q in successful) / len(successful) if successful else 0,
            "avg_neurons_fired": sum(q.get("neurons_fired", 0) for q in successful) / len(successful) if successful else 0
        }
        
        results["levels"][level] = level_results
        
        logger.info(f"\nLevel {level} Summary:")
        logger.info(f"  Successful: {len(successful)}/{len(questions)}")
        logger.info(f"  Avg Duration: {level_results['statistics']['avg_duration']:.2f}s")
    
    # Overall statistics
    results["summary"] = {
        "total_questions": total_questions,
        "total_duration_seconds": total_time,
        "avg_duration_per_question": total_time / total_questions if total_questions > 0 else 0,
        "levels_completed": end_level - start_level + 1
    }
    
    return results


def save_results(results: Dict[str, Any], filename: str = "philosophy_results.json") -> str:
    """Save test results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{filename.split('.')[0]}_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Results saved to {output_file}")
    logger.info(f"{'='*80}")
    
    return output_file


def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description="Philosophy Inference Test for LLM-Swarm-Brain")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (1 question per level, no models)")
    parser.add_argument("--load-models", action="store_true", help="Load actual models (default: simulation)")
    parser.add_argument("--start-level", type=int, default=1, help="Starting level (1-8)")
    parser.add_argument("--end-level", type=int, default=8, help="Ending level (1-8)")
    parser.add_argument("--questions-per-level", type=int, default=5, help="Questions per level (max 5)")
    
    args = parser.parse_args()
    
    # Configure test parameters
    load_models = args.load_models
    max_per_level = 1 if args.quick else min(args.questions_per_level, 5)
    start_level = max(1, min(args.start_level, 8))
    end_level = max(start_level, min(args.end_level, 8))
    
    if args.quick:
        logger.info("🚀 Quick Test Mode Enabled")
        logger.info("  - 1 question per level")
        logger.info("  - Simulation mode (no model loading)")
        load_models = False
    
    logger.info(f"\n{'='*80}")
    logger.info("PHILOSOPHY INFERENCE TEST - LLM-SWARM-BRAIN")
    logger.info(f"{'='*80}")
    logger.info(f"Test Configuration:")
    logger.info(f"  - Load Models: {load_models}")
    logger.info(f"  - Questions per Level: {max_per_level}")
    logger.info(f"  - Levels: {start_level}-{end_level}")
    logger.info(f"  - Total Questions: {(end_level - start_level + 1) * max_per_level}")
    
    try:
        # Initialize brain
        logger.info(f"\n[1] Initializing PhiBrain...")
        config = BrainConfig()
        brain = PhiBrain(config=config, load_models=load_models)
        
        # Run test
        logger.info(f"\n[2] Running philosophy test...")
        results = run_philosophy_test(
            brain=brain,
            max_questions_per_level=max_per_level,
            start_level=start_level,
            end_level=end_level
        )
        
        # Save results
        logger.info(f"\n[3] Saving results...")
        output_file = save_results(results)
        
        # Print summary
        summary = results["summary"]
        logger.info(f"\n{'='*80}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total Questions: {summary['total_questions']}")
        logger.info(f"Total Duration: {summary['total_duration_seconds']:.2f}s")
        logger.info(f"Avg per Question: {summary['avg_duration_per_question']:.2f}s")
        logger.info(f"Results: {output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
