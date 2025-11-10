#!/usr/bin/env python3
"""
Expanded Philosophy Inference Test for LLM-Swarm-Brain

Comprehensive test battery with 100 philosophical questions across 10 complexity levels.
Tests the neural network's ability to handle deep philosophical reasoning from
foundational concepts to advanced meta-philosophical analysis.

Total: 100 questions across 10 levels (10 questions per level)
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


# Expanded Philosophical Test Questions - 100 Questions across 10 Levels
EXPANDED_PHILOSOPHY_QUESTIONS = {
    1: {
        "name": "Foundational Concepts",
        "description": "Basic philosophical terminology and classical ideas",
        "questions": [
            "What is Socrates' definition of knowledge in Plato's Theaetetus?",
            "What does Aristotle mean by 'the good life' (eudaimonia)?",
            "What is Descartes' famous statement about certainty?",
            "Is the statement 'All bachelors are unmarried' true by definition?",
            "What does 'tabula rasa' mean in Locke's philosophy?",
            "What is Plato's Theory of Forms?",
            "What does Aristotle mean by 'the golden mean'?",
            "What is the difference between a priori and a posteriori knowledge?",
            "What is Kant's categorical imperative?",
            "What does 'cogito ergo sum' mean?"
        ]
    },
    2: {
        "name": "Basic Logical Reasoning",
        "description": "Simple philosophical arguments and logical structures",
        "questions": [
            "If pleasure is good, and virtue produces pleasure, does that make virtue good? (Epicurean reasoning)",
            "If we can doubt everything, can we doubt that we are doubting? (Cartesian doubt)",
            "If all swans we've observed are white, must all swans be white? (Induction problem)",
            "If an action maximizes happiness, is it therefore morally right? (Basic utilitarianism)",
            "If nothing changes, can time exist? (Pre-Socratic paradox)",
            "If God is all-powerful, can God create a stone so heavy that God cannot lift it?",
            "If a tree falls in a forest and no one hears it, does it make a sound? (Berkeley's idealism)",
            "Can you step into the same river twice? (Heraclitus)",
            "If everything has a cause, what caused the first cause? (Cosmological argument)",
            "Is it possible to know anything with absolute certainty?"
        ]
    },
    3: {
        "name": "Pattern Recognition & Connections",
        "description": "Identifying philosophical patterns and historical connections",
        "questions": [
            "What pattern connects Plato's Forms, Kant's noumena, and thing-in-itself concepts?",
            "How does Hegel's dialectic (thesis-antithesis-synthesis) apply to historical change?",
            "What connects Descartes' dualism, Berkeley's idealism, and materialism as responses to the mind-body problem?",
            "What pattern underlies skepticism from Pyrrho to Hume to modern epistemology?",
            "How do Aristotle's four causes form a complete explanatory pattern?",
            "What connects empiricism (Locke, Berkeley, Hume) as a unified tradition?",
            "How do rationalists (Descartes, Spinoza, Leibniz) share common methodological assumptions?",
            "What pattern connects virtue ethics, deontology, and consequentialism as ethical frameworks?",
            "How does the problem of universals connect Plato, Aristotle, and medieval philosophy?",
            "What pattern underlies the development from pre-Socratics to Socrates to Plato?"
        ]
    },
    4: {
        "name": "Abstract Conceptual Analysis",
        "description": "Complex philosophical concepts and their relationships",
        "questions": [
            "What is the relationship between essence and existence in existentialist thought?",
            "How does Kant reconcile empiricism and rationalism in his Critique of Pure Reason?",
            "What does Wittgenstein mean by 'the limits of my language are the limits of my world'?",
            "How does Quine's web of belief challenge the analytic-synthetic distinction?",
            "What is the difference between ethical relativism and moral subjectivism?",
            "How does phenomenology differ from traditional epistemology in its approach to consciousness?",
            "What is the relationship between necessity and possibility in modal logic?",
            "How does Heidegger's concept of 'Being' differ from traditional metaphysical substance?",
            "What is the distinction between de re and de dicto necessity?",
            "How does Davidson's anomalous monism attempt to reconcile mental causation with physical determinism?"
        ]
    },
    5: {
        "name": "Counterfactual & Hypothetical Reasoning",
        "description": "Thought experiments and alternative scenarios",
        "questions": [
            "If personal identity depends on memory (Locke), what happens if you have false memories?",
            "What would ethics look like if Kant's categorical imperative were the only moral principle?",
            "If Nietzsche's eternal recurrence were true, how would it change moral decision-making?",
            "What if Leibniz is right and this is the best of all possible worlds despite apparent evil?",
            "If phenomenology is correct that consciousness is always 'consciousness of something,' can there be unconscious mental states?",
            "If determinism is true, can we still hold people morally responsible for their actions?",
            "What would happen to ethics if we discovered that free will is an illusion?",
            "If Plato's Forms exist, where and how do they exist?",
            "What if solipsism is true and you are the only conscious being?",
            "If time travel were possible, what would happen to causation and personal identity?"
        ]
    },
    6: {
        "name": "Meta-Philosophical Analysis",
        "description": "Philosophy of philosophy and methodological questions",
        "questions": [
            "Can philosophy make progress, or does it merely reformulate eternal questions?",
            "What are the limits of rational inquiry in addressing philosophical problems?",
            "How do we know which philosophical method (analytic, phenomenological, pragmatic) is appropriate for which questions?",
            "Is there a fact of the matter about philosophical disagreements, or are they merely conceptual?",
            "Can thought experiments reveal genuine metaphysical truths or only conceptual relations?",
            "What is the relationship between philosophical analysis and scientific investigation?",
            "Can philosophical questions be answered empirically, or are they fundamentally a priori?",
            "Is philosophy continuous with science, or does it occupy a separate domain?",
            "What makes a philosophical explanation better than another?",
            "Can philosophy discover necessary truths, or only contingent conceptual relationships?"
        ]
    },
    7: {
        "name": "Deep Metaphysical Questions",
        "description": "Fundamental questions about reality and existence",
        "questions": [
            "Why is there something rather than nothing? (Leibnizian cosmological question)",
            "Is consciousness fundamental to reality or emergent from physical processes?",
            "Does the hard problem of consciousness (Chalmers) show that physicalism is false?",
            "What grounds the laws of logic themselves? (Meta-logical foundation)",
            "Is authentic existence (Heidegger/Sartre) possible in a deterministic universe?",
            "Can there be necessary beings, or is all existence contingent?",
            "What is the ontological status of mathematical objects?",
            "Is time real or an illusion of consciousness?",
            "Can there be multiple equally valid metaphysical frameworks?",
            "What is the relationship between mind and world in constituting reality?"
        ]
    },
    8: {
        "name": "Complex Multi-Step Problems",
        "description": "Philosophical puzzles requiring sustained reasoning",
        "questions": [
            "Gettier Problem Chain: If justified true belief isn't sufficient for knowledge, and we add a 'no false lemmas' condition, does that solve it? What if the justification is itself only accidentally true? How many conditions are needed?",
            "Modal Metaphysics: If possible worlds are real (Lewis), how do we have knowledge of them? If they're abstract (Plantinga), how do they ground modal truths? If they're linguistic (Carnap), how do they capture metaphysical necessity? Which view is most defensible?",
            "Personal Identity Spectrum: You're slowly replaced neuron-by-neuron with silicon chips that maintain the same functional relations. At what point (if any) do you cease to exist? Does the answer change if it happens instantaneously? What does this reveal about personal identity?",
            "Free Will Trilemma: Hard determinism denies free will. Libertarianism requires causally inexplicable actions. Compatibilism redefines freedom. Each has serious problems. Can you construct a fourth position that avoids all three horns, or show why one horn is actually acceptable?",
            "Semantic Holism Puzzle: If Quine is right that meanings are determined holistically by entire theories, and theories change when we adopt new beliefs, then meaning changes constantly. But if meaning changes, can we say we're changing our minds about the 'same thing'? How can radical translation or theory change be possible?",
            "Sorites Paradox Extended: If removing one grain from a heap doesn't make it not a heap, when does it stop being a heap? Apply this to: personhood, life, consciousness, and moral responsibility. What does this reveal about vagueness in fundamental concepts?",
            "Newcomb's Problem: A perfect predictor has placed either $1M or $0 in box B based on predicting your choice. Box A contains $1000. You can take both boxes or just B. The predictor is never wrong. What should you do, and what does this reveal about rationality, causation, and free will?",
            "Trolley Problem Series: Standard trolley (5 vs 1), fat man variant, loop variant, transplant surgeon. What explains our different intuitions? Is there a unified principle, or do we rely on incompatible moral heuristics?",
            "Liar's Paradox Extended: 'This sentence is false.' If true, it's false. If false, it's true. Now consider: 'This sentence is not provable.' What does this reveal about truth, provability, and the limits of formal systems (Gödel)?",
            "Ship of Theseus + Duplicates: Theseus's ship has all parts replaced. The old parts are assembled into a second ship. Which is the original? Now suppose both ships claim to be Theseus and have his memories. What does this reveal about identity, continuity, and what matters in survival?"
        ]
    },
    9: {
        "name": "Advanced Synthesis & Integration",
        "description": "Integrating multiple philosophical domains and perspectives",
        "questions": [
            "How can we reconcile the phenomenological first-person perspective with the scientific third-person perspective on consciousness?",
            "What is the relationship between moral realism, naturalism, and the is-ought gap? Can all three be maintained consistently?",
            "How do different theories of truth (correspondence, coherence, pragmatic, deflationary) relate to debates about realism vs anti-realism?",
            "Can we develop a unified account of normativity that covers epistemic, moral, and rational norms?",
            "How does the problem of induction relate to the problem of other minds and the problem of the external world?",
            "What is the relationship between semantic externalism, mental content, and self-knowledge?",
            "How can we reconcile quantum indeterminacy with macro-level determinism and free will?",
            "What is the relationship between formal logic, natural language semantics, and human reasoning?",
            "How do evolutionary explanations of morality affect moral realism and moral motivation?",
            "Can we reconcile the manifest image (folk ontology) with the scientific image (physics) without eliminativism?"
        ]
    },
    10: {
        "name": "Cutting-Edge & Speculative Philosophy",
        "description": "Contemporary debates and frontier philosophical questions",
        "questions": [
            "If we create artificial general intelligence with apparent consciousness, what criteria would determine if it's genuinely conscious vs merely simulating consciousness?",
            "How should we think about personal identity in scenarios involving mind uploading, brain emulation, and digital consciousness?",
            "What are the implications of quantum mechanics for causation, determinism, and the nature of physical reality?",
            "If simulation hypothesis is true (we're in a simulation), what implications does this have for metaphysics, epistemology, and ethics?",
            "How should we understand the relationship between information, computation, and physical reality in light of digital physics theories?",
            "What is the moral status of potential future persons, and how does this affect our obligations regarding existential risk?",
            "If we could enhance human cognitive capacities dramatically, would the enhanced beings be the same persons? What does this reveal about personal identity?",
            "How should we think about consciousness in non-human animals, AI systems, and potential alien life forms? Is there a unified theory?",
            "What are the philosophical implications of many-worlds interpretation of quantum mechanics for personal identity and decision theory?",
            "If we discover that consciousness is substrate-independent (implementable in any physical system with the right functional organization), what implications does this have for mind-body problem, personal identity, and ethics?"
        ]
    }
}


def run_expanded_philosophy_test(
    brain: PhiBrain,
    max_questions_per_level: int = 10,
    start_level: int = 1,
    end_level: int = 10
) -> Dict[str, Any]:
    """
    Run expanded philosophical inference test across complexity levels
    
    Args:
        brain: PhiBrain instance
        max_questions_per_level: Maximum questions to test per level
        start_level: Starting complexity level (1-10)
        end_level: Ending complexity level (1-10)
        
    Returns:
        Dictionary containing test results
    """
    results = {
        "test_type": "expanded_philosophy",
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
        level_data = EXPANDED_PHILOSOPHY_QUESTIONS[level]
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
                
                # Extract final output from global workspace or conscious summary
                final_output = ""
                if result.get("global_workspace") and result["global_workspace"].get("conscious_summary"):
                    final_output = result["global_workspace"]["conscious_summary"]
                elif result.get("global_workspace") and result["global_workspace"].get("broadcasts"):
                    # Get top broadcast content
                    broadcasts = result["global_workspace"]["broadcasts"]
                    if broadcasts:
                        final_output = broadcasts[0].get("content", "")
                elif result.get("neural_processing") and result["neural_processing"].get("steps"):
                    # Fallback: get last step output
                    last_step = result["neural_processing"]["steps"][-1]
                    outputs = list(last_step.get("outputs", {}).values())
                    final_output = outputs[0] if outputs else "No output generated"
                else:
                    final_output = "No output generated"
                
                logger.info(f"✓ Response generated in {duration:.2f}s")
                logger.info(f"Answer: {final_output[:200]}...")
                
                question_result = {
                    "question": question,
                    "answer": final_output,
                    "duration_seconds": duration,
                    "neurons_fired": result.get("neural_processing", {}).get("network_metrics", {}).get("neurons_fired", 0),
                    "propagation_steps": len(result.get("neural_processing", {}).get("steps", [])),
                    "global_workspace_broadcasts": len(result.get("global_workspace", {}).get("broadcasts", [])),
                    "consciousness_level": result.get("consciousness_level", 0.0),
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
            "avg_neurons_fired": sum(q.get("neurons_fired", 0) for q in successful) / len(successful) if successful else 0,
            "avg_consciousness": sum(q.get("consciousness_level", 0) for q in successful) / len(successful) if successful else 0
        }
        
        results["levels"][level] = level_results
        
        logger.info(f"\nLevel {level} Summary:")
        logger.info(f"  Successful: {len(successful)}/{len(questions)}")
        logger.info(f"  Avg Duration: {level_results['statistics']['avg_duration']:.2f}s")
        logger.info(f"  Avg Consciousness: {level_results['statistics']['avg_consciousness']:.3f}")
    
    # Overall statistics
    results["summary"] = {
        "total_questions": total_questions,
        "total_duration_seconds": total_time,
        "avg_duration_per_question": total_time / total_questions if total_questions > 0 else 0,
        "levels_completed": end_level - start_level + 1
    }
    
    return results


def save_results(results: Dict[str, Any], filename: str = "expanded_philosophy_results.json") -> str:
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
    parser = argparse.ArgumentParser(description="Expanded Philosophy Test (100 Questions) for LLM-Swarm-Brain")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (1 question per level)")
    parser.add_argument("--sample", action="store_true", help="Sample mode (3 questions per level)")
    parser.add_argument("--load-models", action="store_true", help="Load actual models (default: simulation)")
    parser.add_argument("--use-api", action="store_true", help="Use API instead of local models")
    parser.add_argument("--api-provider", type=str, default="hyperbolic", choices=["hyperbolic", "gemini"], help="API provider (hyperbolic or gemini)")
    parser.add_argument("--api-key", type=str, help="API key (HYPERBOLIC_API_KEY or GOOGLE_API_KEY env var)")
    parser.add_argument("--use-64-neurons", action="store_true", help="Use 64-neuron architecture (default: 8)")
    parser.add_argument("--start-level", type=int, default=1, help="Starting level (1-10)")
    parser.add_argument("--end-level", type=int, default=10, help="Ending level (1-10)")
    parser.add_argument("--questions-per-level", type=int, default=10, help="Questions per level (max 10)")
    
    args = parser.parse_args()
    
    # Configure test parameters
    use_api = args.use_api
    load_models = args.load_models if not use_api else False
    
    if args.quick:
        max_per_level = 1
        mode_desc = "Quick Test (10 questions)"
    elif args.sample:
        max_per_level = 3
        mode_desc = "Sample Test (30 questions)"
    else:
        max_per_level = min(args.questions_per_level, 10)
        mode_desc = f"Full Test ({max_per_level * 10} questions)"
    
    start_level = max(1, min(args.start_level, 10))
    end_level = max(start_level, min(args.end_level, 10))
    
    if args.quick or args.sample:
        use_api = False
        load_models = False
    
    neuron_count = 64 if args.use_64_neurons else 8
    
    # Determine execution mode description
    if use_api:
        if args.api_provider == "gemini":
            exec_mode = "API (Gemini 2.5 Pro)"
        else:
            exec_mode = "API (Llama 3.1 405B)"
    elif load_models:
        exec_mode = "Local Models"
    else:
        exec_mode = "Simulation"
    
    logger.info(f"\n{'='*80}")
    logger.info("EXPANDED PHILOSOPHY TEST - LLM-SWARM-BRAIN")
    logger.info(f"{'='*80}")
    logger.info(f"Test Configuration:")
    logger.info(f"  - Mode: {mode_desc}")
    logger.info(f"  - Execution: {exec_mode}")
    logger.info(f"  - Architecture: {neuron_count} neurons")
    logger.info(f"  - Questions per Level: {max_per_level}")
    logger.info(f"  - Levels: {start_level}-{end_level}")
    logger.info(f"  - Total Questions: {(end_level - start_level + 1) * max_per_level}")
    
    try:
        # Initialize brain
        logger.info(f"\n[1] Initializing PhiBrain...")
        config = None
        if args.use_64_neurons:
            from llm_swarm_brain import config_64
            config = config_64.BrainConfig()
        else:
            config = BrainConfig()
        
        if use_api:
            config.model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct"
        
        brain = PhiBrain(
            config=config,
            load_models=load_models,
            use_api=use_api,
            api_key=args.api_key,
            use_64_neurons=args.use_64_neurons,
            api_provider=args.api_provider
        )
        
        # Run test
        logger.info(f"\n[2] Running expanded philosophy test...")
        results = run_expanded_philosophy_test(
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
        logger.info(f"Total Duration: {summary['total_duration_seconds']:.2f}s ({summary['total_duration_seconds']/60:.1f} minutes)")
        logger.info(f"Avg per Question: {summary['avg_duration_per_question']:.2f}s")
        logger.info(f"Results: {output_file}")
        
        # Cost estimate for API mode
        if use_api:
            estimated_tokens = summary['total_questions'] * 5000  # ~5k tokens per question
            estimated_cost = estimated_tokens * 0.40 / 1_000_000
            logger.info(f"\nEstimated API Cost: ${estimated_cost:.3f} ({estimated_tokens:,} tokens)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
