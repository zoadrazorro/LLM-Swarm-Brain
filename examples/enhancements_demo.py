"""
Demonstration of LLM-Swarm-Brain Enhancements

Shows:
- Summarization neuron for output compression
- Attention windowing for selective broadcasting
- Conceptual thread tracking
- Meta-orchestration for dynamic tuning
- Positronic check-in prompts for coherence
"""

import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_swarm_brain import PhiBrain, BrainConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Demonstration of enhancements"""

    print("=" * 70)
    print("LLM-Swarm-Brain: Enhancements Demonstration")
    print("=" * 70)

    print("\n[1] Initializing brain with all enhancements...")

    config = BrainConfig(
        activation_threshold=0.6,
        max_tokens=256,
        temperature=0.7
    )

    # Initialize brain (load_models=False for demo without actual models)
    brain = PhiBrain(config=config, load_models=False, enable_positronic=True)

    print("\n✓ Brain initialized with:")
    print("  • Summarization neuron for output compression")
    print("  • Attention windowing for selective broadcasting")
    print("  • Conceptual thread tracking")
    print("  • Meta-orchestration for dynamic parameter tuning")
    print("  • Positronic check-in prompts for coherence validation")

    # === Test inputs with varying complexity ===
    test_inputs = [
        ("Simple task", "What is 2+2?"),
        ("Medium complexity", "Explain the concept of emergence in complex systems."),
        ("High complexity", "Analyze the relationship between consciousness, information integration, and dialectical reasoning. How do these concepts interact in cognitive architectures?")
    ]

    print("\n[2] Processing inputs with dynamic complexity adaptation...\n")

    for category, input_text in test_inputs:
        print(f"\n--- {category} ---")
        print(f"Input: {input_text}\n")

        # Process through brain
        result = brain.think(
            input_text=input_text,
            use_memory=True,
            use_global_workspace=True,
            enable_enhancements=True  # Enable all enhancements
        )

        # Display enhancement results
        if "enhancements" in result:
            enhancements = result["enhancements"]

            # Task complexity
            complexity = enhancements["task_complexity"]
            print(f"Task Complexity: {complexity['score']:.3f}")
            print(f"Complexity Factors:")
            for factor, value in complexity['factors'].items():
                print(f"  - {factor}: {value:.2f}")

            # Adjusted parameters
            params = enhancements["adjusted_parameters"]
            print(f"\nDynamic Parameters:")
            print(f"  - Activation threshold: {params['activation_threshold']:.2f}")
            print(f"  - Max propagation steps: {params['max_propagation_steps']}")

            # Summarization stats
            summarization = enhancements["summarization_stats"]
            print(f"\nSummarization:")
            print(f"  - Total compressions: {summarization['total_compressions']}")
            print(f"  - Characters saved: {summarization['total_characters_saved']}")

            # Attention windowing stats
            attention = enhancements["attention_stats"]
            print(f"\nAttention Windowing:")
            print(f"  - Windows created: {attention['total_windows_created']}")
            print(f"  - Broadcasts filtered: {attention['total_broadcasts_filtered']}")

            # Concept tracking stats
            concepts = enhancements["concept_stats"]
            print(f"\nConcept Tracking:")
            print(f"  - Concepts tracked: {concepts['total_concepts_tracked']}")
            print(f"  - Threads created: {concepts['total_threads_created']}")

        # Consciousness level
        print(f"\nConsciousness Level: {result['consciousness_level']:.3f}")
        print(f"Processing Time: {result.get('processing_time', 0):.3f}s")

        print("-" * 70)

    # === Demo: Conceptual thread tracking ===
    print("\n[3] Conceptual Thread Tracking Demo...")

    if brain.concept_tracker.total_concepts_tracked > 0:
        flow_summary = brain.concept_tracker.get_network_flow_summary()

        print(f"\nTop Concepts Tracked:")
        for concept_info in flow_summary['top_concepts'][:5]:
            print(f"  - '{concept_info['name']}': {concept_info['occurrences']} occurrences")

        print(f"\nTop Concept Pairs (co-occurring):")
        for pair_info in flow_summary['top_concept_pairs'][:5]:
            concepts = pair_info['concepts']
            count = pair_info['cooccurrences']
            print(f"  - ({concepts[0]}, {concepts[1]}): {count} times")

    # === Demo: Meta-orchestration ===
    print("\n[4] Meta-Orchestration Statistics...")

    meta_stats = brain.meta_orchestrator.get_stats()
    print(f"\nMeta-Orchestration:")
    print(f"  - Total adjustments: {meta_stats['total_adjustments']}")
    print(f"  - Current threshold: {meta_stats['current_parameters']['activation_threshold']:.2f}")
    print(f"  - Current max steps: {meta_stats['current_parameters']['max_propagation_steps']}")
    print(f"  - Avg recent consciousness: {meta_stats['avg_recent_consciousness']:.3f}")

    # === Demo: Positronic check-in prompts ===
    print("\n[5] Positronic Check-in Prompts Demo...")

    if brain.positronic:
        sample_output = "Consciousness emerges from integrated information processing."
        prior_outputs = [
            "Information integration is key to understanding cognition.",
            "Neural networks can exhibit emergent behaviors."
        ]

        checkin_prompts = brain.positronic.generate_coherence_checkins(
            current_output=sample_output,
            prior_outputs=prior_outputs
        )

        print(f"\nGenerated {len(checkin_prompts)} coherence check-in prompts:")
        for i, prompt in enumerate(checkin_prompts[:3], 1):
            print(f"  {i}. {prompt[:100]}...")

    # === Final summary ===
    print("\n[6] Final Brain State:")
    print(brain.get_summary())

    print("\n" + "=" * 70)
    print("Enhancement demonstration completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
