"""
Basic usage example for LLM-Swarm-Brain

Demonstrates:
- Initializing the brain
- Processing simple inputs
- Viewing results and metrics
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
    """Basic usage example"""

    print("=" * 70)
    print("LLM-Swarm-Brain: Basic Usage Example")
    print("=" * 70)

    # Create brain configuration
    # Note: For first run without models, set load_models=False
    config = BrainConfig(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        max_tokens=256,  # Shorter for demo
        temperature=0.7,
        activation_threshold=0.5,  # Lower threshold for more activation
    )

    print("\n[1] Initializing PhiBrain with 16 neurons...")
    print(f"    - GPU 0: 8 neurons (Perception + Memory)")
    print(f"    - GPU 1: 8 neurons (Reasoning + Action)")

    # Initialize brain
    # Set load_models=False for testing without actual models
    brain = PhiBrain(config=config, load_models=False)

    print(f"\n[2] Brain initialized successfully!")
    print(brain.get_summary())

    # Test inputs
    test_inputs = [
        "What is the meaning of consciousness?",
        "Explain the concept of emergence in complex systems.",
        "How do neural networks learn patterns?",
    ]

    print("\n[3] Processing inputs through the brain...\n")

    for i, input_text in enumerate(test_inputs, 1):
        print(f"\n--- Input {i} ---")
        print(f"Input: {input_text}")

        # Process through brain
        result = brain.think(
            input_text=input_text,
            max_steps=3,  # Limited steps for demo
            use_memory=True,
            use_global_workspace=True
        )

        # Display results
        print(f"\nConsciousness Level: {result['consciousness_level']:.3f}")

        # Show which neurons fired
        if result['neural_processing']['steps']:
            final_step = result['neural_processing']['steps'][-1]
            fired_neurons = final_step['fired_neurons']
            print(f"Neurons Fired: {len(fired_neurons)}")

            if fired_neurons:
                print("\nActive Neurons:")
                for neuron_id in fired_neurons[:5]:  # Show first 5
                    print(f"  - {neuron_id}")

        # Show global workspace broadcasts
        if result['global_workspace'] and result['global_workspace']['broadcasts']:
            print("\nGlobal Workspace Broadcasts:")
            for broadcast in result['global_workspace']['broadcasts'][:3]:
                print(f"  - [{broadcast['source']}] Salience: {broadcast['salience']:.3f}")
                print(f"    {broadcast['content'][:100]}...")

        print("-" * 70)

    # Final brain summary
    print("\n[4] Final Brain State:")
    print(brain.get_summary())

    # Show visualization
    print("\n[5] Neural Activation Visualization:")
    print(brain.visualize_state())

    print("\n[6] Detailed Metrics:")
    metrics = brain._get_brain_metrics()
    print(f"\nNetwork Metrics:")
    print(f"  - Total Neurons: {metrics['network']['total_neurons']}")
    print(f"  - Total Connections: {metrics['network']['total_connections']}")
    print(f"  - Active Neurons: {metrics['network']['active_neurons']}")

    print(f"\nConsciousness Metrics:")
    print(f"  - Average Level: {metrics['consciousness']['average_level']:.3f}")
    print(f"  - Average Integration: {metrics['consciousness']['average_integration']:.3f}")

    print(f"\nMemory Metrics:")
    print(f"  - Short-term: {metrics['memory']['short_term_size']} items")
    print(f"  - Episodic: {metrics['memory']['episodic_size']} episodes")
    print(f"  - Semantic: {metrics['memory']['semantic_size']} entries")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
