#!/usr/bin/env python3
"""
Training Script for LLM-Swarm-Brain

This script trains the neural network through progressive complexity,
using Hebbian learning to strengthen useful connections over time.

The training follows a curriculum:
1. Simple questions (warm-up)
2. Moderate complexity (skill building)
3. Complex questions (mastery)
4. Expert-level questions (peak performance)

Each question strengthens connections between neurons that fire together,
improving integration, coherence, and consciousness over time.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_swarm_brain import PhiBrain, BrainConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingCurriculum:
    """Progressive training curriculum with increasing complexity"""
    
    def __init__(self):
        self.curriculum = {
            "level_1_warmup": {
                "name": "Warm-up (Simple Concepts)",
                "complexity": 0.2,
                "expected_duration": "1-2 min",
                "questions": [
                    "What is the difference between knowledge and belief?",
                    "Explain the concept of causation in simple terms.",
                    "What does it mean for something to be 'true'?",
                    "Define consciousness in one sentence.",
                    "What is the mind-body problem?"
                ]
            },
            "level_2_foundation": {
                "name": "Foundation (Basic Reasoning)",
                "complexity": 0.4,
                "expected_duration": "2-4 min",
                "questions": [
                    "How does Descartes' 'I think, therefore I am' establish certainty?",
                    "What is the difference between deductive and inductive reasoning?",
                    "Explain Plato's allegory of the cave and its meaning.",
                    "What is the problem of induction as described by Hume?",
                    "How does Kant distinguish between phenomena and noumena?"
                ]
            },
            "level_3_intermediate": {
                "name": "Intermediate (Pattern Recognition)",
                "complexity": 0.6,
                "expected_duration": "4-8 min",
                "questions": [
                    "Compare and contrast rationalism and empiricism. What are the strengths and weaknesses of each?",
                    "How does the Ship of Theseus paradox challenge our understanding of identity?",
                    "Explain the Chinese Room argument and what it suggests about artificial intelligence.",
                    "What is the hard problem of consciousness and why is it considered 'hard'?",
                    "How does compatibilism attempt to reconcile free will and determinism?"
                ]
            },
            "level_4_advanced": {
                "name": "Advanced (Complex Integration)",
                "complexity": 0.75,
                "expected_duration": "8-15 min",
                "questions": [
                    "Analyze the relationship between Kant's categorical imperative and utilitarian ethics. Can they be reconciled?",
                    "How does Heidegger's concept of 'Being-in-the-world' differ from Cartesian dualism? What are the implications?",
                    "Examine the problem of other minds. How can we know that other beings are conscious?",
                    "Compare functionalism and biological naturalism as theories of mind. Which is more plausible and why?",
                    "How does Wittgenstein's private language argument challenge our understanding of meaning?"
                ]
            },
            "level_5_expert": {
                "name": "Expert (Multi-Part Synthesis)",
                "complexity": 0.85,
                "expected_duration": "15-25 min",
                "questions": [
                    """Consider the relationship between consciousness, identity, and substrate:
                    1. If consciousness is substrate-independent, what distinguishes genuine consciousness from simulation?
                    2. In gradual neuron replacement, when (if ever) does identity cease?
                    3. How do these questions relate to each other?
                    Provide a unified theory addressing both problems.""",
                    
                    """Analyze the free will debate across multiple dimensions:
                    1. Evaluate hard determinism, libertarianism, and compatibilism
                    2. Does quantum indeterminacy provide room for free will?
                    3. Can we maintain moral responsibility without libertarian free will?
                    4. What are the practical implications for justice and ethics?
                    Construct a coherent position integrating all dimensions.""",
                    
                    """Examine the nature of knowledge and justification:
                    1. How do foundationalism, coherentism, and reliabilism differ?
                    2. Does the Gettier problem refute the justified true belief account?
                    3. What role does social epistemology play in knowledge acquisition?
                    4. Can we have knowledge in the face of radical skepticism?
                    Develop a comprehensive epistemological framework."""
                ]
            }
        }
    
    def get_level(self, level_name: str) -> Dict:
        """Get questions for a specific level"""
        return self.curriculum.get(level_name, {})
    
    def get_all_levels(self) -> List[str]:
        """Get all level names in order"""
        return list(self.curriculum.keys())
    
    def get_total_questions(self) -> int:
        """Get total number of questions across all levels"""
        return sum(len(level["questions"]) for level in self.curriculum.values())


class NetworkTrainer:
    """Trains the neural network through progressive complexity"""
    
    def __init__(
        self,
        brain: PhiBrain,
        curriculum: TrainingCurriculum,
        save_checkpoints: bool = True,
        checkpoint_dir: str = "training_checkpoints",
        memory_file: str = "training_memory.pkl",
        load_memory: bool = True
    ):
        self.brain = brain
        self.curriculum = curriculum
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = Path(checkpoint_dir)
        self.memory_file = self.checkpoint_dir / memory_file
        
        # Create checkpoint directory
        if self.save_checkpoints:
            self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training metrics
        self.training_history = {
            "levels": [],
            "questions": [],
            "consciousness_progression": [],
            "integration_progression": [],
            "coherence_progression": [],
            "connection_strengths": [],
            "training_sessions": []
        }
        
        # Persistent memory
        self.persistent_memory = {
            "total_training_time": 0.0,
            "total_questions_processed": 0,
            "connection_evolution": {},
            "best_performance": {
                "consciousness": 0.0,
                "integration": 0.0,
                "coherence": 0.0
            },
            "learned_patterns": [],
            "session_history": []
        }
        
        # Load previous memory if exists
        if load_memory:
            self._load_memory()
    
    def train_level(
        self,
        level_name: str,
        max_steps: int = 3,
        save_after_level: bool = True
    ) -> Dict[str, Any]:
        """Train on all questions in a level"""
        level = self.curriculum.get_level(level_name)
        if not level:
            raise ValueError(f"Unknown level: {level_name}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING LEVEL: {level['name']}")
        logger.info(f"Complexity: {level['complexity']}")
        logger.info(f"Questions: {len(level['questions'])}")
        logger.info(f"Expected Duration: {level['expected_duration']} per question")
        logger.info(f"{'='*80}\n")
        
        level_results = {
            "level_name": level_name,
            "level_info": level,
            "questions": [],
            "avg_consciousness": 0.0,
            "avg_integration": 0.0,
            "avg_coherence": 0.0,
            "total_duration": 0.0
        }
        
        for i, question in enumerate(level["questions"], 1):
            logger.info(f"\n[Q{i}/{len(level['questions'])}] {question[:80]}...")
            
            # Process question
            start_time = datetime.now()
            result = self.brain.think(
                question,
                max_steps=max_steps,
                use_memory=True,
                use_global_workspace=True,
                enable_enhancements=True
            )
            duration = (datetime.now() - start_time).total_seconds()
            
            # Extract metrics
            consciousness = result.get("consciousness_level", 0.0)
            integration = result.get("global_workspace", {}).get("integration_score", 0.0)
            coherence = result.get("positronic_coherence", {}).get("score", 0.0)
            
            # Log results
            logger.info(f"✓ Completed in {duration:.1f}s")
            logger.info(f"  Consciousness: {consciousness:.3f}")
            logger.info(f"  Integration: {integration:.3f}")
            logger.info(f"  Coherence: {coherence:.3f}")
            
            # Store results
            question_result = {
                "question": question,
                "consciousness": consciousness,
                "integration": integration,
                "coherence": coherence,
                "duration": duration
            }
            level_results["questions"].append(question_result)
            
            # Update training history
            self.training_history["questions"].append(question)
            self.training_history["consciousness_progression"].append(consciousness)
            self.training_history["integration_progression"].append(integration)
            self.training_history["coherence_progression"].append(coherence)
        
        # Calculate level averages
        level_results["avg_consciousness"] = sum(q["consciousness"] for q in level_results["questions"]) / len(level_results["questions"])
        level_results["avg_integration"] = sum(q["integration"] for q in level_results["questions"]) / len(level_results["questions"])
        level_results["avg_coherence"] = sum(q["coherence"] for q in level_results["questions"]) / len(level_results["questions"])
        level_results["total_duration"] = sum(q["duration"] for q in level_results["questions"])
        
        # Log level summary
        logger.info(f"\n{'='*80}")
        logger.info(f"LEVEL COMPLETE: {level['name']}")
        logger.info(f"  Avg Consciousness: {level_results['avg_consciousness']:.3f}")
        logger.info(f"  Avg Integration: {level_results['avg_integration']:.3f}")
        logger.info(f"  Avg Coherence: {level_results['avg_coherence']:.3f}")
        logger.info(f"  Total Duration: {level_results['total_duration']:.1f}s ({level_results['total_duration']/60:.1f} min)")
        logger.info(f"{'='*80}\n")
        
        # Store level results
        self.training_history["levels"].append(level_results)
        
        # Update persistent memory
        self._update_persistent_memory(level_results)
        
        # Save checkpoint
        if save_after_level and self.save_checkpoints:
            self._save_checkpoint(level_name)
        
        return level_results
    
    def train_full_curriculum(
        self,
        max_steps: int = 3,
        start_level: str = None
    ) -> Dict[str, Any]:
        """Train on the full curriculum"""
        logger.info(f"\n{'#'*80}")
        logger.info("STARTING FULL CURRICULUM TRAINING")
        logger.info(f"Total Questions: {self.curriculum.get_total_questions()}")
        logger.info(f"Max Steps per Question: {max_steps}")
        logger.info(f"{'#'*80}\n")
        
        levels = self.curriculum.get_all_levels()
        
        # Start from specific level if requested
        if start_level:
            if start_level not in levels:
                raise ValueError(f"Unknown start level: {start_level}")
            start_idx = levels.index(start_level)
            levels = levels[start_idx:]
            logger.info(f"Starting from level: {start_level}\n")
        
        # Train each level
        for level_name in levels:
            self.train_level(level_name, max_steps=max_steps)
        
        # Generate final report
        report = self._generate_training_report()
        
        # Save final results
        if self.save_checkpoints:
            self._save_final_results(report)
        
        return report
    
    def _load_memory(self):
        """Load persistent memory from previous training sessions"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'rb') as f:
                    loaded_memory = pickle.load(f)
                    self.persistent_memory.update(loaded_memory)
                
                logger.info(f"✓ Loaded persistent memory from {self.memory_file}")
                logger.info(f"  Previous training time: {self.persistent_memory['total_training_time']/60:.1f} minutes")
                logger.info(f"  Previous questions: {self.persistent_memory['total_questions_processed']}")
                logger.info(f"  Previous sessions: {len(self.persistent_memory['session_history'])}")
                logger.info(f"  Best consciousness: {self.persistent_memory['best_performance']['consciousness']:.3f}")
            except Exception as e:
                logger.warning(f"Could not load memory: {e}. Starting fresh.")
        else:
            logger.info("No previous memory found. Starting fresh training.")
    
    def _save_memory(self):
        """Save persistent memory for future sessions"""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.persistent_memory, f)
            logger.info(f"✓ Persistent memory saved to {self.memory_file}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def _update_persistent_memory(self, level_results: Dict):
        """Update persistent memory with new training data"""
        # Update totals
        self.persistent_memory["total_training_time"] += level_results["total_duration"]
        self.persistent_memory["total_questions_processed"] += len(level_results["questions"])
        
        # Update best performance
        if level_results["avg_consciousness"] > self.persistent_memory["best_performance"]["consciousness"]:
            self.persistent_memory["best_performance"]["consciousness"] = level_results["avg_consciousness"]
        if level_results["avg_integration"] > self.persistent_memory["best_performance"]["integration"]:
            self.persistent_memory["best_performance"]["integration"] = level_results["avg_integration"]
        if level_results["avg_coherence"] > self.persistent_memory["best_performance"]["coherence"]:
            self.persistent_memory["best_performance"]["coherence"] = level_results["avg_coherence"]
        
        # Track connection evolution (if available)
        if hasattr(self.brain, 'neurons'):
            connection_snapshot = {}
            for neuron in self.brain.neurons[:10]:  # Sample first 10 neurons
                neuron_connections = {}
                for conn in neuron.connections[:5]:  # Sample first 5 connections
                    target_id = conn.target_neuron.neuron_id if hasattr(conn, 'target_neuron') else 'unknown'
                    neuron_connections[target_id] = conn.weight
                connection_snapshot[neuron.neuron_id] = neuron_connections
            
            timestamp = datetime.now().isoformat()
            self.persistent_memory["connection_evolution"][timestamp] = connection_snapshot
        
        # Add to session history
        session_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level_results["level_name"],
            "avg_consciousness": level_results["avg_consciousness"],
            "avg_integration": level_results["avg_integration"],
            "avg_coherence": level_results["avg_coherence"],
            "questions_count": len(level_results["questions"])
        }
        self.persistent_memory["session_history"].append(session_entry)
    
    def _save_checkpoint(self, level_name: str):
        """Save training checkpoint"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{level_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        checkpoint_data = {
            "level": level_name,
            "timestamp": datetime.now().isoformat(),
            "training_history": self.training_history,
            "persistent_memory_summary": {
                "total_training_time": self.persistent_memory["total_training_time"],
                "total_questions": self.persistent_memory["total_questions_processed"],
                "best_performance": self.persistent_memory["best_performance"]
            },
            "brain_metrics": self.brain._get_metrics() if hasattr(self.brain, '_get_metrics') else {}
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        
        # Save persistent memory
        self._save_memory()
    
    def _save_final_results(self, report: Dict):
        """Save final training results"""
        results_file = self.checkpoint_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nFinal results saved: {results_file}")
    
    def _generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        logger.info(f"\n{'#'*80}")
        logger.info("TRAINING COMPLETE - FINAL REPORT")
        logger.info(f"{'#'*80}\n")
        
        # Calculate overall metrics
        all_consciousness = self.training_history["consciousness_progression"]
        all_integration = self.training_history["integration_progression"]
        all_coherence = self.training_history["coherence_progression"]
        
        # Learning progression
        first_5_consciousness = sum(all_consciousness[:5]) / min(5, len(all_consciousness)) if all_consciousness else 0
        last_5_consciousness = sum(all_consciousness[-5:]) / min(5, len(all_consciousness)) if all_consciousness else 0
        improvement = ((last_5_consciousness - first_5_consciousness) / first_5_consciousness * 100) if first_5_consciousness > 0 else 0
        
        report = {
            "training_summary": {
                "total_questions": len(self.training_history["questions"]),
                "total_levels": len(self.training_history["levels"]),
                "total_duration_minutes": sum(level["total_duration"] for level in self.training_history["levels"]) / 60
            },
            "performance_metrics": {
                "overall_avg_consciousness": sum(all_consciousness) / len(all_consciousness) if all_consciousness else 0,
                "overall_avg_integration": sum(all_integration) / len(all_integration) if all_integration else 0,
                "overall_avg_coherence": sum(all_coherence) / len(all_coherence) if all_coherence else 0,
                "peak_consciousness": max(all_consciousness) if all_consciousness else 0,
                "peak_integration": max(all_integration) if all_integration else 0,
                "peak_coherence": max(all_coherence) if all_coherence else 0
            },
            "learning_progression": {
                "first_5_avg_consciousness": first_5_consciousness,
                "last_5_avg_consciousness": last_5_consciousness,
                "improvement_percentage": improvement,
                "consciousness_trend": "improving" if improvement > 0 else "stable" if improvement == 0 else "declining"
            },
            "persistent_memory": {
                "lifetime_training_time_minutes": self.persistent_memory["total_training_time"] / 60,
                "lifetime_questions_processed": self.persistent_memory["total_questions_processed"],
                "lifetime_best_performance": self.persistent_memory["best_performance"],
                "total_training_sessions": len(self.persistent_memory["session_history"]),
                "connection_evolution_snapshots": len(self.persistent_memory["connection_evolution"])
            },
            "level_performance": self.training_history["levels"],
            "full_history": self.training_history
        }
        
        # Log report
        logger.info(f"Total Questions: {report['training_summary']['total_questions']}")
        logger.info(f"Total Duration: {report['training_summary']['total_duration_minutes']:.1f} minutes")
        logger.info(f"\nOverall Performance:")
        logger.info(f"  Avg Consciousness: {report['performance_metrics']['overall_avg_consciousness']:.3f}")
        logger.info(f"  Avg Integration: {report['performance_metrics']['overall_avg_integration']:.3f}")
        logger.info(f"  Avg Coherence: {report['performance_metrics']['overall_avg_coherence']:.3f}")
        logger.info(f"\nLearning Progression:")
        logger.info(f"  First 5 Questions: {report['learning_progression']['first_5_avg_consciousness']:.3f}")
        logger.info(f"  Last 5 Questions: {report['learning_progression']['last_5_avg_consciousness']:.3f}")
        logger.info(f"  Improvement: {report['learning_progression']['improvement_percentage']:.1f}%")
        logger.info(f"  Trend: {report['learning_progression']['consciousness_trend']}")
        logger.info(f"\nPersistent Memory (Lifetime Stats):")
        logger.info(f"  Total Training Time: {report['persistent_memory']['lifetime_training_time_minutes']:.1f} minutes")
        logger.info(f"  Total Questions: {report['persistent_memory']['lifetime_questions_processed']}")
        logger.info(f"  Training Sessions: {report['persistent_memory']['total_training_sessions']}")
        logger.info(f"  Best Consciousness: {report['persistent_memory']['lifetime_best_performance']['consciousness']:.3f}")
        logger.info(f"  Best Integration: {report['persistent_memory']['lifetime_best_performance']['integration']:.3f}")
        logger.info(f"  Best Coherence: {report['persistent_memory']['lifetime_best_performance']['coherence']:.3f}")
        logger.info(f"\n{'#'*80}\n")
        
        # Save final persistent memory
        self._save_memory()
        
        return report


def main():
    """Main training execution"""
    parser = argparse.ArgumentParser(description="Train LLM-Swarm-Brain through progressive complexity")
    parser.add_argument("--neurons", type=int, default=8, choices=[8, 64, 128], help="Number of neurons")
    parser.add_argument("--api-provider", type=str, default="gemini", choices=["hyperbolic", "gemini"], help="API provider")
    parser.add_argument("--gemini-model", type=str, default="gemini-exp-1206", help="Gemini model name")
    parser.add_argument("--hyperbolic-model", type=str, default="meta-llama/Meta-Llama-3.1-405B-Instruct", help="Hyperbolic model name")
    parser.add_argument("--max-steps", type=int, default=3, help="Max reasoning steps per question")
    parser.add_argument("--start-level", type=str, help="Start from specific level (e.g., level_3_intermediate)")
    parser.add_argument("--single-level", type=str, help="Train only on a single level")
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable checkpoint saving")
    parser.add_argument("--no-memory", action="store_true", help="Don't load/save persistent memory")
    parser.add_argument("--reset-memory", action="store_true", help="Reset persistent memory (start fresh)")
    parser.add_argument("--api-key", type=str, help="API key (or use environment variable)")
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*80}")
    logger.info("LLM-SWARM-BRAIN TRAINING")
    logger.info(f"{'='*80}")
    logger.info(f"Configuration:")
    logger.info(f"  Neurons: {args.neurons}")
    logger.info(f"  API Provider: {args.api_provider}")
    logger.info(f"  Model: {args.gemini_model if args.api_provider == 'gemini' else args.hyperbolic_model}")
    logger.info(f"  Max Steps: {args.max_steps}")
    logger.info(f"  Checkpoints: {'Disabled' if args.no_checkpoints else 'Enabled'}")
    logger.info(f"{'='*80}\n")
    
    try:
        # Load appropriate config
        if args.neurons == 128:
            from llm_swarm_brain import config_128
            config = config_128.BrainConfig()
        elif args.neurons == 64:
            from llm_swarm_brain import config_64
            config = config_64.BrainConfig()
        else:
            config = BrainConfig()
        
        # Set model based on provider
        if args.api_provider == "gemini":
            config.gemini_model_name = args.gemini_model
        else:
            config.hyperbolic_model_name = args.hyperbolic_model
        
        # Initialize brain
        logger.info("[1] Initializing brain...")
        brain = PhiBrain(
            config=config,
            load_models=False,
            use_api=True,
            api_key=args.api_key,
            use_64_neurons=(args.neurons == 64),
            api_provider=args.api_provider
        )
        logger.info("✓ Brain initialized\n")
        
        # Handle memory reset if requested
        memory_dir = Path("training_checkpoints")
        memory_file = memory_dir / "training_memory.pkl"
        if args.reset_memory and memory_file.exists():
            logger.info("[2] Resetting persistent memory...")
            memory_file.unlink()
            logger.info("✓ Memory reset complete\n")
        
        # Initialize curriculum and trainer
        logger.info("[3] Loading training curriculum...")
        curriculum = TrainingCurriculum()
        trainer = NetworkTrainer(
            brain=brain,
            curriculum=curriculum,
            save_checkpoints=not args.no_checkpoints,
            load_memory=not args.no_memory
        )
        logger.info(f"✓ Curriculum loaded ({curriculum.get_total_questions()} questions)\n")
        
        # Train
        logger.info("[3] Starting training...\n")
        if args.single_level:
            # Train single level
            results = trainer.train_level(args.single_level, max_steps=args.max_steps)
        else:
            # Train full curriculum
            results = trainer.train_full_curriculum(
                max_steps=args.max_steps,
                start_level=args.start_level
            )
        
        logger.info("✓ Training complete!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n\nTraining failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
