"""
Advanced Training Script with Stanford Encyclopedia of Philosophy Dataset
Downloads and trains the neural network on 12K+ philosophy Q&A pairs
"""

import argparse
import json
import logging
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from datasets import load_dataset
from tqdm import tqdm

from llm_swarm_brain.brain import PhiBrain
from llm_swarm_brain.config import BrainConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhilosophyDatasetTrainer:
    """Trains PhiBrain on Stanford Encyclopedia of Philosophy dataset"""
    
    def __init__(
        self,
        neurons: int = 8,
        api_provider: str = "hyperbolic",
        hyperbolic_model: str = "openai/gpt-oss-20b",
        max_steps: int = 2,
        memory_file: str = "training_checkpoints/training_memory.pkl",
        checkpoint_dir: str = "training_checkpoints"
    ):
        self.neurons = neurons
        self.api_provider = api_provider
        self.hyperbolic_model = hyperbolic_model
        self.max_steps = max_steps
        self.memory_file = Path(memory_file)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize brain
        logger.info(f"Initializing {neurons}-neuron brain...")
        
        # Create config object
        config = BrainConfig()
        if api_provider == "hyperbolic":
            config.hyperbolic_model_name = hyperbolic_model
        
        # Initialize brain with API mode
        self.brain = PhiBrain(
            config,
            use_api=True,
            use_64_neurons=(neurons == 64),
            api_provider=api_provider
        )
        
        # Load persistent memory if exists
        self.memory = self._load_memory()
        
        # Training statistics
        self.stats = {
            "questions_processed": 0,
            "total_duration": 0,
            "consciousness_scores": [],
            "integration_scores": [],
            "coherence_scores": []
        }
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load persistent memory from disk"""
        default_memory = {
            "training_history": [],
            "best_performance": {
                "consciousness": 0.0,
                "integration": 0.0,
                "coherence": 0.0
            },
            "total_training_time": 0.0,
            "total_questions": 0,
            "training_sessions": 0
        }
        
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'rb') as f:
                    memory = pickle.load(f)
                # Ensure all required keys exist (for backwards compatibility)
                for key, value in default_memory.items():
                    if key not in memory:
                        memory[key] = value
                logger.info(f"Loaded memory: {len(memory.get('training_history', []))} past experiences")
                return memory
            except Exception as e:
                logger.warning(f"Could not load memory: {e}")
        
        return default_memory
    
    def _save_memory(self):
        """Save persistent memory to disk"""
        try:
            self.memory_file.parent.mkdir(exist_ok=True)
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.memory, f)
            logger.info(f"Memory saved: {len(self.memory['training_history'])} experiences")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def _update_memory(self, question: str, result: Dict[str, Any]):
        """Update persistent memory with new training experience"""
        experience = {
            "question": question,
            "consciousness": result.get("consciousness_level", 0.0),
            "integration": result.get("integration_score", 0.0),
            "coherence": result.get("coherence_score", 0.0),
            "duration": result.get("duration_seconds", 0.0),
            "timestamp": datetime.now().isoformat()
        }
        
        self.memory["training_history"].append(experience)
        
        # Update best performance
        if experience["consciousness"] > self.memory["best_performance"]["consciousness"]:
            self.memory["best_performance"]["consciousness"] = experience["consciousness"]
        if experience["integration"] > self.memory["best_performance"]["integration"]:
            self.memory["best_performance"]["integration"] = experience["integration"]
        if experience["coherence"] > self.memory["best_performance"]["coherence"]:
            self.memory["best_performance"]["coherence"] = experience["coherence"]
    
    def download_dataset(self, sample_size: int = None) -> List[Dict[str, str]]:
        """Download Stanford Encyclopedia of Philosophy dataset"""
        logger.info("Downloading Stanford Encyclopedia of Philosophy dataset...")
        
        try:
            dataset = load_dataset("ruggsea/stanford-encyclopedia-of-philosophy_instruct", split="train")
            logger.info(f"Dataset loaded: {len(dataset)} Q&A pairs")
            
            # Convert to list of dicts
            data = [{"question": item["question"], "answer": item["answer"]} for item in dataset]
            
            # Sample if requested
            if sample_size and sample_size < len(data):
                data = random.sample(data, sample_size)
                logger.info(f"Sampled {sample_size} questions for training")
            
            return data
        
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def train_on_dataset(
        self,
        dataset: List[Dict[str, str]],
        batch_size: int = 10,
        save_interval: int = 50
    ):
        """Train brain on philosophy dataset"""
        logger.info(f"Starting training on {len(dataset)} questions...")
        logger.info(f"Batch size: {batch_size}, Save interval: {save_interval}")
        
        start_time = datetime.now()
        
        for i, item in enumerate(tqdm(dataset, desc="Training")):
            question = item["question"]
            
            try:
                # Process question through brain
                result = self.brain.think(
                    question,
                    max_steps=self.max_steps,
                    enable_enhancements=True
                )
                
                # Update statistics
                self.stats["questions_processed"] += 1
                self.stats["consciousness_scores"].append(result.get("consciousness_level", 0.0))
                self.stats["integration_scores"].append(result.get("integration_score", 0.0))
                self.stats["coherence_scores"].append(result.get("coherence_score", 0.0))
                self.stats["total_duration"] += result.get("duration_seconds", 0.0)
                
                # Update persistent memory
                self._update_memory(question, result)
                
                # Update brain's RAG memory for cumulative learning
                self.brain.rag_memory.add_experience(
                    question=question,
                    answer=result.get("conscious_content", ""),
                    metadata={
                        "consciousness": result.get("consciousness_level", 0.0),
                        "integration": result.get("integration_score", 0.0),
                        "coherence": result.get("coherence_score", 0.0),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Log progress
                if (i + 1) % batch_size == 0:
                    avg_consciousness = sum(self.stats["consciousness_scores"][-batch_size:]) / batch_size
                    avg_integration = sum(self.stats["integration_scores"][-batch_size:]) / batch_size
                    logger.info(
                        f"Batch {(i+1)//batch_size}: "
                        f"Consciousness={avg_consciousness:.3f}, "
                        f"Integration={avg_integration:.3f}"
                    )
                
                # Save checkpoint
                if (i + 1) % save_interval == 0:
                    self._save_checkpoint(i + 1)
            
            except Exception as e:
                import traceback
                logger.error(f"Error processing question {i+1}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        # Final save
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.memory["total_training_time"] += duration / 60
        self.memory["total_questions"] += len(dataset)
        self.memory["training_sessions"] += 1
        
        self._save_memory()
        self._save_final_report()
        
        logger.info(f"Training complete! Duration: {duration/60:.1f} minutes")
    
    def _save_checkpoint(self, question_num: int):
        """Save training checkpoint"""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "questions_processed": question_num,
            "stats": self.stats.copy(),
            "memory_summary": {
                "total_experiences": len(self.memory["training_history"]),
                "best_consciousness": self.memory["best_performance"]["consciousness"],
                "best_integration": self.memory["best_performance"]["integration"]
            }
        }
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{question_num}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        self._save_memory()
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    def _save_final_report(self):
        """Save final training report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "neurons": self.neurons,
                "api_provider": self.api_provider,
                "model": self.hyperbolic_model,
                "max_steps": self.max_steps
            },
            "statistics": {
                "questions_processed": self.stats["questions_processed"],
                "total_duration_minutes": self.stats["total_duration"] / 60,
                "avg_consciousness": sum(self.stats["consciousness_scores"]) / len(self.stats["consciousness_scores"]) if self.stats["consciousness_scores"] else 0,
                "avg_integration": sum(self.stats["integration_scores"]) / len(self.stats["integration_scores"]) if self.stats["integration_scores"] else 0,
                "avg_coherence": sum(self.stats["coherence_scores"]) / len(self.stats["coherence_scores"]) if self.stats["coherence_scores"] else 0,
                "peak_consciousness": max(self.stats["consciousness_scores"]) if self.stats["consciousness_scores"] else 0,
                "peak_integration": max(self.stats["integration_scores"]) if self.stats["integration_scores"] else 0
            },
            "memory": {
                "total_experiences": len(self.memory["training_history"]),
                "lifetime_questions": self.memory["total_questions"],
                "lifetime_training_time_minutes": self.memory["total_training_time"],
                "training_sessions": self.memory["training_sessions"],
                "best_performance": self.memory["best_performance"]
            }
        }
        
        report_file = self.checkpoint_dir / f"philosophy_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final report saved: {report_file}")
        logger.info(f"Average Consciousness: {report['statistics']['avg_consciousness']:.3f}")
        logger.info(f"Average Integration: {report['statistics']['avg_integration']:.3f}")
        logger.info(f"Peak Consciousness: {report['statistics']['peak_consciousness']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train PhiBrain on Stanford Encyclopedia of Philosophy dataset")
    parser.add_argument("--neurons", type=int, default=8, choices=[8, 64], help="Number of neurons (8 or 64)")
    parser.add_argument("--api-provider", type=str, default="hyperbolic", choices=["hyperbolic", "gemini"], help="API provider")
    parser.add_argument("--hyperbolic-model", type=str, default="openai/gpt-oss-20b", help="Hyperbolic model name")
    parser.add_argument("--max-steps", type=int, default=2, help="Max reasoning steps per question")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of questions to sample (default: all)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for progress logging")
    parser.add_argument("--save-interval", type=int, default=50, help="Save checkpoint every N questions")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("PHILOSOPHY DATASET TRAINING")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Neurons: {args.neurons}")
    logger.info(f"  API Provider: {args.api_provider}")
    logger.info(f"  Model: {args.hyperbolic_model}")
    logger.info(f"  Max Steps: {args.max_steps}")
    logger.info(f"  Sample Size: {args.sample_size or 'ALL (12K+)'}")
    logger.info("=" * 80)
    
    # Initialize trainer
    trainer = PhilosophyDatasetTrainer(
        neurons=args.neurons,
        api_provider=args.api_provider,
        hyperbolic_model=args.hyperbolic_model,
        max_steps=args.max_steps
    )
    
    # Download dataset
    dataset = trainer.download_dataset(sample_size=args.sample_size)
    
    # Train
    trainer.train_on_dataset(
        dataset=dataset,
        batch_size=args.batch_size,
        save_interval=args.save_interval
    )
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
