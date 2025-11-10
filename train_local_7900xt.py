"""
Local Training Script for AMD Radeon RX 7900 XT

Trains PhiBrain using Phi-4 (14B) models via LM Studio on local GPU.
Optimized for 20GB VRAM with 2 powerful neurons using Q4 quantization.

Phi-4 offers significantly enhanced reasoning capabilities compared to Phi-3.

Usage:
    python train_local_7900xt.py --questions 100 --max-steps 2

Requirements:
    - LM Studio running on localhost:1234
    - Phi-4 model loaded (Q4_K_M quantization, ~8GB)
    - AMD Radeon RX 7900 XT with 20GB VRAM
"""

import argparse
import logging
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from datasets import load_dataset
from tqdm import tqdm

from llm_swarm_brain.brain import PhiBrain
from config_local_7900xt import LocalBrainConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalTrainer:
    """Local training on 7900XT via LM Studio"""
    
    def __init__(
        self,
        max_steps: int = 2,
        memory_file: str = "training_checkpoints/training_memory.pkl",  # SHARED with cloud
        checkpoint_dir: str = "training_checkpoints/local",
        share_rag_with_cloud: bool = True
    ):
        self.max_steps = max_steps
        self.memory_file = Path(memory_file)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.share_rag_with_cloud = share_rag_with_cloud
        
        # Initialize brain for local LM Studio with Phi-4
        logger.info("Initializing 4-neuron Phi-4 brain for dual-GPU training...")
        logger.info("GPU 0: 2 neurons | GPU 1: 2 neurons")
        logger.info("Using LM Studio at http://localhost:1234")
        logger.info("Model: Phi-4 (14B parameters, enhanced reasoning)")
        if share_rag_with_cloud:
            logger.info("RAG Memory: SHARED with cloud training")
        
        config = LocalBrainConfig()
        
        # Initialize brain with LM Studio API mode
        self.brain = PhiBrain(
            config,
            use_api=True,
            use_64_neurons=False,
            api_provider="hyperbolic"  # Uses OpenAI-compatible API
        )
        
        # Override API URL for LM Studio
        for neuron in self.brain.orchestrator.neurons.values():
            if hasattr(neuron, 'api_url'):
                neuron.api_url = "http://localhost:1234/v1/chat/completions"
        
        # Load persistent memory
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
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
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
                import random
                data = random.sample(data, sample_size)
                logger.info(f"Sampled {sample_size} questions for training")
            
            return data
        
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def train(self, dataset: List[Dict[str, str]], batch_size: int = 5, save_interval: int = 20):
        """Train brain on philosophy dataset"""
        logger.info(f"Starting LOCAL training on {len(dataset)} questions...")
        logger.info(f"Batch size: {batch_size}, Save interval: {save_interval}")
        logger.info(f"Hardware: 2x AMD Radeon RX 7900 XT (40GB total VRAM)")
        logger.info(f"Model: Phi-4 (14B, Q4_K_M) - 4 neurons (2 per GPU)")
        logger.info(f"RAG Sharing: {'ENABLED - syncing with cloud' if self.share_rag_with_cloud else 'DISABLED'}")
        
        start_time = datetime.now()
        
        for i, item in enumerate(tqdm(dataset, desc="Local Training")):
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
                    metrics={
                        "consciousness": result.get("consciousness_level", 0.0),
                        "integration": result.get("integration_score", 0.0),
                        "coherence": result.get("coherence_score", 0.0),
                        "source": "local_dual_gpu"  # Mark as local training
                    }
                )
                
                # Save RAG memory periodically to share with cloud (if enabled)
                if self.share_rag_with_cloud and (i + 1) % 10 == 0:
                    self._save_memory()  # Cloud can pick up new experiences
                
                # Log progress
                if (i + 1) % batch_size == 0:
                    avg_consciousness = sum(self.stats["consciousness_scores"][-batch_size:]) / batch_size
                    avg_integration = sum(self.stats["integration_scores"][-batch_size:]) / batch_size
                    avg_time = self.stats["total_duration"] / self.stats["questions_processed"]
                    logger.info(
                        f"Batch {(i+1)//batch_size}: "
                        f"Consciousness={avg_consciousness:.3f}, "
                        f"Integration={avg_integration:.3f}, "
                        f"Avg Time={avg_time:.1f}s/question"
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
        
        logger.info(f"Local training complete! Duration: {duration/60:.1f} minutes")
    
    def _save_checkpoint(self, question_num: int):
        """Save training checkpoint"""
        checkpoint = {
            "question_num": question_num,
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats.copy(),
            "memory_summary": {
                "total_experiences": len(self.memory["training_history"]),
                "best_performance": self.memory["best_performance"]
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
            "training_complete": datetime.now().isoformat(),
            "hardware": "2x AMD Radeon RX 7900 XT (40GB total VRAM)",
            "model": "Phi-4 (14B, Q4_K_M)",
            "neurons": 4,
            "neurons_per_gpu": 2,
            "rag_shared_with_cloud": self.share_rag_with_cloud,
            "max_steps": self.max_steps,
            "total_questions": self.stats["questions_processed"],
            "total_duration_minutes": self.stats["total_duration"] / 60,
            "avg_time_per_question": self.stats["total_duration"] / max(self.stats["questions_processed"], 1),
            "final_metrics": {
                "avg_consciousness": sum(self.stats["consciousness_scores"]) / max(len(self.stats["consciousness_scores"]), 1),
                "avg_integration": sum(self.stats["integration_scores"]) / max(len(self.stats["integration_scores"]), 1),
                "avg_coherence": sum(self.stats["coherence_scores"]) / max(len(self.stats["coherence_scores"]), 1),
            },
            "best_performance": self.memory["best_performance"],
            "total_training_sessions": self.memory["training_sessions"],
            "cumulative_questions": self.memory["total_questions"],
            "cumulative_time_minutes": self.memory["total_training_time"]
        }
        
        report_file = self.checkpoint_dir / f"local_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final report saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Local training on AMD Radeon RX 7900 XT")
    parser.add_argument("--questions", type=int, default=100, help="Number of questions to train on")
    parser.add_argument("--max-steps", type=int, default=2, help="Maximum reasoning steps per question")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for progress logging")
    parser.add_argument("--save-interval", type=int, default=20, help="Save checkpoint every N questions")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("LOCAL TRAINING - Dual AMD Radeon RX 7900 XT with Phi-4")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Hardware: 2x AMD Radeon RX 7900 XT (40GB total VRAM)")
    logger.info(f"  Model: Phi-4 (14B parameters, Q4_K_M)")
    logger.info(f"  Neurons: 4 (2 per GPU, enhanced reasoning)")
    logger.info(f"  Max Steps: {args.max_steps}")
    logger.info(f"  Questions: {args.questions}")
    logger.info(f"  LM Studio: http://localhost:1234")
    logger.info(f"  RAG Sharing: ENABLED (syncs with cloud training)")
    logger.info("=" * 80)
    
    # Initialize trainer
    trainer = LocalTrainer(
        max_steps=args.max_steps
    )
    
    # Download dataset
    dataset = trainer.download_dataset(sample_size=args.questions)
    
    # Train
    trainer.train(
        dataset,
        batch_size=args.batch_size,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()
