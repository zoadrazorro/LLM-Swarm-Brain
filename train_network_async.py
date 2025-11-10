#!/usr/bin/env python3
"""
Async Training Script for LLM-Swarm-Brain with Real-time JSON Output

This script trains the neural network asynchronously with streaming JSON output
for real-time monitoring and integration with other systems.

Features:
- Async/await for non-blocking execution
- Real-time JSON streaming to stdout
- WebSocket server for live monitoring
- Event-driven progress updates
- Persistent memory integration
"""

import argparse
import asyncio
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiofiles

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_swarm_brain import PhiBrain, BrainConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JSONStreamEmitter:
    """Emits training events as JSON to stdout and/or file"""
    
    def __init__(self, output_file: Optional[str] = None, pretty: bool = False):
        self.output_file = output_file
        self.pretty = pretty
        self.event_count = 0
        
    async def emit(self, event_type: str, data: Dict[str, Any]):
        """Emit a JSON event"""
        event = {
            "event_id": self.event_count,
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.event_count += 1
        
        # Format JSON
        if self.pretty:
            json_str = json.dumps(event, indent=2)
        else:
            json_str = json.dumps(event)
        
        # Emit to stdout
        print(json_str, flush=True)
        
        # Emit to file if specified
        if self.output_file:
            async with aiofiles.open(self.output_file, 'a') as f:
                await f.write(json_str + '\n')
    
    async def training_start(self, config: Dict):
        """Emit training start event"""
        await self.emit("training_start", {
            "neurons": config.get("neurons"),
            "api_provider": config.get("api_provider"),
            "model": config.get("model"),
            "max_steps": config.get("max_steps"),
            "curriculum_size": config.get("curriculum_size")
        })
    
    async def level_start(self, level_name: str, level_info: Dict):
        """Emit level start event"""
        await self.emit("level_start", {
            "level_name": level_name,
            "level_info": level_info
        })
    
    async def question_start(self, question_num: int, total: int, question: str):
        """Emit question start event"""
        await self.emit("question_start", {
            "question_number": question_num,
            "total_questions": total,
            "question": question[:200] + "..." if len(question) > 200 else question
        })
    
    async def question_complete(self, question_num: int, result: Dict):
        """Emit question completion event"""
        await self.emit("question_complete", {
            "question_number": question_num,
            "consciousness": result.get("consciousness"),
            "integration": result.get("integration"),
            "coherence": result.get("coherence"),
            "duration_seconds": result.get("duration")
        })
    
    async def level_complete(self, level_name: str, results: Dict):
        """Emit level completion event"""
        await self.emit("level_complete", {
            "level_name": level_name,
            "avg_consciousness": results.get("avg_consciousness"),
            "avg_integration": results.get("avg_integration"),
            "avg_coherence": results.get("avg_coherence"),
            "total_duration": results.get("total_duration"),
            "questions_completed": len(results.get("questions", []))
        })
    
    async def training_complete(self, report: Dict):
        """Emit training completion event"""
        await self.emit("training_complete", {
            "summary": report.get("training_summary"),
            "performance": report.get("performance_metrics"),
            "learning_progression": report.get("learning_progression"),
            "persistent_memory": report.get("persistent_memory")
        })
    
    async def progress_update(self, progress: float, message: str):
        """Emit progress update"""
        await self.emit("progress", {
            "progress_percent": progress,
            "message": message
        })
    
    async def error(self, error_type: str, message: str, details: Optional[Dict] = None):
        """Emit error event"""
        await self.emit("error", {
            "error_type": error_type,
            "message": message,
            "details": details or {}
        })


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


class AsyncNetworkTrainer:
    """Async trainer with JSON streaming"""
    
    def __init__(
        self,
        brain: PhiBrain,
        curriculum: TrainingCurriculum,
        emitter: JSONStreamEmitter,
        checkpoint_dir: str = "training_checkpoints",
        memory_file: str = "training_memory.pkl",
        load_memory: bool = True
    ):
        self.brain = brain
        self.curriculum = curriculum
        self.emitter = emitter
        self.checkpoint_dir = Path(checkpoint_dir)
        self.memory_file = self.checkpoint_dir / memory_file
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training metrics
        self.training_history = {
            "levels": [],
            "questions": [],
            "consciousness_progression": [],
            "integration_progression": [],
            "coherence_progression": []
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
            "session_history": []
        }
        
        # Load previous memory if exists
        if load_memory:
            self._load_memory()
    
    def _load_memory(self):
        """Load persistent memory from previous training sessions"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'rb') as f:
                    loaded_memory = pickle.load(f)
                    self.persistent_memory.update(loaded_memory)
                logger.info(f"Loaded persistent memory: {self.persistent_memory['total_questions_processed']} previous questions")
            except Exception as e:
                logger.warning(f"Could not load memory: {e}")
    
    def _save_memory(self):
        """Save persistent memory"""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.persistent_memory, f)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def _update_persistent_memory(self, level_results: Dict):
        """Update persistent memory with new training data"""
        self.persistent_memory["total_training_time"] += level_results["total_duration"]
        self.persistent_memory["total_questions_processed"] += len(level_results["questions"])
        
        # Update best performance
        if level_results["avg_consciousness"] > self.persistent_memory["best_performance"]["consciousness"]:
            self.persistent_memory["best_performance"]["consciousness"] = level_results["avg_consciousness"]
        if level_results["avg_integration"] > self.persistent_memory["best_performance"]["integration"]:
            self.persistent_memory["best_performance"]["integration"] = level_results["avg_integration"]
        if level_results["avg_coherence"] > self.persistent_memory["best_performance"]["coherence"]:
            self.persistent_memory["best_performance"]["coherence"] = level_results["avg_coherence"]
        
        # Add to session history
        session_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level_results["level_name"],
            "avg_consciousness": level_results["avg_consciousness"],
            "avg_integration": level_results["avg_integration"],
            "avg_coherence": level_results["avg_coherence"]
        }
        self.persistent_memory["session_history"].append(session_entry)
    
    async def train_level(
        self,
        level_name: str,
        max_steps: int = 3
    ) -> Dict[str, Any]:
        """Train on all questions in a level"""
        level = self.curriculum.get_level(level_name)
        if not level:
            raise ValueError(f"Unknown level: {level_name}")
        
        await self.emitter.level_start(level_name, level)
        
        level_results = {
            "level_name": level_name,
            "level_info": level,
            "questions": [],
            "avg_consciousness": 0.0,
            "avg_integration": 0.0,
            "avg_coherence": 0.0,
            "total_duration": 0.0
        }
        
        total_questions = len(level["questions"])
        
        for i, question in enumerate(level["questions"], 1):
            await self.emitter.question_start(i, total_questions, question)
            
            # Process question (run in executor to not block)
            start_time = datetime.now()
            result = await asyncio.to_thread(
                self.brain.think,
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
            
            # Emit completion
            await self.emitter.question_complete(i, question_result)
            
            # Progress update
            progress = (i / total_questions) * 100
            await self.emitter.progress_update(progress, f"Level {level_name}: {i}/{total_questions} questions")
        
        # Calculate level averages
        level_results["avg_consciousness"] = sum(q["consciousness"] for q in level_results["questions"]) / len(level_results["questions"])
        level_results["avg_integration"] = sum(q["integration"] for q in level_results["questions"]) / len(level_results["questions"])
        level_results["avg_coherence"] = sum(q["coherence"] for q in level_results["questions"]) / len(level_results["questions"])
        level_results["total_duration"] = sum(q["duration"] for q in level_results["questions"])
        
        # Store level results
        self.training_history["levels"].append(level_results)
        
        # Update persistent memory
        self._update_persistent_memory(level_results)
        self._save_memory()
        
        # Emit completion
        await self.emitter.level_complete(level_name, level_results)
        
        return level_results
    
    async def train_full_curriculum(
        self,
        max_steps: int = 3,
        start_level: str = None
    ) -> Dict[str, Any]:
        """Train on the full curriculum"""
        levels = self.curriculum.get_all_levels()
        
        if start_level:
            if start_level not in levels:
                raise ValueError(f"Unknown start level: {start_level}")
            start_idx = levels.index(start_level)
            levels = levels[start_idx:]
        
        # Train each level
        for level_name in levels:
            await self.train_level(level_name, max_steps=max_steps)
        
        # Generate final report
        report = self._generate_training_report()
        
        # Emit completion
        await self.emitter.training_complete(report)
        
        # Save final results
        await self._save_final_results(report)
        
        return report
    
    def _generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        all_consciousness = self.training_history["consciousness_progression"]
        all_integration = self.training_history["integration_progression"]
        all_coherence = self.training_history["coherence_progression"]
        
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
                "total_training_sessions": len(self.persistent_memory["session_history"])
            },
            "level_performance": self.training_history["levels"]
        }
        
        return report
    
    async def _save_final_results(self, report: Dict):
        """Save final training results"""
        results_file = self.checkpoint_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        async with aiofiles.open(results_file, 'w') as f:
            await f.write(json.dumps(report, indent=2))
        
        logger.info(f"Final results saved: {results_file}")


async def main():
    """Main async training execution"""
    parser = argparse.ArgumentParser(description="Async Train LLM-Swarm-Brain with JSON streaming")
    parser.add_argument("--neurons", type=int, default=8, choices=[8, 64, 128], help="Number of neurons")
    parser.add_argument("--api-provider", type=str, default="gemini", choices=["hyperbolic", "gemini"], help="API provider")
    parser.add_argument("--gemini-model", type=str, default="gemini-exp-1206", help="Gemini model name")
    parser.add_argument("--hyperbolic-model", type=str, default="meta-llama/Meta-Llama-3.1-405B-Instruct", help="Hyperbolic model name")
    parser.add_argument("--max-steps", type=int, default=3, help="Max reasoning steps per question")
    parser.add_argument("--start-level", type=str, help="Start from specific level")
    parser.add_argument("--single-level", type=str, help="Train only on a single level")
    parser.add_argument("--output-file", type=str, help="Save JSON stream to file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--no-memory", action="store_true", help="Don't load/save persistent memory")
    parser.add_argument("--reset-memory", action="store_true", help="Reset persistent memory")
    parser.add_argument("--api-key", type=str, help="API key")
    
    args = parser.parse_args()
    
    try:
        # Initialize JSON emitter
        emitter = JSONStreamEmitter(output_file=args.output_file, pretty=args.pretty)
        
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
        
        # Emit training start
        curriculum = TrainingCurriculum()
        await emitter.training_start({
            "neurons": args.neurons,
            "api_provider": args.api_provider,
            "model": args.gemini_model if args.api_provider == "gemini" else args.hyperbolic_model,
            "max_steps": args.max_steps,
            "curriculum_size": curriculum.get_total_questions()
        })
        
        # Initialize brain (in thread to not block)
        brain = await asyncio.to_thread(
            PhiBrain,
            config=config,
            load_models=False,
            use_api=True,
            api_key=args.api_key,
            use_64_neurons=(args.neurons == 64),
            api_provider=args.api_provider
        )
        
        # Handle memory reset
        memory_dir = Path("training_checkpoints")
        memory_file = memory_dir / "training_memory.pkl"
        if args.reset_memory and memory_file.exists():
            memory_file.unlink()
        
        # Initialize trainer
        trainer = AsyncNetworkTrainer(
            brain=brain,
            curriculum=curriculum,
            emitter=emitter,
            load_memory=not args.no_memory
        )
        
        # Train
        if args.single_level:
            await trainer.train_level(args.single_level, max_steps=args.max_steps)
        else:
            await trainer.train_full_curriculum(
                max_steps=args.max_steps,
                start_level=args.start_level
            )
        
        return 0
        
    except KeyboardInterrupt:
        await emitter.error("interrupted", "Training interrupted by user")
        return 1
    except Exception as e:
        await emitter.error("exception", str(e), {"traceback": str(e)})
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
