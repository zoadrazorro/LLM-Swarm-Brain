#!/usr/bin/env python3
"""
Budget-Optimized Training Script for LLM-Swarm-Brain

This script maximizes learning efficiency within a strict budget by:
- Using GPT-OSS 20B (10x cheaper than Llama 405B)
- Optimized question selection (fewer, higher-impact questions)
- Reduced max steps (2 instead of 3-5)
- Smart checkpointing to avoid re-running
- Real-time cost tracking

Budget: $14.86 Hyperbolic credit
Strategy: 8-neuron with GPT-OSS for maximum questions within budget
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

sys.path.insert(0, str(Path(__file__).parent))

from llm_swarm_brain import PhiBrain, BrainConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BudgetTracker:
    """Track API costs in real-time"""
    
    # Hyperbolic pricing (per 1M tokens)
    PRICING = {
        "openai/gpt-oss-20b": 0.10,  # $0.10 per 1M tokens
        "meta-llama/Meta-Llama-3.1-70B-Instruct": 0.20,
        "meta-llama/Meta-Llama-3.1-405B-Instruct": 0.40
    }
    
    def __init__(self, budget: float, model: str):
        self.budget = budget
        self.model = model
        self.spent = 0.0
        self.tokens_used = 0
        self.api_calls = 0
        
        # Estimate tokens per API call (conservative)
        self.avg_tokens_per_call = 1000  # ~1K tokens per neuron fire
    
    def estimate_cost(self, api_calls: int) -> float:
        """Estimate cost for N API calls"""
        tokens = api_calls * self.avg_tokens_per_call
        cost_per_token = self.PRICING.get(self.model, 0.10) / 1_000_000
        return tokens * cost_per_token
    
    def record_api_call(self):
        """Record an API call"""
        self.api_calls += 1
        self.tokens_used += self.avg_tokens_per_call
        self.spent = self.estimate_cost(self.api_calls)
    
    def can_afford(self, api_calls: int) -> bool:
        """Check if we can afford N more API calls"""
        estimated_cost = self.estimate_cost(api_calls)
        return (self.spent + estimated_cost) <= self.budget
    
    def remaining_budget(self) -> float:
        """Get remaining budget"""
        return self.budget - self.spent
    
    def get_status(self) -> Dict:
        """Get budget status"""
        return {
            "budget": self.budget,
            "spent": self.spent,
            "remaining": self.remaining_budget(),
            "percent_used": (self.spent / self.budget * 100) if self.budget > 0 else 0,
            "api_calls": self.api_calls,
            "tokens_used": self.tokens_used
        }


class OptimizedCurriculum:
    """Budget-optimized curriculum with high-impact questions"""
    
    def __init__(self):
        # Fewer, more impactful questions
        self.curriculum = {
            "level_1_foundation": {
                "name": "Foundation (Core Concepts)",
                "complexity": 0.3,
                "questions": [
                    "What is consciousness?",
                    "Explain the mind-body problem.",
                    "What is the difference between knowledge and belief?"
                ],
                "estimated_api_calls": 24  # 8 neurons × 1 step × 3 questions
            },
            "level_2_reasoning": {
                "name": "Reasoning (Logic & Analysis)",
                "complexity": 0.5,
                "questions": [
                    "Compare deductive and inductive reasoning.",
                    "Explain the problem of induction.",
                    "What is the Chinese Room argument?"
                ],
                "estimated_api_calls": 32  # 8 neurons × 1.5 steps × 3 questions (avg)
            },
            "level_3_integration": {
                "name": "Integration (Complex Synthesis)",
                "complexity": 0.7,
                "questions": [
                    "How does the hard problem of consciousness relate to functionalism?",
                    "Can compatibilism reconcile free will and determinism?"
                ],
                "estimated_api_calls": 32  # 8 neurons × 2 steps × 2 questions
            }
        }
    
    def get_level(self, level_name: str) -> Dict:
        return self.curriculum.get(level_name, {})
    
    def get_all_levels(self) -> List[str]:
        return list(self.curriculum.keys())
    
    def get_total_questions(self) -> int:
        return sum(len(level["questions"]) for level in self.curriculum.values())
    
    def estimate_total_api_calls(self) -> int:
        return sum(level["estimated_api_calls"] for level in self.curriculum.values())


class BudgetTrainer:
    """Budget-conscious trainer"""
    
    def __init__(
        self,
        brain: PhiBrain,
        curriculum: OptimizedCurriculum,
        budget_tracker: BudgetTracker,
        checkpoint_dir: str = "training_checkpoints"
    ):
        self.brain = brain
        self.curriculum = curriculum
        self.budget = budget_tracker
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.results = {
            "levels": [],
            "total_questions": 0,
            "total_duration": 0.0,
            "budget_status": {}
        }
    
    def train_level(self, level_name: str, max_steps: int = 2) -> Dict:
        """Train on a level with budget checking"""
        level = self.curriculum.get_level(level_name)
        if not level:
            raise ValueError(f"Unknown level: {level_name}")
        
        # Check if we can afford this level
        if not self.budget.can_afford(level["estimated_api_calls"]):
            logger.warning(f"⚠️ Insufficient budget for {level_name}")
            logger.warning(f"   Estimated cost: ${self.budget.estimate_cost(level['estimated_api_calls']):.2f}")
            logger.warning(f"   Remaining: ${self.budget.remaining_budget():.2f}")
            return None
        
        logger.info(f"\n{'='*80}")
        logger.info(f"LEVEL: {level['name']}")
        logger.info(f"Questions: {len(level['questions'])}")
        logger.info(f"Estimated API calls: {level['estimated_api_calls']}")
        logger.info(f"Estimated cost: ${self.budget.estimate_cost(level['estimated_api_calls']):.2f}")
        logger.info(f"{'='*80}\n")
        
        level_results = {
            "level_name": level_name,
            "questions": [],
            "avg_consciousness": 0.0,
            "avg_integration": 0.0,
            "avg_coherence": 0.0,
            "total_duration": 0.0,
            "api_calls_used": 0
        }
        
        for i, question in enumerate(level["questions"], 1):
            logger.info(f"[Q{i}/{len(level['questions'])}] {question}")
            
            # Process with budget tracking
            start_time = datetime.now()
            api_calls_before = self.budget.api_calls
            
            result = self.brain.think(
                question,
                max_steps=max_steps,
                use_memory=True,
                use_global_workspace=True,
                enable_enhancements=True
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Estimate API calls (neurons fired)
            neurons_fired = result.get("brain_metrics", {}).get("total_firings", 8)
            for _ in range(neurons_fired):
                self.budget.record_api_call()
            
            api_calls_used = self.budget.api_calls - api_calls_before
            
            # Extract metrics
            consciousness = result.get("consciousness_level", 0.0)
            integration = result.get("global_workspace", {}).get("integration_score", 0.0)
            coherence = result.get("positronic_coherence", {}).get("score", 0.0)
            
            logger.info(f"✓ Completed in {duration:.1f}s")
            logger.info(f"  Consciousness: {consciousness:.3f}")
            logger.info(f"  Integration: {integration:.3f}")
            logger.info(f"  Coherence: {coherence:.3f}")
            logger.info(f"  API calls: {api_calls_used}")
            logger.info(f"  Cost so far: ${self.budget.spent:.2f} / ${self.budget.budget:.2f}")
            logger.info(f"  Remaining: ${self.budget.remaining_budget():.2f}\n")
            
            level_results["questions"].append({
                "question": question,
                "consciousness": consciousness,
                "integration": integration,
                "coherence": coherence,
                "duration": duration,
                "api_calls": api_calls_used
            })
            
            level_results["api_calls_used"] += api_calls_used
            
            # Check budget after each question
            if self.budget.remaining_budget() < 1.0:  # Less than $1 left
                logger.warning("⚠️ Budget running low! Stopping level early.")
                break
        
        # Calculate averages
        if level_results["questions"]:
            level_results["avg_consciousness"] = sum(q["consciousness"] for q in level_results["questions"]) / len(level_results["questions"])
            level_results["avg_integration"] = sum(q["integration"] for q in level_results["questions"]) / len(level_results["questions"])
            level_results["avg_coherence"] = sum(q["coherence"] for q in level_results["questions"]) / len(level_results["questions"])
            level_results["total_duration"] = sum(q["duration"] for q in level_results["questions"])
        
        logger.info(f"\n{'='*80}")
        logger.info(f"LEVEL COMPLETE: {level['name']}")
        logger.info(f"  Avg Consciousness: {level_results['avg_consciousness']:.3f}")
        logger.info(f"  Avg Integration: {level_results['avg_integration']:.3f}")
        logger.info(f"  Avg Coherence: {level_results['avg_coherence']:.3f}")
        logger.info(f"  API calls: {level_results['api_calls_used']}")
        logger.info(f"  Duration: {level_results['total_duration']:.1f}s")
        logger.info(f"{'='*80}\n")
        
        self.results["levels"].append(level_results)
        self.results["total_questions"] += len(level_results["questions"])
        self.results["total_duration"] += level_results["total_duration"]
        
        return level_results
    
    def train_full_curriculum(self, max_steps: int = 2) -> Dict:
        """Train on full curriculum within budget"""
        logger.info(f"\n{'#'*80}")
        logger.info("BUDGET-OPTIMIZED TRAINING")
        logger.info(f"Budget: ${self.budget.budget:.2f}")
        logger.info(f"Model: {self.budget.model}")
        logger.info(f"Total Questions: {self.curriculum.get_total_questions()}")
        logger.info(f"Estimated API calls: {self.curriculum.estimate_total_api_calls()}")
        logger.info(f"Estimated cost: ${self.budget.estimate_cost(self.curriculum.estimate_total_api_calls()):.2f}")
        logger.info(f"{'#'*80}\n")
        
        # Check if we can afford full curriculum
        total_estimated_cost = self.budget.estimate_cost(self.curriculum.estimate_total_api_calls())
        if total_estimated_cost > self.budget.budget:
            logger.warning(f"⚠️ Full curriculum may exceed budget!")
            logger.warning(f"   Estimated: ${total_estimated_cost:.2f}")
            logger.warning(f"   Budget: ${self.budget.budget:.2f}")
            logger.warning(f"   Will stop when budget runs out.\n")
        
        # Train each level
        for level_name in self.curriculum.get_all_levels():
            if self.budget.remaining_budget() < 0.50:  # Less than $0.50 left
                logger.warning("⚠️ Budget exhausted! Stopping training.")
                break
            
            result = self.train_level(level_name, max_steps=max_steps)
            if result is None:  # Couldn't afford level
                break
        
        # Final report
        self.results["budget_status"] = self.budget.get_status()
        
        logger.info(f"\n{'#'*80}")
        logger.info("TRAINING COMPLETE")
        logger.info(f"{'#'*80}")
        logger.info(f"Questions Completed: {self.results['total_questions']}")
        logger.info(f"Total Duration: {self.results['total_duration']/60:.1f} minutes")
        logger.info(f"\nBudget Status:")
        logger.info(f"  Budget: ${self.budget.budget:.2f}")
        logger.info(f"  Spent: ${self.budget.spent:.2f}")
        logger.info(f"  Remaining: ${self.budget.remaining_budget():.2f}")
        logger.info(f"  Utilization: {self.budget.get_status()['percent_used']:.1f}%")
        logger.info(f"  API Calls: {self.budget.api_calls}")
        logger.info(f"  Tokens: {self.budget.tokens_used:,}")
        
        if self.results["levels"]:
            avg_consciousness = sum(l["avg_consciousness"] for l in self.results["levels"]) / len(self.results["levels"])
            avg_integration = sum(l["avg_integration"] for l in self.results["levels"]) / len(self.results["levels"])
            avg_coherence = sum(l["avg_coherence"] for l in self.results["levels"]) / len(self.results["levels"])
            
            logger.info(f"\nPerformance:")
            logger.info(f"  Avg Consciousness: {avg_consciousness:.3f}")
            logger.info(f"  Avg Integration: {avg_integration:.3f}")
            logger.info(f"  Avg Coherence: {avg_coherence:.3f}")
        
        logger.info(f"{'#'*80}\n")
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save training results"""
        results_file = self.checkpoint_dir / f"budget_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved: {results_file}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Budget-optimized training for LLM-Swarm-Brain")
    parser.add_argument("--budget", type=float, default=14.86, help="Budget in USD (default: 14.86)")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", 
                       choices=["openai/gpt-oss-20b", "meta-llama/Meta-Llama-3.1-70B-Instruct"],
                       help="Hyperbolic model (default: gpt-oss-20b for max efficiency)")
    parser.add_argument("--neurons", type=int, default=8, choices=[8, 64],
                       help="Number of neurons (default: 8 for budget efficiency)")
    parser.add_argument("--max-steps", type=int, default=2, help="Max steps per question (default: 2)")
    parser.add_argument("--api-key", type=str, help="Hyperbolic API key")
    
    args = parser.parse_args()
    
    try:
        logger.info(f"\n{'='*80}")
        logger.info("BUDGET-OPTIMIZED TRAINING SETUP")
        logger.info(f"{'='*80}")
        logger.info(f"Budget: ${args.budget:.2f}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Neurons: {args.neurons}")
        logger.info(f"Max Steps: {args.max_steps}")
        logger.info(f"{'='*80}\n")
        
        # Initialize budget tracker
        budget = BudgetTracker(args.budget, args.model)
        
        # Load config
        if args.neurons == 64:
            from llm_swarm_brain import config_64
            config = config_64.BrainConfig()
        else:
            config = BrainConfig()
        
        config.hyperbolic_model_name = args.model
        
        # Initialize brain
        logger.info("[1] Initializing brain...")
        brain = PhiBrain(
            config=config,
            load_models=False,
            use_api=True,
            api_key=args.api_key,
            use_64_neurons=(args.neurons == 64),
            api_provider="hyperbolic"
        )
        logger.info("✓ Brain initialized\n")
        
        # Initialize curriculum and trainer
        logger.info("[2] Loading optimized curriculum...")
        curriculum = OptimizedCurriculum()
        trainer = BudgetTrainer(brain, curriculum, budget)
        logger.info(f"✓ Curriculum loaded ({curriculum.get_total_questions()} questions)\n")
        
        # Train
        logger.info("[3] Starting training...\n")
        results = trainer.train_full_curriculum(max_steps=args.max_steps)
        
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
