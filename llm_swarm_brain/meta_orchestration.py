"""
Meta-Orchestration Layer

Dynamically adjusts network parameters based on task complexity,
performance metrics, and system state.

Implements:
- Dynamic threshold adjustment
- Adaptive resource allocation
- Performance-based tuning
- Task complexity estimation
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging

from llm_swarm_brain.utils import EmbeddingManager


logger = logging.getLogger(__name__)


@dataclass
class TaskComplexityEstimate:
    """Estimate of task complexity"""
    complexity_score: float  # 0-1
    factors: Dict[str, float] = field(default_factory=dict)
    recommended_threshold: float = 0.6
    recommended_max_steps: int = 4


@dataclass
class PerformanceMetrics:
    """Performance metrics for meta-orchestration"""
    avg_consciousness_level: float = 0.0
    avg_coherence_score: float = 0.0
    avg_integration_score: float = 0.0
    avg_neurons_activated: float = 0.0
    avg_processing_time: float = 0.0
    success_rate: float = 1.0


class TaskComplexityEstimator:
    """
    Estimates task complexity to guide parameter adjustment

    Analyzes input to determine:
    - Semantic complexity
    - Question depth
    - Required reasoning type
    - Expected output length
    """

    def __init__(self):
        self.embedding_manager = EmbeddingManager(device="cpu")

        # Complexity indicators
        self.complex_markers = [
            'explain', 'analyze', 'compare', 'synthesize', 'evaluate',
            'complex', 'intricate', 'detailed', 'comprehensive', 'elaborate',
            'why', 'how', 'what if', 'implications', 'consequences'
        ]

        self.simple_markers = [
            'yes', 'no', 'is', 'are', 'define', 'list', 'name', 'what is'
        ]

    def estimate_complexity(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskComplexityEstimate:
        """
        Estimate task complexity

        Args:
            input_text: Input text to analyze
            context: Optional context (e.g., prior interactions)

        Returns:
            Task complexity estimate
        """
        factors = {}

        # Factor 1: Text length
        text_length = len(input_text.split())
        length_factor = min(1.0, text_length / 100.0)
        factors['text_length'] = length_factor

        # Factor 2: Complex markers
        text_lower = input_text.lower()
        complex_count = sum(1 for marker in self.complex_markers if marker in text_lower)
        simple_count = sum(1 for marker in self.simple_markers if marker in text_lower)

        marker_factor = min(1.0, max(0.1, (complex_count - simple_count) / 5.0 + 0.5))
        factors['complexity_markers'] = marker_factor

        # Factor 3: Question depth (number of clauses/sentences)
        sentences = input_text.count('.') + input_text.count('?') + input_text.count('!')
        depth_factor = min(1.0, sentences / 5.0)
        factors['question_depth'] = depth_factor

        # Factor 4: Conceptual breadth (unique meaningful words)
        words = set([
            w.strip('.,;:!?"()[]').lower()
            for w in input_text.split()
            if len(w) > 3
        ])
        breadth_factor = min(1.0, len(words) / 20.0)
        factors['conceptual_breadth'] = breadth_factor

        # Factor 5: Context complexity
        context_factor = 0.5  # Default
        if context and 'memory' in context:
            memory = context['memory']
            if memory.get('episodic', []):
                context_factor = 0.7
        factors['context_complexity'] = context_factor

        # Weighted combination
        weights = {
            'text_length': 0.2,
            'complexity_markers': 0.3,
            'question_depth': 0.2,
            'conceptual_breadth': 0.2,
            'context_complexity': 0.1
        }

        complexity_score = sum(
            factors[key] * weights[key]
            for key in factors
        )

        # Determine recommendations based on complexity
        if complexity_score < 0.3:
            # Simple task
            recommended_threshold = 0.7  # Higher threshold, fewer neurons
            recommended_max_steps = 2
        elif complexity_score < 0.6:
            # Medium complexity
            recommended_threshold = 0.6
            recommended_max_steps = 3
        else:
            # High complexity
            recommended_threshold = 0.5  # Lower threshold, more neurons
            recommended_max_steps = 5

        return TaskComplexityEstimate(
            complexity_score=complexity_score,
            factors=factors,
            recommended_threshold=recommended_threshold,
            recommended_max_steps=recommended_max_steps
        )


class MetaOrchestrator:
    """
    Meta-orchestration layer for dynamic parameter tuning

    Monitors system performance and adjusts parameters:
    - Activation thresholds
    - Processing depth
    - Resource allocation
    - Attention focus
    """

    def __init__(
        self,
        adaptation_rate: float = 0.1,
        enable_auto_tuning: bool = True
    ):
        """
        Initialize meta-orchestrator

        Args:
            adaptation_rate: Rate of parameter adaptation (0-1)
            enable_auto_tuning: Enable automatic tuning
        """
        self.adaptation_rate = adaptation_rate
        self.enable_auto_tuning = enable_auto_tuning

        self.complexity_estimator = TaskComplexityEstimator()

        # Current parameters
        self.current_activation_threshold = 0.6
        self.current_max_steps = 4
        self.current_workspace_capacity = 5

        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.adjustment_history: List[Dict[str, Any]] = []

        self.total_adjustments = 0

        logger.info(
            f"Initialized MetaOrchestrator "
            f"(adaptation_rate={adaptation_rate}, auto_tuning={enable_auto_tuning})"
        )

    def analyze_task(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskComplexityEstimate:
        """
        Analyze task complexity

        Args:
            input_text: Input text
            context: Optional context

        Returns:
            Complexity estimate
        """
        return self.complexity_estimator.estimate_complexity(input_text, context)

    def recommend_parameters(
        self,
        complexity_estimate: TaskComplexityEstimate,
        current_performance: Optional[PerformanceMetrics] = None
    ) -> Dict[str, Any]:
        """
        Recommend parameters based on complexity and performance

        Args:
            complexity_estimate: Task complexity estimate
            current_performance: Current performance metrics

        Returns:
            Dictionary of recommended parameters
        """
        recommendations = {
            "activation_threshold": complexity_estimate.recommended_threshold,
            "max_propagation_steps": complexity_estimate.recommended_max_steps,
            "workspace_capacity": 5  # Default
        }

        # Adjust based on performance if available
        if current_performance and self.performance_history:
            recommendations = self._adjust_for_performance(
                recommendations,
                current_performance
            )

        return recommendations

    def _adjust_for_performance(
        self,
        base_recommendations: Dict[str, Any],
        current_performance: PerformanceMetrics
    ) -> Dict[str, Any]:
        """
        Adjust recommendations based on performance

        Args:
            base_recommendations: Base recommendations
            current_performance: Current performance metrics

        Returns:
            Adjusted recommendations
        """
        recommendations = base_recommendations.copy()

        # Get average historical performance
        if len(self.performance_history) < 3:
            return recommendations

        avg_consciousness = np.mean([
            p.avg_consciousness_level for p in self.performance_history[-10:]
        ])
        avg_coherence = np.mean([
            p.avg_coherence_score for p in self.performance_history[-10:]
        ])

        # If consciousness is low, lower threshold (activate more neurons)
        if avg_consciousness < 0.5:
            threshold_adjustment = -0.1
        elif avg_consciousness > 0.8:
            # If consciousness is high, can raise threshold slightly
            threshold_adjustment = 0.05
        else:
            threshold_adjustment = 0.0

        # If coherence is low, might need to reduce propagation steps
        if avg_coherence < 0.6:
            recommendations["max_propagation_steps"] = max(
                2,
                recommendations.get("max_propagation_steps", 4) - 1
            )

        # Apply threshold adjustment
        new_threshold = recommendations["activation_threshold"] + threshold_adjustment
        recommendations["activation_threshold"] = np.clip(new_threshold, 0.4, 0.8)

        return recommendations

    def apply_adjustments(
        self,
        recommendations: Dict[str, Any],
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Apply parameter adjustments

        Args:
            recommendations: Recommended parameters
            force: Force application without gradual adaptation

        Returns:
            Actually applied parameters
        """
        if not self.enable_auto_tuning and not force:
            return {
                "activation_threshold": self.current_activation_threshold,
                "max_propagation_steps": self.current_max_steps,
                "workspace_capacity": self.current_workspace_capacity
            }

        applied = {}

        # Apply activation threshold (with adaptation rate)
        recommended_threshold = recommendations.get(
            "activation_threshold",
            self.current_activation_threshold
        )

        if force:
            new_threshold = recommended_threshold
        else:
            # Gradual adaptation
            new_threshold = (
                self.current_activation_threshold * (1 - self.adaptation_rate) +
                recommended_threshold * self.adaptation_rate
            )

        self.current_activation_threshold = np.clip(new_threshold, 0.4, 0.8)
        applied["activation_threshold"] = self.current_activation_threshold

        # Apply max steps
        recommended_steps = recommendations.get("max_propagation_steps", self.current_max_steps)

        if force or abs(recommended_steps - self.current_max_steps) >= 2:
            self.current_max_steps = recommended_steps
        # Otherwise keep current

        applied["max_propagation_steps"] = self.current_max_steps

        # Apply workspace capacity
        recommended_capacity = recommendations.get("workspace_capacity", self.current_workspace_capacity)
        self.current_workspace_capacity = recommended_capacity
        applied["workspace_capacity"] = self.current_workspace_capacity

        # Record adjustment
        self.adjustment_history.append({
            "timestamp": datetime.now(),
            "applied": applied,
            "recommendations": recommendations
        })
        self.total_adjustments += 1

        logger.info(f"Applied parameter adjustments: {applied}")

        return applied

    def record_performance(
        self,
        metrics: PerformanceMetrics
    ):
        """
        Record performance metrics

        Args:
            metrics: Performance metrics
        """
        self.performance_history.append(metrics)

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return {
            "activation_threshold": self.current_activation_threshold,
            "max_propagation_steps": self.current_max_steps,
            "workspace_capacity": self.current_workspace_capacity
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get meta-orchestration statistics"""
        return {
            "total_adjustments": self.total_adjustments,
            "current_parameters": self.get_current_parameters(),
            "performance_samples": len(self.performance_history),
            "avg_recent_consciousness": (
                np.mean([p.avg_consciousness_level for p in self.performance_history[-10:]])
                if self.performance_history else 0.0
            ),
            "avg_recent_coherence": (
                np.mean([p.avg_coherence_score for p in self.performance_history[-10:]])
                if self.performance_history else 0.0
            )
        }

    def __repr__(self) -> str:
        return (
            f"MetaOrchestrator(adjustments={self.total_adjustments}, "
            f"threshold={self.current_activation_threshold:.2f})"
        )
