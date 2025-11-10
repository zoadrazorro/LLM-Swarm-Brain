"""
Positronic Dialectical Logic-Gated Coherence Framework

Implements:
- Dialectical reasoning (thesis-antithesis-synthesis)
- Logic gates for neuron signal processing
- Coherence validation and checking
- Positronic laws and ethical constraints
- Multi-valued logic operations

Inspired by Asimov's positronic brain and dialectical logic.
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import logging

from llm_swarm_brain.neuron import NeuronSignal


logger = logging.getLogger(__name__)


class LogicGate(Enum):
    """Logic gate types for signal processing"""
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"
    NAND = "nand"
    NOR = "nor"
    IMPLICATION = "implication"
    EQUIVALENCE = "equivalence"
    # Fuzzy logic variants
    FUZZY_AND = "fuzzy_and"  # min
    FUZZY_OR = "fuzzy_or"    # max
    DIALECTICAL_SYNTHESIS = "dialectical_synthesis"  # mean + innovation


class PositronicLaw(Enum):
    """Positronic laws (inspired by Asimov's Three Laws)"""
    FIRST_LAW = "first_law"      # Coherence and truth preservation
    SECOND_LAW = "second_law"    # Logic consistency
    THIRD_LAW = "third_law"      # Self-consistency and stability


@dataclass
class DialecticalTriad:
    """Represents a dialectical thesis-antithesis-synthesis structure"""
    thesis: str
    antithesis: str
    synthesis: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoherenceReport:
    """Report on logical coherence of outputs"""
    is_coherent: bool
    coherence_score: float
    violations: List[str] = field(default_factory=list)
    logical_consistency: float = 0.0
    dialectical_resolution: Optional[str] = None


class LogicGateProcessor:
    """
    Processes signals through logic gates

    Implements both classical and fuzzy logic operations
    """

    @staticmethod
    def apply_gate(
        gate_type: LogicGate,
        signal_a: float,
        signal_b: Optional[float] = None
    ) -> float:
        """
        Apply logic gate to signal(s)

        Args:
            gate_type: Type of logic gate
            signal_a: First signal (0-1 range)
            signal_b: Second signal if binary gate

        Returns:
            Result signal (0-1 range)
        """
        # Ensure signals are in valid range
        signal_a = np.clip(signal_a, 0.0, 1.0)
        if signal_b is not None:
            signal_b = np.clip(signal_b, 0.0, 1.0)

        if gate_type == LogicGate.NOT:
            return 1.0 - signal_a

        if signal_b is None:
            raise ValueError(f"Gate {gate_type} requires two signals")

        # Binary gates
        if gate_type == LogicGate.AND:
            return float(signal_a > 0.5 and signal_b > 0.5)

        elif gate_type == LogicGate.OR:
            return float(signal_a > 0.5 or signal_b > 0.5)

        elif gate_type == LogicGate.XOR:
            return float((signal_a > 0.5) != (signal_b > 0.5))

        elif gate_type == LogicGate.NAND:
            return 1.0 - float(signal_a > 0.5 and signal_b > 0.5)

        elif gate_type == LogicGate.NOR:
            return 1.0 - float(signal_a > 0.5 or signal_b > 0.5)

        elif gate_type == LogicGate.IMPLICATION:
            # A → B ≡ ¬A ∨ B
            return float(signal_a <= 0.5 or signal_b > 0.5)

        elif gate_type == LogicGate.EQUIVALENCE:
            # A ↔ B
            return float((signal_a > 0.5) == (signal_b > 0.5))

        # Fuzzy logic gates
        elif gate_type == LogicGate.FUZZY_AND:
            return min(signal_a, signal_b)

        elif gate_type == LogicGate.FUZZY_OR:
            return max(signal_a, signal_b)

        elif gate_type == LogicGate.DIALECTICAL_SYNTHESIS:
            # Synthesis: mean + innovation factor
            mean = (signal_a + signal_b) / 2.0
            # Add creative synthesis (go beyond simple average)
            innovation = abs(signal_a - signal_b) * 0.2
            return np.clip(mean + innovation, 0.0, 1.0)

        else:
            raise ValueError(f"Unknown gate type: {gate_type}")


class DialecticalReasoner:
    """
    Implements dialectical reasoning: thesis → antithesis → synthesis

    Processes information through contradictions to reach higher understanding.
    """

    def __init__(self):
        self.triads: List[DialecticalTriad] = []
        self.synthesis_count = 0

    def generate_antithesis(
        self,
        thesis: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate antithesis for a thesis

        This would ideally use an LLM to generate contradictory position.
        For now, we implement a rule-based approximation.

        Args:
            thesis: The thesis statement
            context: Additional context

        Returns:
            Antithesis statement
        """
        # Simplified implementation
        # In full version, this would use a neuron to generate opposition

        negation_markers = [
            ("is", "is not"),
            ("should", "should not"),
            ("must", "must not"),
            ("can", "cannot"),
            ("will", "will not"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("possible", "impossible"),
        ]

        antithesis = thesis
        for pos, neg in negation_markers:
            if pos in thesis.lower():
                antithesis = thesis.replace(pos, neg)
                break
            elif neg in thesis.lower():
                antithesis = thesis.replace(neg, pos)
                break

        # If no negation found, use general contradiction
        if antithesis == thesis:
            antithesis = f"On the contrary, {thesis.lower()}"

        return antithesis

    def synthesize(
        self,
        thesis: str,
        antithesis: str,
        signal_a: float = 1.0,
        signal_b: float = 1.0
    ) -> DialecticalTriad:
        """
        Synthesize thesis and antithesis into higher-order understanding

        Args:
            thesis: Thesis statement
            antithesis: Antithesis statement
            signal_a: Strength of thesis
            signal_b: Strength of antithesis

        Returns:
            Dialectical triad with synthesis
        """
        # Use dialectical synthesis gate for confidence
        gate_processor = LogicGateProcessor()
        confidence = gate_processor.apply_gate(
            LogicGate.DIALECTICAL_SYNTHESIS,
            signal_a,
            signal_b
        )

        # Generate synthesis
        # In full version, this would use a specialized synthesis neuron
        synthesis = self._generate_synthesis(thesis, antithesis)

        triad = DialecticalTriad(
            thesis=thesis,
            antithesis=antithesis,
            synthesis=synthesis,
            confidence=confidence,
            metadata={
                "thesis_strength": signal_a,
                "antithesis_strength": signal_b,
                "synthesis_id": self.synthesis_count
            }
        )

        self.triads.append(triad)
        self.synthesis_count += 1

        logger.info(f"Dialectical synthesis #{self.synthesis_count} created (confidence={confidence:.3f})")

        return triad

    def _generate_synthesis(self, thesis: str, antithesis: str) -> str:
        """
        Generate synthesis from thesis and antithesis

        Args:
            thesis: Thesis statement
            antithesis: Antithesis statement

        Returns:
            Synthesis statement
        """
        # Simplified synthesis generation
        # Full version would use LLM
        synthesis = (
            f"Integrating both perspectives: {thesis[:50]}... "
            f"and {antithesis[:50]}... we arrive at a higher understanding "
            f"that reconciles both positions through their complementary aspects."
        )
        return synthesis

    def get_dialectical_chain(self, depth: int = 3) -> List[DialecticalTriad]:
        """
        Get recent dialectical reasoning chain

        Args:
            depth: Number of triads to return

        Returns:
            List of recent triads
        """
        return self.triads[-depth:] if self.triads else []


class CoherenceValidator:
    """
    Validates logical coherence of neuron outputs

    Implements positronic laws to ensure consistent, coherent processing
    """

    def __init__(
        self,
        coherence_threshold: float = 0.7,
        enforce_positronic_laws: bool = True
    ):
        self.coherence_threshold = coherence_threshold
        self.enforce_positronic_laws = enforce_positronic_laws
        self.validation_count = 0
        self.violation_history: List[str] = []

    def validate(
        self,
        outputs: Dict[str, str],
        activations: Dict[str, float],
        prior_context: Optional[Dict[str, Any]] = None
    ) -> CoherenceReport:
        """
        Validate coherence of neuron outputs

        Args:
            outputs: Neuron outputs to validate
            activations: Neuron activation levels
            prior_context: Prior context for consistency checking

        Returns:
            Coherence report
        """
        self.validation_count += 1

        violations = []
        coherence_scores = []

        # Check First Law: Truth preservation and coherence
        first_law_score = self._check_first_law(outputs, activations)
        coherence_scores.append(first_law_score)
        if first_law_score < self.coherence_threshold:
            violations.append("First Law violation: Insufficient coherence in outputs")

        # Check Second Law: Logic consistency
        second_law_score = self._check_second_law(outputs, activations)
        coherence_scores.append(second_law_score)
        if second_law_score < self.coherence_threshold:
            violations.append("Second Law violation: Logical inconsistencies detected")

        # Check Third Law: Self-consistency
        third_law_score = self._check_third_law(outputs, prior_context)
        coherence_scores.append(third_law_score)
        if third_law_score < self.coherence_threshold:
            violations.append("Third Law violation: Inconsistent with prior state")

        # Calculate overall coherence
        overall_coherence = np.mean(coherence_scores)
        is_coherent = overall_coherence >= self.coherence_threshold and not violations

        # Store violations
        if violations:
            self.violation_history.extend(violations)

        report = CoherenceReport(
            is_coherent=is_coherent,
            coherence_score=overall_coherence,
            violations=violations,
            logical_consistency=second_law_score
        )

        if not is_coherent:
            logger.warning(
                f"Coherence validation #{self.validation_count} FAILED: "
                f"score={overall_coherence:.3f}, violations={len(violations)}"
            )
        else:
            logger.debug(f"Coherence validation #{self.validation_count} passed")

        return report

    def _check_first_law(
        self,
        outputs: Dict[str, str],
        activations: Dict[str, float]
    ) -> float:
        """
        First Law: Coherence and truth preservation

        Check that outputs are coherent and activations are reasonable
        """
        if not outputs:
            return 0.0

        # Check activation consistency
        activation_variance = np.var(list(activations.values()))
        activation_score = 1.0 - min(1.0, activation_variance)

        # Check output presence (neurons that fired should have outputs)
        output_consistency = len(outputs) / max(1, len(activations))

        return (activation_score + output_consistency) / 2.0

    def _check_second_law(
        self,
        outputs: Dict[str, str],
        activations: Dict[str, float]
    ) -> float:
        """
        Second Law: Logic consistency

        Check for logical contradictions between outputs
        """
        if len(outputs) < 2:
            return 1.0  # Can't have contradictions with one output

        # Simplified check: look for explicit contradictions
        output_texts = list(outputs.values())

        contradiction_markers = [
            ("yes", "no"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("possible", "impossible"),
        ]

        contradictions = 0
        comparisons = 0

        for i, text_a in enumerate(output_texts):
            for text_b in output_texts[i+1:]:
                comparisons += 1
                text_a_lower = text_a.lower()
                text_b_lower = text_b.lower()

                for pos, neg in contradiction_markers:
                    if (pos in text_a_lower and neg in text_b_lower) or \
                       (neg in text_a_lower and pos in text_b_lower):
                        contradictions += 1
                        break

        if comparisons == 0:
            return 1.0

        consistency = 1.0 - (contradictions / comparisons)
        return consistency

    def _check_third_law(
        self,
        outputs: Dict[str, str],
        prior_context: Optional[Dict[str, Any]]
    ) -> float:
        """
        Third Law: Self-consistency and stability

        Check consistency with prior state
        """
        if not prior_context:
            return 1.0  # No prior context to compare

        # Check if current outputs align with prior context
        # Simplified: just check if we have outputs
        if not outputs:
            return 0.5

        return 0.9  # Assume high self-consistency by default


class PositronicFramework:
    """
    Complete positronic dialectical logic-gated coherence framework

    Integrates:
    - Logic gate processing
    - Dialectical reasoning
    - Coherence validation
    - Positronic law enforcement
    """

    def __init__(
        self,
        coherence_threshold: float = 0.7,
        enable_dialectical_reasoning: bool = True,
        enforce_positronic_laws: bool = True
    ):
        self.gate_processor = LogicGateProcessor()
        self.dialectical_reasoner = DialecticalReasoner()
        self.coherence_validator = CoherenceValidator(
            coherence_threshold=coherence_threshold,
            enforce_positronic_laws=enforce_positronic_laws
        )
        self.enable_dialectical = enable_dialectical_reasoning

        logger.info("Initialized PositronicFramework")

    def process_signals(
        self,
        signals: List[Tuple[float, float, LogicGate]],
    ) -> List[float]:
        """
        Process multiple signal pairs through logic gates

        Args:
            signals: List of (signal_a, signal_b, gate_type) tuples

        Returns:
            List of processed signals
        """
        results = []
        for signal_a, signal_b, gate_type in signals:
            result = self.gate_processor.apply_gate(gate_type, signal_a, signal_b)
            results.append(result)
        return results

    def apply_dialectical_reasoning(
        self,
        thesis_output: str,
        thesis_activation: float = 1.0,
        generate_antithesis: bool = True
    ) -> DialecticalTriad:
        """
        Apply dialectical reasoning to an output

        Args:
            thesis_output: The thesis to process
            thesis_activation: Strength of thesis
            generate_antithesis: Whether to auto-generate antithesis

        Returns:
            Dialectical triad
        """
        if not self.enable_dialectical:
            # Return simple pass-through
            return DialecticalTriad(
                thesis=thesis_output,
                antithesis="",
                synthesis=thesis_output,
                confidence=thesis_activation
            )

        # Generate or receive antithesis
        if generate_antithesis:
            antithesis = self.dialectical_reasoner.generate_antithesis(thesis_output)
            antithesis_activation = 0.7  # Default strength
        else:
            antithesis = f"Alternative to: {thesis_output}"
            antithesis_activation = thesis_activation * 0.5

        # Synthesize
        triad = self.dialectical_reasoner.synthesize(
            thesis=thesis_output,
            antithesis=antithesis,
            signal_a=thesis_activation,
            signal_b=antithesis_activation
        )

        return triad

    def validate_coherence(
        self,
        outputs: Dict[str, str],
        activations: Dict[str, float],
        prior_context: Optional[Dict[str, Any]] = None,
        enforce: bool = True
    ) -> CoherenceReport:
        """
        Validate logical coherence

        Args:
            outputs: Neuron outputs
            activations: Activation levels
            prior_context: Prior context
            enforce: Whether to enforce positronic laws

        Returns:
            Coherence report
        """
        report = self.coherence_validator.validate(
            outputs=outputs,
            activations=activations,
            prior_context=prior_context
        )

        if enforce and not report.is_coherent:
            logger.warning(
                f"Positronic Framework: Coherence enforcement triggered. "
                f"Score: {report.coherence_score:.3f}"
            )
            # In a full implementation, this could trigger corrective actions

        return report

    def get_framework_stats(self) -> Dict[str, Any]:
        """Get framework statistics"""
        return {
            "dialectical_syntheses": self.dialectical_reasoner.synthesis_count,
            "coherence_validations": self.coherence_validator.validation_count,
            "total_violations": len(self.coherence_validator.violation_history),
            "recent_violations": self.coherence_validator.violation_history[-5:],
            "dialectical_chain": [
                {
                    "thesis": t.thesis[:50] + "...",
                    "antithesis": t.antithesis[:50] + "...",
                    "synthesis": t.synthesis[:50] + "...",
                    "confidence": t.confidence
                }
                for t in self.dialectical_reasoner.get_dialectical_chain(3)
            ]
        }

    def __repr__(self) -> str:
        return (
            f"PositronicFramework("
            f"syntheses={self.dialectical_reasoner.synthesis_count}, "
            f"validations={self.coherence_validator.validation_count})"
        )
