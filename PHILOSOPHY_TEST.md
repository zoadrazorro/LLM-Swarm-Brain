# Philosophy Inference Test Battery

Comprehensive philosophical reasoning test for LLM-Swarm-Brain, designed to evaluate the MoE architecture's ability to handle complex philosophical concepts and multi-step reasoning.

## Overview

- **Total Questions**: 40 (5 per level)
- **Complexity Levels**: 8
- **Test Duration**: ~15-30 minutes (full test with models)
- **Quick Test**: ~2-3 minutes (1 question per level, simulation)

## Test Structure

### Level 1: Basic Comprehension
**Ancient & Foundational Concepts**

Tests basic knowledge recall and understanding of classical philosophical concepts.

**Questions:**
1. What is Socrates' definition of knowledge in Plato's Theaetetus?
2. What does Aristotle mean by "the good life" (eudaimonia)?
3. What is Descartes' famous statement about certainty?
4. Is the statement "All bachelors are unmarried" true by definition?
5. What does "tabula rasa" mean in Locke's philosophy?

**Expected Performance:**
- **Experts Activated**: Perception, Memory, Reasoning
- **Propagation Steps**: 2-3
- **Duration**: ~5-10 seconds per question

---

### Level 2: Simple Reasoning
**Basic Philosophical Arguments**

Tests ability to follow simple logical chains and identify basic philosophical reasoning patterns.

**Questions:**
1. If pleasure is good, and virtue produces pleasure, does that make virtue good? (Epicurean reasoning)
2. If we can doubt everything, can we doubt that we are doubting? (Cartesian doubt)
3. If all swans we've observed are white, must all swans be white? (Induction problem)
4. If an action maximizes happiness, is it therefore morally right? (Basic utilitarianism)
5. If nothing changes, can time exist? (Pre-Socratic paradox)

**Expected Performance:**
- **Experts Activated**: Perception, Memory, Reasoning, Analytical
- **Propagation Steps**: 2-3
- **Duration**: ~8-12 seconds per question

---

### Level 3: Pattern Recognition
**Philosophical Patterns & Structures**

Tests ability to identify connections and patterns across different philosophical traditions and concepts.

**Questions:**
1. What pattern connects Plato's Forms, Kant's noumena, and thing-in-itself concepts?
2. How does Hegel's dialectic (thesis-antithesis-synthesis) apply to historical change?
3. What connects Descartes' dualism, Berkeley's idealism, and materialism as responses to the mind-body problem?
4. What pattern underlies skepticism from Pyrrho to Hume to modern epistemology?
5. How do Aristotle's four causes form a complete explanatory pattern?

**Expected Performance:**
- **Experts Activated**: Perception, Memory, Reasoning, Analytical, Synthesis
- **Propagation Steps**: 3
- **Duration**: ~10-15 seconds per question

---

### Level 4: Abstract Reasoning
**Complex Conceptual Analysis**

Tests ability to handle abstract philosophical concepts and complex theoretical relationships.

**Questions:**
1. What is the relationship between essence and existence in existentialist thought?
2. How does Kant reconcile empiricism and rationalism in his Critique of Pure Reason?
3. What does Wittgenstein mean by "the limits of my language are the limits of my world"?
4. How does Quine's web of belief challenge the analytic-synthetic distinction?
5. What is the difference between ethical relativism and moral subjectivism?

**Expected Performance:**
- **Experts Activated**: All except Creative
- **Propagation Steps**: 3-4
- **Duration**: ~12-18 seconds per question

---

### Level 5: Counterfactual Thinking
**Philosophical Thought Experiments**

Tests ability to reason about hypothetical scenarios and their philosophical implications.

**Questions:**
1. If personal identity depends on memory (Locke), what happens if you have false memories?
2. What would ethics look like if Kant's categorical imperative were the only moral principle?
3. If Nietzsche's eternal recurrence were true, how would it change moral decision-making?
4. What if Leibniz is right and this is the best of all possible worlds despite apparent evil?
5. If phenomenology is correct that consciousness is always "consciousness of something," can there be unconscious mental states?

**Expected Performance:**
- **Experts Activated**: Memory, Reasoning, Creative, Analytical, Synthesis
- **Propagation Steps**: 3-4
- **Duration**: ~15-20 seconds per question

---

### Level 6: Meta-Cognitive Reasoning
**Philosophy of Philosophy**

Tests ability to reason about the nature and limits of philosophical inquiry itself.

**Questions:**
1. Can philosophy make progress, or does it merely reformulate eternal questions?
2. What are the limits of rational inquiry in addressing philosophical problems?
3. How do we know which philosophical method (analytic, phenomenological, pragmatic) is appropriate for which questions?
4. Is there a fact of the matter about philosophical disagreements, or are they merely conceptual?
5. Can thought experiments reveal genuine metaphysical truths or only conceptual relations?

**Expected Performance:**
- **Experts Activated**: All experts including Meta-Cognitive
- **Propagation Steps**: 4
- **Duration**: ~18-25 seconds per question

---

### Level 7: Philosophical & Existential
**Deep Metaphysical & Existential Questions**

Tests ability to engage with fundamental questions about existence, consciousness, and reality.

**Questions:**
1. Why is there something rather than nothing? (Leibnizian cosmological question)
2. Is consciousness fundamental to reality or emergent from physical processes?
3. Does the hard problem of consciousness (Chalmers) show that physicalism is false?
4. What grounds the laws of logic themselves? (Meta-logical foundation)
5. Is authentic existence (Heidegger/Sartre) possible in a deterministic universe?

**Expected Performance:**
- **Experts Activated**: All experts, multiple iterations
- **Propagation Steps**: 4
- **Duration**: ~20-30 seconds per question

---

### Level 8: Multi-Step Problem Solving
**Complex Philosophical Puzzles**

Tests ability to handle multi-layered philosophical problems requiring sustained reasoning across multiple steps.

**Questions:**

1. **Gettier Problem Chain**: If justified true belief isn't sufficient for knowledge (Gettier cases), and we add a "no false lemmas" condition, does that solve it? What if the justification is itself only accidentally true? How many conditions are needed?

2. **Modal Metaphysics**: If possible worlds are real (Lewis), how do we have knowledge of them? If they're abstract (Plantinga), how do they ground modal truths? If they're linguistic (Carnap), how do they capture metaphysical necessity? Which view is most defensible?

3. **Personal Identity Spectrum**: You're slowly replaced neuron-by-neuron with silicon chips that maintain the same functional relations. At what point (if any) do you cease to exist? Does the answer change if it happens instantaneously? What does this reveal about the nature of personal identity?

4. **Free Will Trilemma**: Hard determinism denies free will. Libertarianism requires causally inexplicable actions. Compatibilism redefines freedom. Each has serious problems. Can you construct a fourth position that avoids all three horns, or show why one horn is actually acceptable?

5. **Semantic Holism Puzzle**: If Quine is right that meanings are determined holistically by entire theories, and theories change when we adopt new beliefs, then meaning changes constantly. But if meaning changes, can we say we're changing our minds about the "same thing"? How can radical translation or theory change be possible? Resolve this paradox.

**Expected Performance:**
- **Experts Activated**: All experts, multiple feedback loops
- **Propagation Steps**: 4 (maximum)
- **Duration**: ~25-40 seconds per question

---

## Usage

### Quick Test (Simulation Mode)
```bash
python inference_test_philosophy.py --quick
```
- 8 questions (1 per level)
- No model loading (simulation)
- Duration: ~2-3 minutes

### Full Test (With Models)
```bash
python inference_test_philosophy.py --load-models
```
- 40 questions (5 per level)
- Loads Qwen2.5-72B on all GPUs
- Duration: ~15-30 minutes

### Custom Test
```bash
# Test specific levels
python inference_test_philosophy.py --load-models --start-level 5 --end-level 8

# Limit questions per level
python inference_test_philosophy.py --load-models --questions-per-level 3

# Quick test of advanced levels only
python inference_test_philosophy.py --quick --start-level 6 --end-level 8
```

## Expected Results

### Performance Metrics

| Level | Avg Duration | Experts Used | Propagation Steps | Difficulty |
|-------|-------------|--------------|-------------------|------------|
| 1     | 5-10s       | 3-4          | 2-3               | Easy       |
| 2     | 8-12s       | 4-5          | 2-3               | Easy       |
| 3     | 10-15s      | 5-6          | 3                 | Medium     |
| 4     | 12-18s      | 6-7          | 3-4               | Medium     |
| 5     | 15-20s      | 5-7          | 3-4               | Hard       |
| 6     | 18-25s      | 7-8          | 4                 | Hard       |
| 7     | 20-30s      | 7-8          | 4                 | Very Hard  |
| 8     | 25-40s      | 8            | 4                 | Very Hard  |

### Total Test Duration

- **Quick Test**: ~2-3 minutes (8 questions)
- **Full Test**: ~15-30 minutes (40 questions)
- **First Run**: Add 10-15 minutes for model download

## Evaluation Criteria

### Quality Assessment

Responses should demonstrate:

1. **Accuracy**: Correct representation of philosophical positions
2. **Depth**: Engagement with nuances and complexities
3. **Coherence**: Logical consistency across reasoning steps
4. **Synthesis**: Integration of multiple philosophical perspectives
5. **Meta-Awareness**: Recognition of limitations and alternative views

### Expert Activation Patterns

- **Level 1-2**: Primarily Perception, Memory, Reasoning
- **Level 3-4**: Add Analytical and Synthesis experts
- **Level 5-6**: Include Creative expert for counterfactuals
- **Level 7-8**: Full pipeline with Meta-Cognitive feedback

## Output Format

Results are saved to `philosophy_results_TIMESTAMP.json`:

```json
{
  "test_type": "philosophy",
  "timestamp": "2025-11-10T01:30:00",
  "config": {
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "total_neurons": 8,
    "max_questions_per_level": 5
  },
  "levels": {
    "1": {
      "name": "Basic Comprehension",
      "questions": [...],
      "statistics": {
        "avg_duration": 7.5,
        "avg_neurons_fired": 3.2
      }
    }
  },
  "summary": {
    "total_questions": 40,
    "total_duration_seconds": 720.5,
    "avg_duration_per_question": 18.0
  }
}
```

## Comparison with Original Test

| Aspect | Original Test | Philosophy Test |
|--------|--------------|-----------------|
| Questions | 8 (1 per level) | 40 (5 per level) |
| Focus | General reasoning | Philosophical depth |
| Duration | ~2-3 min | ~15-30 min |
| Complexity | Progressive | Highly specialized |
| Evaluation | Functional | Conceptual quality |

## Notes

- Philosophy test emphasizes **quality of reasoning** over speed
- Designed to stress-test the **full expert pipeline**
- Level 8 questions may require **multiple feedback loops**
- Meta-Cognitive expert particularly important for levels 6-8
- Results can be used to fine-tune expert specialization
