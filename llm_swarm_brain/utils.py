"""
Utility functions for LLM-Swarm-Brain

Includes embedding generation, similarity calculations, and helper functions.
"""

import numpy as np
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation"""

    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}

    def _get_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache"""
        key = self._get_key(text)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache"""
        key = self._get_key(text)

        # Evict least used if cache is full
        if len(self.cache) >= self.max_size:
            least_used_key = min(self.access_count, key=self.access_count.get)
            del self.cache[least_used_key]
            del self.access_count[least_used_key]

        self.cache[key] = embedding
        self.access_count[key] = 0


class EmbeddingManager:
    """Manages embedding generation and similarity calculations"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize embedding manager

        Args:
            model_name: Name of sentence transformer model
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.cache = EmbeddingCache()

        # Generate role embeddings for all neuron roles
        self.role_embeddings = {}

    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for text

        Args:
            text: Input text
            use_cache: Whether to use cache

        Returns:
            Embedding vector
        """
        if use_cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached

        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

        if use_cache:
            self.cache.set(text, embedding)

        return embedding

    def batch_generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32
        )
        return [emb for emb in embeddings]

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        # Embeddings are already normalized
        return float(np.dot(embedding1, embedding2))

    def semantic_relevance(self, text: str, role_description: str) -> float:
        """
        Calculate semantic relevance of text to a role

        Args:
            text: Input text
            role_description: Description of the role

        Returns:
            Relevance score between 0 and 1
        """
        text_embedding = self.generate_embedding(text)
        role_embedding = self.generate_embedding(role_description)
        return self.cosine_similarity(text_embedding, role_embedding)


def calculate_phi(activations: List[float], connections: np.ndarray) -> float:
    """
    Calculate integrated information (Phi) from IIT

    Simplified implementation that measures how much information is generated
    by the system as a whole beyond its parts.

    Args:
        activations: List of neuron activation levels
        connections: Connection matrix between neurons

    Returns:
        Phi value (integrated information)
    """
    if len(activations) < 2:
        return 0.0

    activations = np.array(activations)

    # Calculate whole-system entropy
    p_whole = np.clip(activations, 1e-10, 1.0)
    H_whole = -np.sum(p_whole * np.log2(p_whole) + (1 - p_whole) * np.log2(1 - p_whole))

    # Calculate sum of individual entropies
    H_parts = 0.0
    for p in p_whole:
        H_parts += -(p * np.log2(p + 1e-10) + (1 - p) * np.log2(1 - p + 1e-10))

    # Phi is the difference (information integration)
    phi = max(0.0, H_whole - H_parts)

    # Normalize by connection strength
    if connections.size > 0:
        connectivity = np.mean(connections)
        phi *= connectivity

    return float(phi)


def calculate_global_workspace_salience(
    neuron_outputs: Dict[str, Any],
    activations: Dict[str, float]
) -> List[Tuple[str, float]]:
    """
    Calculate salience scores for global workspace broadcasting (GWT)

    Args:
        neuron_outputs: Dictionary of neuron outputs
        activations: Dictionary of neuron activation levels

    Returns:
        List of (neuron_id, salience_score) tuples sorted by salience
    """
    salience_scores = []

    for neuron_id, activation in activations.items():
        if neuron_id not in neuron_outputs:
            continue

        output = neuron_outputs[neuron_id]

        # Factors contributing to salience:
        # 1. Activation level
        # 2. Output coherence (length and structure)
        # 3. Novelty (can be enhanced with history tracking)

        coherence = min(1.0, len(str(output)) / 500.0)
        salience = activation * 0.6 + coherence * 0.4

        salience_scores.append((neuron_id, salience))

    # Sort by salience (descending)
    salience_scores.sort(key=lambda x: x[1], reverse=True)

    return salience_scores


def sigmoid(x: float) -> float:
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def relu(x: float) -> float:
    """ReLU activation function"""
    return max(0.0, x)


def softmax(values: List[float]) -> List[float]:
    """Softmax function for probability distribution"""
    values = np.array(values)
    exp_values = np.exp(values - np.max(values))  # Numerical stability
    return (exp_values / exp_values.sum()).tolist()


def hebbian_update(
    weight: float,
    pre_activation: float,
    post_activation: float,
    learning_rate: float
) -> float:
    """
    Hebbian learning rule: "Neurons that fire together, wire together"

    Args:
        weight: Current connection weight
        pre_activation: Pre-synaptic neuron activation
        post_activation: Post-synaptic neuron activation
        learning_rate: Learning rate

    Returns:
        Updated weight
    """
    delta_w = learning_rate * pre_activation * post_activation
    new_weight = np.clip(weight + delta_w, 0.0, 1.0)
    return float(new_weight)


def format_neuron_context(
    role: str,
    input_signal: str,
    prior_signals: List[str] = None,
    global_context: Dict[str, Any] = None
) -> str:
    """
    Format context for a neuron's processing

    Args:
        role: Neuron's role description
        input_signal: Current input signal
        prior_signals: Previous signals received
        global_context: Global workspace context

    Returns:
        Formatted prompt string
    """
    context_parts = [f"Your role: {role}\n"]

    if global_context:
        context_parts.append(f"Global context: {json.dumps(global_context, indent=2)}\n")

    if prior_signals:
        context_parts.append("Recent signals:")
        for i, signal in enumerate(prior_signals[-3:], 1):
            context_parts.append(f"  {i}. {signal}")
        context_parts.append("")

    context_parts.append(f"Current input: {input_signal}")

    return "\n".join(context_parts)


class CircularBuffer:
    """Circular buffer for efficient memory management"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Any] = []
        self.index = 0

    def append(self, item: Any):
        """Add item to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.index] = item
            self.index = (self.index + 1) % self.capacity

    def get_all(self) -> List[Any]:
        """Get all items in chronological order"""
        if len(self.buffer) < self.capacity:
            return self.buffer.copy()
        return self.buffer[self.index:] + self.buffer[:self.index]

    def get_recent(self, n: int) -> List[Any]:
        """Get n most recent items"""
        all_items = self.get_all()
        return all_items[-n:] if n < len(all_items) else all_items

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.index = 0


def estimate_vram_usage(model_size_gb: float, num_instances: int, kv_cache_gb: float = 1.0) -> float:
    """
    Estimate VRAM usage for multiple model instances

    Args:
        model_size_gb: Size of single model in GB
        num_instances: Number of instances
        kv_cache_gb: KV cache per instance in GB

    Returns:
        Estimated total VRAM in GB
    """
    per_instance = model_size_gb + kv_cache_gb
    total = per_instance * num_instances
    overhead = total * 0.1  # 10% overhead for PyTorch/operations
    return total + overhead
