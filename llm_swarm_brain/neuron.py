"""
Phi3Neuron: Individual LLM-based neuron unit

Implements neural behavior including activation, firing, and signal propagation.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

from llm_swarm_brain.config import NeuronRole, ROLE_PROMPTS
from llm_swarm_brain.utils import (
    EmbeddingManager,
    format_neuron_context,
    CircularBuffer,
    sigmoid,
    hebbian_update
)


logger = logging.getLogger(__name__)


@dataclass
class NeuronConnection:
    """Represents a connection to another neuron"""
    target_neuron: 'Phi3Neuron'
    weight: float
    last_updated: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash(id(self.target_neuron))


@dataclass
class NeuronSignal:
    """Signal passed between neurons"""
    content: str
    source_role: NeuronRole
    activation_level: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Phi3Neuron:
    """
    Individual neuron in the LLM swarm brain

    Implements:
    - Activation calculation based on input relevance
    - Signal processing via Phi-3 model
    - Propagation to connected neurons
    - Hebbian learning for connection weights
    """

    def __init__(
        self,
        role: NeuronRole,
        gpu_id: int,
        neuron_id: str,
        activation_threshold: float = 0.6,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        max_tokens: int = 512,
        temperature: float = 0.7,
        load_model: bool = True
    ):
        """
        Initialize Phi3 neuron

        Args:
            role: Neuron's functional role
            gpu_id: GPU device ID
            neuron_id: Unique identifier for this neuron
            activation_threshold: Threshold for neuron activation
            model_name: Hugging Face model name
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            load_model: Whether to load model immediately (False for testing)
        """
        self.role = role
        self.gpu_id = gpu_id
        self.neuron_id = neuron_id
        self.activation_threshold = activation_threshold
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.load_model = load_model

        # Neural properties
        self.connections: List[NeuronConnection] = []
        self.activation_level: float = 0.0
        self.is_firing: bool = False

        # Memory and history
        self.signal_history = CircularBuffer(capacity=50)
        self.output_history = CircularBuffer(capacity=50)
        self.activation_history = CircularBuffer(capacity=100)

        # Statistics
        self.total_firings: int = 0
        self.total_signals_received: int = 0
        self.total_signals_sent: int = 0

        # Model and tokenizer (lazy loaded)
        self.model = None
        self.tokenizer = None
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        # Embedding manager for activation calculation
        self.embedding_manager = EmbeddingManager(device="cpu")  # Use CPU for embeddings
        self.role_embedding = self.embedding_manager.generate_embedding(
            ROLE_PROMPTS[self.role]
        )

        if load_model:
            self._load_model()

        logger.info(f"Initialized {self.neuron_id} with role {self.role.value} on {self.device}")

    def _load_model(self):
        """Load Phi-3 model with 4-bit quantization"""
        try:
            logger.info(f"Loading model {self.model_name} on {self.device}...")

            # 4-bit quantization configuration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map={"": self.gpu_id},
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def calculate_activation(
        self,
        input_signal: NeuronSignal,
        global_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate activation level based on input relevance

        Uses semantic similarity between input and neuron's role.

        Args:
            input_signal: Input signal to process
            global_context: Global workspace context

        Returns:
            Activation level (0.0 to 1.0)
        """
        # Generate embedding for input
        input_embedding = self.embedding_manager.generate_embedding(input_signal.content)

        # Calculate relevance to this neuron's role
        relevance = self.embedding_manager.cosine_similarity(
            input_embedding,
            self.role_embedding
        )

        # Modulate by input signal's activation level
        activation = relevance * input_signal.activation_level

        # Additional boost if there's relevant global context
        if global_context and "focus_areas" in global_context:
            if self.role.value in global_context["focus_areas"]:
                activation *= 1.2

        # Apply sigmoid for smooth activation
        activation = sigmoid(activation)

        self.activation_level = float(np.clip(activation, 0.0, 1.0))
        self.activation_history.append(self.activation_level)

        return self.activation_level

    def should_fire(self) -> bool:
        """
        Determine if neuron should fire based on activation level

        Returns:
            True if activation exceeds threshold
        """
        return self.activation_level >= self.activation_threshold

    def fire(
        self,
        input_signal: NeuronSignal,
        global_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate output using Phi-3 model

        Args:
            input_signal: Input signal to process
            global_context: Global workspace context

        Returns:
            Generated output string, or None if not firing
        """
        if not self.should_fire():
            return None

        self.is_firing = True
        self.total_firings += 1

        try:
            # Format context for this neuron
            prior_signals = [
                sig.content for sig in self.signal_history.get_recent(3)
            ]

            context = format_neuron_context(
                role=ROLE_PROMPTS[self.role],
                input_signal=input_signal.content,
                prior_signals=prior_signals,
                global_context=global_context
            )

            # Generate response using Phi-3 or simulation fallback
            if self.model is None or self.tokenizer is None:
                if self.load_model:
                    self._load_model()
                else:
                    output = self._simulate_output(input_signal, global_context)
            else:
                output = self._generate(context)

            if output is None:
                output = self._simulate_output(input_signal, global_context)

            # Store in history
            self.output_history.append(output)

            logger.debug(f"{self.neuron_id} fired with activation {self.activation_level:.3f}")

            return output

        except Exception as e:
            logger.error(f"Error during firing: {e}")
            return None

        finally:
            self.is_firing = False

    def _generate(self, prompt: str) -> str:
        """
        Generate text using Phi-3 model

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Format prompt for Phi-3
        messages = [
            {"role": "system", "content": ROLE_PROMPTS[self.role]},
            {"role": "user", "content": prompt}
        ]

        # Tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def _simulate_output(
        self,
        input_signal: NeuronSignal,
        global_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Return a lightweight simulated response when models are not available."""
        context_bits = []
        if global_context:
            if memory := global_context.get("memory"):
                short_term = memory.get("short_term") or []
                if short_term:
                    context_bits.append(f"Recent memory: {short_term[-1]}")
        base_response = (
            f"[{self.role.value}] processed input '{input_signal.content[:60]}'. "
            "(simulation mode)"
        )
        if context_bits:
            base_response += " Context -> " + " | ".join(context_bits)
        return base_response

    def receive_signal(
        self,
        signal: NeuronSignal,
        source_weight: float = 1.0
    ):
        """
        Receive signal from another neuron

        Args:
            signal: Input signal
            source_weight: Weight of source connection
        """
        self.total_signals_received += 1

        # Modulate signal by connection weight
        weighted_signal = NeuronSignal(
            content=signal.content,
            source_role=signal.source_role,
            activation_level=signal.activation_level * source_weight,
            metadata={**signal.metadata, "source_weight": source_weight}
        )

        self.signal_history.append(weighted_signal)

    def propagate(
        self,
        output: str,
        learning_rate: float = 0.01
    ) -> int:
        """
        Propagate output to connected neurons

        Implements Hebbian learning to update connection weights.

        Args:
            output: Output to propagate
            learning_rate: Hebbian learning rate

        Returns:
            Number of neurons signal was propagated to
        """
        if not output:
            return 0

        propagation_count = 0

        # Create outgoing signal
        signal = NeuronSignal(
            content=output,
            source_role=self.role,
            activation_level=self.activation_level,
            metadata={"neuron_id": self.neuron_id}
        )

        for connection in self.connections:
            # Only propagate if connection is strong enough
            if connection.weight >= 0.3:  # Minimum propagation threshold
                # Send signal
                connection.target_neuron.receive_signal(signal, connection.weight)
                self.total_signals_sent += 1
                propagation_count += 1

                # Hebbian learning: strengthen connection if both neurons are active
                target_activation = connection.target_neuron.activation_level
                new_weight = hebbian_update(
                    connection.weight,
                    self.activation_level,
                    target_activation,
                    learning_rate
                )

                connection.weight = new_weight
                connection.last_updated = datetime.now()

        return propagation_count

    def connect_to(self, target_neuron: 'Phi3Neuron', weight: float):
        """
        Create connection to another neuron

        Args:
            target_neuron: Target neuron
            weight: Initial connection weight
        """
        connection = NeuronConnection(
            target_neuron=target_neuron,
            weight=np.clip(weight, 0.0, 1.0)
        )

        # Avoid duplicate connections
        if connection not in self.connections:
            self.connections.append(connection)
            logger.debug(f"Connected {self.neuron_id} → {target_neuron.neuron_id} (weight={weight:.2f})")

    def decay_connections(self, decay_rate: float = 0.001):
        """
        Apply decay to connection weights (synaptic pruning)

        Args:
            decay_rate: Rate of weight decay
        """
        for connection in self.connections:
            connection.weight = max(0.0, connection.weight - decay_rate)

    def get_stats(self) -> Dict[str, Any]:
        """Get neuron statistics"""
        return {
            "neuron_id": self.neuron_id,
            "role": self.role.value,
            "gpu_id": self.gpu_id,
            "activation_level": self.activation_level,
            "total_firings": self.total_firings,
            "total_signals_received": self.total_signals_received,
            "total_signals_sent": self.total_signals_sent,
            "num_connections": len(self.connections),
            "avg_connection_weight": np.mean([c.weight for c in self.connections]) if self.connections else 0.0,
            "recent_activations": self.activation_history.get_recent(10)
        }

    def reset(self):
        """Reset neuron state"""
        self.activation_level = 0.0
        self.is_firing = False
        self.signal_history.clear()

    def __repr__(self) -> str:
        return f"Phi3Neuron(id={self.neuron_id}, role={self.role.value}, activation={self.activation_level:.3f})"
