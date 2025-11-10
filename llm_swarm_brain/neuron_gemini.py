"""
Gemini API-based Neuron: Individual LLM-based neuron using Google Gemini API

Implements neural behavior using Gemini 2.5 Pro via Google AI API.
"""

import google.generativeai as genai
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

from llm_swarm_brain.config import NeuronRole
from llm_swarm_brain.utils import (
    format_neuron_context,
    CircularBuffer,
    sigmoid,
    hebbian_update
)

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class NeuronConnection:
    """Represents a connection to another neuron"""
    target_neuron: 'GeminiNeuron'
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


class GeminiNeuron:
    """
    Individual neuron in the LLM swarm brain using Google Gemini API
    
    Implements:
    - Activation calculation based on input relevance
    - Signal processing via Gemini 2.5 Pro API
    - Propagation to connected neurons
    - Hebbian learning for connection weights
    """

    def __init__(
        self,
        role: NeuronRole,
        gpu_id: int,  # Kept for compatibility, not used
        neuron_id: str,
        activation_threshold: float = 0.6,
        model_name: str = "gemini-2.0-flash-exp",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ):
        """
        Initialize Gemini-based neuron
        
        Args:
            role: Neuron's functional role
            gpu_id: Kept for compatibility (not used with API)
            neuron_id: Unique identifier for this neuron
            activation_threshold: Threshold for neuron activation
            model_name: Gemini model name
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            api_key: Google AI API key (or set GOOGLE_API_KEY env var)
        """
        self.role = role
        self.gpu_id = gpu_id
        self.neuron_id = neuron_id
        self.activation_threshold = activation_threshold
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None
            logger.warning(f"{neuron_id}: No API key provided. Will use simulation mode.")
        
        # Neural properties
        self.connections: List[NeuronConnection] = []
        self.activation_level: float = 0.0
        self.is_firing: bool = False
        
        # Memory and history
        self.signal_history = CircularBuffer(capacity=50)
        self.activation_history = CircularBuffer(capacity=100)
        self.output_history: List[str] = []
        
        # Statistics
        self.total_firings: int = 0
        self.total_signals_received: int = 0
        self.total_signals_sent: int = 0
        self.total_api_calls: int = 0
        self.total_api_errors: int = 0
        
        # Role prompt (can be overridden by brain)
        self._role_prompt = f"You are a {self.role.value} expert."
        
        logger.info(f"Initialized {neuron_id} (Gemini-based, role={role.value})")

    @property
    def device(self) -> str:
        """Return device string for compatibility"""
        return "gemini-api"

    def calculate_activation(
        self,
        input_signal: NeuronSignal,
        global_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate activation level based on input signal
        
        Args:
            input_signal: Input signal to process
            global_context: Global workspace context
            
        Returns:
            Activation level (0-1)
        """
        # Base activation from signal strength
        base_activation = input_signal.activation_level
        
        # Modulate by role relevance (simplified)
        role_boost = 0.1 if input_signal.source_role != self.role else 0.2
        
        # Global context boost
        context_boost = 0.0
        if global_context and "active_concepts" in global_context:
            context_boost = 0.1
        
        # Calculate final activation
        activation = sigmoid(base_activation + role_boost + context_boost)
        
        # Store in history
        self.activation_level = activation
        self.activation_history.append(activation)
        
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
        Generate output using Gemini API
        
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
                role=self._role_prompt,
                input_signal=input_signal.content,
                prior_signals=prior_signals,
                global_context=global_context
            )
            
            # Generate response using API or simulation fallback
            if self.model:
                output = self._generate_gemini(context)
            else:
                output = self._simulate_output(input_signal, global_context)
            
            if output is None:
                output = self._simulate_output(input_signal, global_context)
            
            # Store in history
            self.output_history.append(output)
            
            logger.debug(f"{self.neuron_id} fired with activation {self.activation_level:.3f}")
            
            return output
            
        except Exception as e:
            logger.error(f"Error during firing: {e}")
            self.total_api_errors += 1
            return self._simulate_output(input_signal, global_context)
            
        finally:
            self.is_firing = False

    def _generate_gemini(self, prompt: str) -> Optional[str]:
        """
        Generate text using Gemini API
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text or None on error
        """
        try:
            self.total_api_calls += 1
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
            )
            
            # Format the full prompt with role
            full_prompt = f"{self._role_prompt}\n\n{prompt}"
            
            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            # Extract text from response
            if response.text:
                return response.text.strip()
            else:
                logger.error(f"{self.neuron_id}: Empty response from Gemini API")
                return None
                
        except Exception as e:
            logger.error(f"{self.neuron_id}: Gemini API error: {e}")
            self.total_api_errors += 1
            return None

    def _simulate_output(
        self,
        input_signal: NeuronSignal,
        global_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Return a lightweight simulated response when API is not available."""
        context_bits = []
        if global_context:
            if memory := global_context.get("memory"):
                short_term = memory.get("short_term") or []
                if short_term:
                    context_bits.append(f"Recent memory: {short_term[-1]}")
        
        base_response = (
            f"[{self.role.value}] processed input '{input_signal.content[:60]}'. "
            "(simulation mode - no API key)"
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

    def connect_to(self, target_neuron: 'GeminiNeuron', weight: float):
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
            "device": "gemini-api",
            "activation_level": self.activation_level,
            "total_firings": self.total_firings,
            "total_signals_received": self.total_signals_received,
            "total_signals_sent": self.total_signals_sent,
            "total_api_calls": self.total_api_calls,
            "total_api_errors": self.total_api_errors,
            "api_success_rate": (self.total_api_calls - self.total_api_errors) / self.total_api_calls if self.total_api_calls > 0 else 0.0,
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
        return f"GeminiNeuron(id={self.neuron_id}, role={self.role.value}, activation={self.activation_level:.3f})"
