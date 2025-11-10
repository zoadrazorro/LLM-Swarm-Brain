"""
RAG (Retrieval-Augmented Generation) Memory System

Provides semantic search over training history to retrieve relevant
past experiences and inject them as context for improved reasoning.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from .utils import EmbeddingManager


class RAGMemory:
    """RAG-based memory system with semantic retrieval"""
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        memory_file: str = "training_checkpoints/training_memory.pkl",
        top_k: int = 3
    ):
        """
        Initialize RAG memory
        
        Args:
            embedding_manager: Manager for generating embeddings
            memory_file: Path to persistent memory file
            top_k: Number of top similar experiences to retrieve
        """
        self.embedding_manager = embedding_manager
        self.memory_file = Path(memory_file)
        self.top_k = top_k
        
        # Memory storage
        self.experiences = []  # List of {question, answer, embedding, metrics}
        self.experience_embeddings = []  # Precomputed embeddings
        
        # Load from persistent memory if exists
        self._load_from_persistent_memory()
    
    def _load_from_persistent_memory(self):
        """Load training history from persistent memory"""
        if not self.memory_file.exists():
            return
        
        try:
            with open(self.memory_file, 'rb') as f:
                persistent_memory = pickle.load(f)
            
            # Extract experiences from session history
            for session in persistent_memory.get("session_history", []):
                # Each session has level info, but we need individual Q&A
                # For now, we'll use level-level aggregates
                # TODO: Store individual Q&A in future training runs
                pass
            
            # Check if we have detailed training results
            checkpoint_dir = self.memory_file.parent
            result_files = list(checkpoint_dir.glob("training_results_*.json"))
            
            if result_files:
                # Load most recent training results
                latest_results = sorted(result_files)[-1]
                with open(latest_results, 'r') as f:
                    results = json.load(f)
                
                # Extract Q&A from level performance
                for level in results.get("level_performance", []):
                    for q_data in level.get("questions", []):
                        self.add_experience(
                            question=q_data.get("question", ""),
                            answer="",  # Answer not stored in current format
                            metrics={
                                "consciousness": q_data.get("consciousness", 0.0),
                                "integration": q_data.get("integration", 0.0),
                                "coherence": q_data.get("coherence", 0.0)
                            }
                        )
        
        except Exception as e:
            print(f"Warning: Could not load RAG memory: {e}")
    
    def add_experience(
        self,
        question: str,
        answer: str,
        metrics: Optional[Dict] = None
    ):
        """
        Add a new experience to memory
        
        Args:
            question: The question asked
            answer: The answer generated
            metrics: Performance metrics (consciousness, integration, etc.)
        """
        # Generate embedding for question
        embedding = self.embedding_manager.generate_embedding(question)
        
        experience = {
            "question": question,
            "answer": answer,
            "metrics": metrics or {},
            "embedding": embedding
        }
        
        self.experiences.append(experience)
        self.experience_embeddings.append(embedding)
    
    def retrieve_similar(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most similar past experiences
        
        Args:
            query: Current question/query
            top_k: Number of results to return (default: self.top_k)
        
        Returns:
            List of similar experiences with similarity scores
        """
        if not self.experiences:
            return []
        
        k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, exp_embedding in enumerate(self.experience_embeddings):
            similarity = self._cosine_similarity(query_embedding, exp_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k
        top_indices = [idx for idx, _ in similarities[:k]]
        
        # Return experiences with scores
        results = []
        for idx in top_indices:
            exp = self.experiences[idx].copy()
            exp["similarity_score"] = similarities[idx][1]
            results.append(exp)
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def format_context(
        self,
        similar_experiences: List[Dict[str, Any]],
        max_context_length: int = 500
    ) -> str:
        """
        Format retrieved experiences as context string
        
        Args:
            similar_experiences: List of similar experiences
            max_context_length: Maximum length of context
        
        Returns:
            Formatted context string
        """
        if not similar_experiences:
            return ""
        
        context_parts = ["### Relevant Past Experiences:\n"]
        
        for i, exp in enumerate(similar_experiences, 1):
            question = exp["question"]
            similarity = exp.get("similarity_score", 0.0)
            metrics = exp.get("metrics", {})
            
            # Format experience
            exp_text = f"\n{i}. Similar Question (similarity: {similarity:.2f}):\n"
            exp_text += f"   Q: {question[:200]}...\n" if len(question) > 200 else f"   Q: {question}\n"
            
            if metrics:
                exp_text += f"   Performance: consciousness={metrics.get('consciousness', 0):.2f}, "
                exp_text += f"integration={metrics.get('integration', 0):.2f}\n"
            
            # Check length
            if len("".join(context_parts)) + len(exp_text) > max_context_length:
                break
            
            context_parts.append(exp_text)
        
        return "".join(context_parts)
    
    def get_augmented_prompt(
        self,
        question: str,
        base_prompt: str,
        top_k: Optional[int] = None
    ) -> str:
        """
        Get prompt augmented with retrieved context
        
        Args:
            question: Current question
            base_prompt: Base prompt/instruction
            top_k: Number of similar experiences to retrieve
        
        Returns:
            Augmented prompt with context
        """
        # Retrieve similar experiences
        similar = self.retrieve_similar(question, top_k)
        
        if not similar:
            return base_prompt
        
        # Format context
        context = self.format_context(similar)
        
        # Augment prompt
        augmented = f"{context}\n\n### Current Question:\n{base_prompt}"
        
        return augmented
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if not self.experiences:
            return {
                "total_experiences": 0,
                "avg_consciousness": 0.0,
                "avg_integration": 0.0
            }
        
        total = len(self.experiences)
        avg_consciousness = np.mean([
            exp["metrics"].get("consciousness", 0.0)
            for exp in self.experiences
        ])
        avg_integration = np.mean([
            exp["metrics"].get("integration", 0.0)
            for exp in self.experiences
        ])
        
        return {
            "total_experiences": total,
            "avg_consciousness": float(avg_consciousness),
            "avg_integration": float(avg_integration)
        }


def create_rag_memory(
    embedding_manager: EmbeddingManager,
    memory_file: str = "training_checkpoints/training_memory.pkl",
    top_k: int = 3
) -> RAGMemory:
    """
    Factory function to create RAG memory
    
    Args:
        embedding_manager: Embedding manager instance
        memory_file: Path to persistent memory
        top_k: Number of similar experiences to retrieve
    
    Returns:
        RAGMemory instance
    """
    return RAGMemory(embedding_manager, memory_file, top_k)
