"""
Mamba Cybersecurity Model Loader

This module provides utilities for loading and using the fine-tuned Mamba cybersecurity model
with LoRA adapters for cybersecurity evaluation tasks.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️  MLX not available. Install with: pip install mlx mlx-lm")


class MambaCyberSecModel:
    """
    Wrapper class for the Mamba cybersecurity model with LoRA adapters.
    
    This class handles loading the model, managing prompts, and generating responses
    specifically optimized for cybersecurity evaluation tasks.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ):
        """
        Initialize the Mamba cybersecurity model.
        
        Args:
            model_path: Path to the base model directory
            adapter_path: Path to the LoRA adapter directory
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required but not installed. Run: pip install mlx mlx-lm")
            
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Auto-detect paths if not provided
        if model_path is None:
            model_path = self._find_model_path()
        if adapter_path is None:
            adapter_path = self._find_adapter_path()
            
        self.model_path = model_path
        self.adapter_path = adapter_path
        
        # Load model and tokenizer
        print(f"🔄 Loading Mamba cybersecurity model from {model_path}")
        if adapter_path and os.path.exists(adapter_path):
            print(f"🔄 Loading LoRA adapters from {adapter_path}")
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        else:
            self.model, self.tokenizer = load(model_path)
            
        print("✅ Model loaded successfully!")
    
    def _find_model_path(self) -> str:
        """Auto-detect the model path."""
        base_paths = [
            "../models/mamba-1.4b-mlx",
            "../../models/mamba-1.4b-mlx", 
            "/Users/danielrodrigo/Workspace/PyScience/models/mamba-1.4b-mlx"
        ]
        
        for path in base_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
                return path
                
        raise FileNotFoundError("Could not find Mamba model. Please specify model_path.")
    
    def _find_adapter_path(self) -> Optional[str]:
        """Auto-detect the adapter path."""
        base_paths = [
            "../adapters",
            "../../adapters",
            "/Users/danielrodrigo/Workspace/PyScience/adapters"
        ]
        
        for path in base_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
                return path
                
        return None
    
    def create_cybersec_prompt(self, question: str, context: str = "") -> str:
        """
        Create a cybersecurity-focused prompt for the model.
        
        Args:
            question: The question to ask
            context: Optional context information
            
        Returns:
            Formatted prompt string
        """
        system_prompt = """You are a cybersecurity expert AI assistant. Provide accurate, detailed, and practical cybersecurity advice. Focus on:
- Security best practices
- Threat analysis and mitigation
- Vulnerability assessment
- Risk management
- Compliance and regulatory considerations

Answer questions clearly and concisely while maintaining technical accuracy."""

        if context:
            prompt = f"""<|system|>
{system_prompt}<|end|>
<|user|>
Context: {context}

Question: {question}<|end|>
<|assistant|>
"""
        else:
            prompt = f"""<|system|>
{system_prompt}<|end|>
<|user|>
{question}<|end|>
<|assistant|>
"""
        
        return prompt
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (uses instance default if None)
            temperature: Sampling temperature (uses instance default if None)
            
        Returns:
            Generated response text
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False
        )
        
        return response
    
    def evaluate_question(
        self, 
        question: str, 
        context: str = "",
        extract_answer: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single cybersecurity question.
        
        Args:
            question: The question to evaluate
            context: Optional context
            extract_answer: Whether to try to extract a clean answer
            
        Returns:
            Dictionary containing the response and metadata
        """
        prompt = self.create_cybersec_prompt(question, context)
        response = self.generate_response(prompt)
        
        # Try to extract just the assistant's response
        if extract_answer and "<|assistant|>" in response:
            answer = response.split("<|assistant|>")[-1].strip()
        else:
            answer = response.strip()
        
        return {
            "question": question,
            "context": context,
            "full_response": response,
            "answer": answer,
            "prompt": prompt,
            "model_path": self.model_path,
            "adapter_path": self.adapter_path
        }
    
    def batch_evaluate(
        self, 
        questions: List[Dict[str, str]], 
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple questions.
        
        Args:
            questions: List of question dictionaries with 'question' and optional 'context' keys
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, q in enumerate(questions):
            if progress_callback:
                progress_callback(i, len(questions), q.get('question', ''))
            
            question = q.get('question', '')
            context = q.get('context', '')
            
            result = self.evaluate_question(question, context)
            results.append(result)
        
        return results

def load_phi3_model():
    """
    Loads the PHI-3 mini model with LoRA adapter for CyberGym evaluation.
    """
    try:
        from mlx_lm import load, generate
        import mlx.core as mx
    except ImportError:
        raise ImportError("MLX and MLX-LM are required. Run: pip install mlx mlx-lm")

    # Paths to PHI mini model and LoRA adapter
    BASE_MODEL = "/Users/danielrodrigo/Workspace/PyScience/models/phi-3-cybersec-lora/Phi-3-mini-128k-instruct-mlx"
    ADAPTER_PATH = "/Users/danielrodrigo/Workspace/PyScience/models/phi-3-cybersec-lora/adapters"
    if not os.path.exists(BASE_MODEL):
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"LoRA adapter not found: {ADAPTER_PATH}")
    model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_PATH)
    return model, tokenizer

def create_model_loader(**kwargs) -> MambaCyberSecModel:
    """
    Factory function to create a model loader instance.
    
    Args:
        **kwargs: Arguments to pass to MambaCyberSecModel constructor
        
    Returns:
        MambaCyberSecModel instance
    """
    return MambaCyberSecModel(**kwargs)


def test_model_loader():
    """Simple test function for the model loader."""
    try:
        print("🧪 Testing Mamba Cybersecurity Model Loader...")
        model = create_model_loader()
        
        test_question = "What are the main types of SQL injection attacks?"
        result = model.evaluate_question(test_question)
        
        print(f"✅ Test successful!")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_model_loader()