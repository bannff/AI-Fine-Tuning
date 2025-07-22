#!/usr/bin/env python3
"""
Chat with your LoRA-trained cybersecurity model using MLX-LM
"""

import mlx.core as mx
from mlx_lm import load, generate

def chat_with_model():
    print("🔒 Loading cybersecurity-trained model with LoRA adapters...")
    
    # Load the base model with your trained LoRA adapters
    model, tokenizer = load(
        "/Users/danielrodrigo/Workspace/PyScience/datasets/mlx_models/tinyllama_mlx",  # Base model
        adapter_path="/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_finetuned_models/mlx_adapters_v3"  # Your trained adapters
    )
    
    print("✅ Model loaded successfully!")
    print("💬 Type 'quit' to exit\n")
    
    while True:
        user_input = input("🔐 Ask about cybersecurity: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not user_input:
            continue
            
        # Format as a proper chat prompt
        prompt = f"### User:\n{user_input}\n\n### Assistant:\n"
        
        print("🤖 Generating response...")
        
        try:
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt,
                max_tokens=512,
                temp=0.7,
                top_p=0.9
            )
            
            # Extract just the assistant's response
            assistant_response = response.split("### Assistant:\n")[-1]
            print(f"\n💡 Response:\n{assistant_response}\n")
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            print("Try a different question or restart the script.\n")

if __name__ == "__main__":
    chat_with_model()
