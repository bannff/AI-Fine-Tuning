#!/usr/bin/env python3
"""
PHI-3 Cybersecurity LoRA Chat Interface
Interactive chat with fine-tuned cybersecurity model using MLX-LM best practices
"""
import os
import sys
from pathlib import Path

def main():
    # Use trained adapters location 
    ADAPTER_PATH = "/Volumes/Crucial X9/ai-models/PHI-3.5-cybersec-finetune/adapters"
    BASE_MODEL = "/Volumes/Crucial X9/ai-models/Phi-3-mini-128k-instruct-mlx"
    
    if not Path(ADAPTER_PATH).exists():
        print("❌ No trained adapters found at:")
        print(f"   {ADAPTER_PATH}")
        print("Run training first: python3 train.py")
        return 1
    
    print("🤖 PHI-3 Cybersecurity Assistant (Enhanced with MLX-LM Best Practices)")
    print("=" * 70)
    print("Loading model with LoRA adapters...")
    
    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler, make_repetition_penalty, apply_min_p, apply_top_p
        import mlx.core as mx
        
        # Load model with adapters
        model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_PATH)
        print("✅ Model loaded successfully!")
        print("\nEnhanced Settings:")
        print("• Temperature: 0.7 (balanced creativity/coherence)")
        print("• Min-P: 0.05 (adaptive token filtering)")
        print("• Top-P: 0.9 (nucleus sampling backup)")
        print("• Repetition penalty: 1.05 (light anti-repetition)")
        print("• Max tokens: 512 (comprehensive cybersec responses)")
        print("\nCommands: 'quit' to exit, 'clear' to reset, 'settings' to show parameters")
        print("-" * 70)
        
        # Create enhanced sampler with best practices
        def enhanced_sampler(logits):
            """Enhanced sampler with Min-P, Top-P fallback, and temperature"""
            # Apply Min-P filtering first (adaptive)
            logits = apply_min_p(logits, min_p=0.05, min_tokens_to_keep=1)
            
            # Apply Top-P as fallback (in case Min-P is too restrictive)
            logits = apply_top_p(logits, top_p=0.9)
            
            # Apply temperature scaling
            temperature = 0.7
            logits = logits / temperature
            
            # Sample from the filtered/scaled distribution
            probs = mx.softmax(logits, axis=-1)
            return mx.random.categorical(mx.log(probs))
        
        # Create repetition penalty processor (correct parameter name)
        repetition_penalty = make_repetition_penalty(penalty=1.05, context_size=64)
        
        conversation_history = ""
        
        while True:
            user_input = input("\n🔒 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Stay secure! 👨‍💻 Goodbye! 👋")
                break
            elif user_input.lower() == 'clear':
                conversation_history = ""
                print("🔄 Conversation cleared - Fresh start for new cybersecurity topic")
                continue
            elif user_input.lower() == 'settings':
                print("\n📊 Current Generation Settings:")
                print("   Temperature: 0.7 (balance creativity & accuracy)")
                print("   Min-P: 0.05 (exclude unlikely tokens adaptively)")
                print("   Top-P: 0.9 (nucleus sampling fallback)")
                print("   Repetition Penalty: 1.05 (light anti-repetition)")
                print("   Max Tokens: 512 (comprehensive cybersec responses)")
                continue
            elif not user_input:
                continue
            
            # Build conversation-aware prompt
            if conversation_history:
                prompt = f"{conversation_history}\nUser: {user_input}\n\nAssistant:"
            else:
                # First message - set cybersecurity context
                prompt = f"You are a cybersecurity expert assistant. Provide accurate, detailed, and actionable cybersecurity advice.\n\nUser: {user_input}\n\nAssistant:"
            
            print("\n🤖 Assistant: ", end="", flush=True)
            
            try:
                # Enhanced generation with MLX-LM best practices
                response = generate(
                    model, 
                    tokenizer, 
                    prompt,
                    max_tokens=512,  # Longer for detailed cybersecurity responses
                    verbose=False,
                    sampler=enhanced_sampler,  # Our custom enhanced sampler
                    logits_processors=[repetition_penalty]  # Anti-repetition
                )
                
                # Clean response - extract just assistant part
                if "Assistant:" in response:
                    assistant_response = response.split("Assistant:")[-1].strip()
                else:
                    assistant_response = response.strip()
                
                # Remove any accidental user prompts that might have been generated
                if "\nUser:" in assistant_response:
                    assistant_response = assistant_response.split("\nUser:")[0].strip()
                if "User:" in assistant_response and assistant_response.startswith("User:"):
                    assistant_response = assistant_response.split("User:", 1)[1].strip()
                
                print(assistant_response)
                
                # Update conversation history with smart length management
                conversation_history += f"User: {user_input}\n\nAssistant: {assistant_response}\n\n"
                
                # Smart conversation history management (keep last ~2000 chars)
                if len(conversation_history) > 3000:
                    # Find a good breaking point (end of a complete exchange)
                    lines = conversation_history.split('\n')
                    total_chars = 0
                    keep_lines = []
                    
                    # Keep from the end, working backwards
                    for line in reversed(lines):
                        if total_chars + len(line) < 2000:
                            keep_lines.insert(0, line)
                            total_chars += len(line) + 1  # +1 for newline
                        else:
                            break
                    
                    conversation_history = '\n'.join(keep_lines)
            
            except KeyboardInterrupt:
                print("\n⚡ Generation interrupted by user")
                continue
            except Exception as gen_error:
                print(f"\n❌ Generation error: {gen_error}")
                print("Continuing with next message...")
                continue
    
    except ImportError:
        print("❌ MLX-LM not installed. Install with:")
        print("   pip install mlx-lm")
        return 1
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
