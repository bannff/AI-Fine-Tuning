#!/usr/bin/env python3
"""
Enhanced Mamba Cybersecurity Chat Interface
Optimized for the newly trained M4 cybersecurity model
"""

import argparse

def make_repetition_penalty(penalty: float, context_size: int = 20):
    import mlx.core as mx
    def repetition_penalty_processor(tokens, logits):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            selected_logits = logits[:, tokens]
            selected_logits = mx.where(
                selected_logits < 0,
                selected_logits * penalty,
                selected_logits / penalty,
            )
            logits[:, tokens] = selected_logits
        return logits
    return repetition_penalty_processor

def chat_with_mamba(use_adapter=True):
    from mlx_lm import load, generate
    print("🐍 MAMBA CYBERSECURITY MODEL CHAT")
    print("=================================")
    print("Loading your model...")

    model_path = "models/mamba-1.4b-mlx"
    adapter_path = "adapters"

    try:
        if use_adapter:
            model, tokenizer = load(model_path, adapter_path=adapter_path)
            print("✅ Model loaded with LoRA adapter!")
            print("🧠 Mamba 1.4B with cybersecurity LoRA adapters")
            print("🎯 Specialized for: threat analysis, security patterns, incident response")
        else:
            model, tokenizer = load(model_path)
            print("✅ Model loaded WITHOUT LoRA adapter!")
            print("🧠 Mamba 1.4B base model (no adapter)")
            print("⚠️ Output will be more generic, less cybersecurity-specific")
        print("")
        print("Commands:")
        print("- Type 'q' or 'quit' to exit")
        print("- Type 'r' or 'reset' to clear conversation")
        print("- Type 'h' or 'help' for help")
        print("")

        conversation_history = []
        verbose = True

        while True:
            user_input = input("🔐 User: ").strip()

            if user_input.lower() in ['q', 'quit', 'exit']:
                print("👋 Goodbye! Stay secure!")
                break
            elif user_input.lower() in ['r', 'reset']:
                conversation_history = []
                print("🔄 Conversation reset.")
                continue
            elif user_input.lower() in ['h', 'help']:
                print("\n📚 Try asking about:")
                print("- SQL injection prevention")
                print("- Network security best practices")
                print("- Malware analysis techniques")
                print("- AWS security patterns")
                print("- Incident response procedures")
                print("- Cryptography concepts")
                print("")
                continue
            elif user_input.lower() in ['v', 'verbose']:
                verbose = not verbose
                print(f"🔎 Verbose mode {'enabled' if verbose else 'disabled'}.")
                continue
            elif not user_input:
                continue

            # Format the prompt for cybersecurity context
            if conversation_history:
                prompt = ""
                for turn in conversation_history[-3:]:
                    prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
                prompt += f"User: {user_input}\nAssistant:"
            else:
                prompt = f"User: {user_input}\nAssistant:"
            if verbose:
                print("\n[DEBUG] Prompt sent to model:\n" + prompt + "\n")

            try:
                print("🤖 Assistant: ", end="", flush=True)
                response = generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=1024,
                    logits_processors=[make_repetition_penalty(1.2, 20)]
                )
                if verbose:
                    print("[DEBUG] Raw model response:\n" + str(response) + "\n")
                stop_tokens = ["\nUser:", "</s>", "<|endoftext|>", "User:"]
                if "Assistant:" in response:
                    assistant_response = response.split("Assistant:")[-1].strip()
                else:
                    assistant_response = response.replace(prompt, "").strip()
                for stop_token in stop_tokens:
                    if stop_token in assistant_response:
                        assistant_response = assistant_response.split(stop_token)[0].strip()
                        break
                segments = assistant_response.split("Assistant:")
                cleaned_segments = []
                for seg in segments:
                    seg = seg.strip()
                    if seg and seg not in cleaned_segments:
                        cleaned_segments.append(seg)
                assistant_response = cleaned_segments[0] if cleaned_segments else ""
                if verbose:
                    print("[DEBUG] Extracted assistant response:\n" + assistant_response + "\n")
                print(assistant_response)
                conversation_history.append({
                    'user': user_input,
                    'assistant': assistant_response
                })
                if len(conversation_history) > 5:
                    conversation_history = conversation_history[-5:]
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                print("🔄 Try rephrasing your question or type 'r' to reset.")
            print()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Make sure you're in the PyScience directory and adapters exist.")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mamba Cybersecurity Chat")
    parser.add_argument('--no-adapter', action='store_true', help='Run without LoRA adapter (base model only)')
    args = parser.parse_args()
    chat_with_mamba(use_adapter=not args.no_adapter)
