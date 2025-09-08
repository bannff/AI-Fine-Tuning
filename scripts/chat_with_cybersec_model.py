#!/usr/bin/env python3
"""
Chat with your LoRA-trained cybersecurity model using MLX-LM
"""

import mlx.core as mx
from mlx_lm import load, generate

def chat_with_model():
    print("🔒 Loading cybersecurity-trained Mamba-1.4b-mlx with latest adapters...")
    print("📊 Model: mamba-1.4b-mlx | Adapter: adapters/adapters.safetensors")
    print("=" * 70)

    # Load the Mamba-1.4b-mlx base model with your latest trained adapters
    model, tokenizer = load(
        "models/mamba-1.4b-mlx",  # Base model
        adapter_path="adapters"  # Directory containing adapter_config.json and weights
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

        # Use a role-based JSON prompt matching the training data
        prompt = (
            '{"messages": ['
            '{"role": "user", "content": "' + user_input.replace('"', '\"') + '"}, '
            '{"role": "assistant", "content": "'
        )

        print("🤖 Generating response...")

        try:
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=200,  # Allow a bit more room for technical answers
                verbose=False
            )

            # Extract just the assistant's response (after the assistant content opening)
            assistant_response = response.strip().split('"content": "', 1)[-1]
            # Remove any trailing quote or JSON artifacts
            assistant_response = assistant_response.split('"')[0].replace('"}', '').strip()

            # Stop at first sign of a new user/assistant turn or double newline
            for stop_token in ["{\"role\": \"user\"", "{\"role\": \"assistant\"", "\n\n"]:
                if stop_token in assistant_response:
                    assistant_response = assistant_response.split(stop_token)[0].strip()

            # Clean up repeated "Assistant:" or "User:" lines
            lines = assistant_response.split('\n')
            clean_lines = []
            seen_lines = set()
            for line in lines:
                line = line.strip()
                if not line or line in seen_lines:
                    continue
                if line.startswith('Assistant:') or line.startswith('User:'):
                    break
                clean_lines.append(line)
                seen_lines.add(line)
            final_response = '\n'.join(clean_lines)
            print(f"\n💡 Response:\n{final_response}\n")

        except Exception as e:
            print(f"❌ Error generating response: {e}")
            print("Try a different question or restart the script.\n")

if __name__ == "__main__":
    chat_with_model()
