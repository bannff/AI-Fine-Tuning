"""
Multi-turn agentic chat test for TinyLlama fine-tuned on Heimdall v1.1.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = "./tinyllama_heimdall_finetuned"
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer from base model
print("Loading tokenizer from base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Load fine-tuned model
print("Loading fine-tuned model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
model = model.to(device)

def build_prompt(system, history):
    prompt = f"System: {system}\n"
    for turn in history:
        prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    prompt += f"User: {history[-1]['user']}\nAssistant: "
    return prompt

system_prompt = "You are a highly specialized AI assistant for advanced cyber-defense whose mission is to deliver accurate, in-depth, actionable guidance on information-security principles."

chat_history = []
for turn in range(3):
    user_input = input(f"User ({turn+1}): ")
    chat_history.append({"user": user_input, "assistant": ""})
    prompt = build_prompt(system_prompt, chat_history)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"[DEBUG] Full model output:\n{response}\n")
    # Robust extraction: get everything after the last 'Assistant:' or after the prompt
    if 'Assistant:' in response:
        agent_response = response.split('Assistant:')[-1].strip()
    else:
        agent_response = response[len(prompt):].strip()
    if not agent_response:
        print("[Warning] Agent did not generate a response. Output was:", response)
        agent_response = "[No response generated]"
    print(f"Agent: {agent_response}")
    chat_history[-1]["assistant"] = agent_response

print("Multi-turn agentic chat test complete.")
