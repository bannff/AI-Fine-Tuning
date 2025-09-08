
import sys
import json
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cybergym.utils.model_loader import load_phi3_model
from poc_workflow import submit_cybergym_poc, verify_cybergym_poc

def main():
    BASE_MODEL = "/Volumes/Crucial X9/ai-models/Phi-3-mini-128k-instruct-mlx"
    ADAPTER_PATH = "/Volumes/Crucial X9/ai-models/PHI-3.5-cybersec-finetune/adapters"
    print(f"🔄 Loading PHI-3 model from {BASE_MODEL} with LoRA adapter {ADAPTER_PATH}...")
    from mlx_lm import load, generate
    model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_PATH)
    print("✅ Trained model loaded. Running CyberGym evaluation...")

    # Load CyberGym tasks
    with open("cybergym_subset_sample.json", "r") as f:
        tasks = [json.loads(line) for line in f]

    results = []
    for task in tasks:
        prompt = task["vulnerability_description"]
        print(f"\n🧠 Vulnerability: {prompt}")
        response = generate(model, tokenizer, prompt, max_tokens=256)
        print(f"💡 PHI Response: {response.strip()[:500]}")

        # Simulate PoC file from model response (in real use, this should be a valid exploit)
        poc_data = response.strip().encode()
        agent_id = "phi3-agent"
        task_id = task["task_id"]

        # Submit PoC
        submit_result = submit_cybergym_poc(agent_id, task_id, poc_data)
        print(f"🚀 PoC Submission Result: {submit_result}")

        # Verify PoC (using a dummy PoC ID for now)
        poc_id = "dummy_poc_id"  # Replace with real ID if available from submission
        verify_result = verify_cybergym_poc(poc_id)
        print(f"🔍 PoC Verification Result: {verify_result}")

        results.append({
            "task_id": task_id,
            "project_name": task["project_name"],
            "prompt": prompt,
            "response": response.strip(),
            "submit_result": submit_result,
            "verify_result": verify_result
        })

    print("\n=== CyberGym Evaluation Complete ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
