import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Usage: python run_model.py /path/to/model_dir "Your prompt here"
if len(sys.argv) < 3:
    print("Usage: python run_model.py /path/to/model_dir 'Your prompt here'")
    sys.exit(1)

model_dir = sys.argv[1]
prompt = sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
