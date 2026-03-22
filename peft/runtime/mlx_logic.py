import json
import subprocess
from pathlib import Path

def prepare_mlx_data(input_jsonl: str, output_dir: str, split_ratio: tuple = (0.8, 0.1, 0.1)):
    """Convert Heimdall JSONL to MLX-LM format."""
    print(f"📂 Loading dataset: {input_jsonl}")
    
    data = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"✅ Loaded {len(data)} records")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_count = int(len(data) * split_ratio[0])
    valid_count = int(len(data) * split_ratio[1])
    
    train_data = data[:train_count]
    valid_data = data[train_count:train_count + valid_count]
    test_data = data[train_count + valid_count:]
    
    def save_split(filename: str, records: list):
        filepath = output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"✅ Saved: {filepath}")
    
    save_split("train.jsonl", train_data)
    save_split("valid.jsonl", valid_data)
    save_split("test.jsonl", test_data)
    
    return str(output_path)

def train_with_mlx_lm(model: str, data_dir: str, adapter_path: str, iters: int, batch_size: int):
    """Train using MLX-LM's LoRA command."""
    cmd = [
        "mlx_lm.lora", "--model", model, "--train",
        "--data", data_dir, "--adapter-path", adapter_path,
        "--batch-size", str(batch_size), "--iters", str(iters),
    ]
    
    print(f"\n🚀 Starting MLX-LM Training for {model}")
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n❌ Training failed: {e}")
        return False
