#!/usr/bin/env python3
# Prepare cybersecurity training data from existing high-quality datasets
# ...

def main():
    """Main function to prepare training data"""
    # ...
    # (rest of the existing code)

if __name__ == "__main__":
    main()

    # Move this file to scripts/
#!/usr/bin/env python3
"""
Prepare cybersecurity training data from existing high-quality datasets
"""

import os
import json
import glob
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_from_disk

def prepare_cybersec_training_data():
    """Prepare high-quality cybersecurity training data from existing datasets"""
    print("🛡️ Preparing cybersecurity training data from existing datasets...")
    
    # Input directories with high-quality cybersecurity data
    input_sources = [
        "/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_datasets/AlicanKiraz0_Cybersecurity-Dataset-Heimdall-v1.1",
        "/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_datasets/AlicanKiraz0_Cybersecurity-Dataset-v1",
        "/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_datasets/Vanessasml_cybersecurity_32k_instruction_input_output"
    ]
    
    # Output file
    output_dir = "/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_datasets/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/cybersec_enhanced_training.jsonl"
    
    total_samples = 0
    training_data = []
    
    for source_dir in input_sources:
        if os.path.exists(source_dir):
            print(f"📂 Processing {source_dir}")
            
            # Check if it's a HuggingFace dataset directory (has dataset_dict.json)
            if os.path.exists(f"{source_dir}/dataset_dict.json"):
                try:
                    print(f"  📊 Loading HuggingFace dataset from {source_dir}")
                    dataset = load_from_disk(source_dir)
                    
                    # Get the train split
                    if 'train' in dataset:
                        train_data = dataset['train']
                        print(f"  📈 Found {len(train_data)} samples in train split")
                        
                        for item in train_data:
                            # Convert to messages format
                            if 'system' in item and 'user' in item and 'assistant' in item:
                                # Heimdall format - convert to messages
                                messages = []
                                if item.get('system'):
                                    messages.append({'role': 'system', 'content': item['system']})
                                messages.append({'role': 'user', 'content': item['user']})
                                messages.append({'role': 'assistant', 'content': item['assistant']})
                                
                                training_data.append({'messages': messages})
                                total_samples += 1
                            
                            elif 'instruction' in item and 'output' in item:
                                # Instruction format - convert to messages
                                messages = [
                                    {'role': 'user', 'content': item['instruction']},
                                    {'role': 'assistant', 'content': item['output']}
                                ]
                                
                                training_data.append({'messages': messages})
                                total_samples += 1
                                
                except Exception as e:
                    print(f"    ❌ Error loading HuggingFace dataset: {e}")
                    continue
            
            else:
                # Look for JSONL files in the directory
                jsonl_files = glob.glob(f"{source_dir}/**/*.jsonl", recursive=True)
                
                for file_path in jsonl_files:
                    print(f"  📄 Reading {file_path}")
                    
                    try:
                        with open(file_path, 'r') as f:
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if not line:
                                    continue
                                    
                                try:
                                    data = json.loads(line)
                                    
                                    # Convert to messages format if needed
                                    if 'messages' in data:
                                        # Already in correct format
                                        if isinstance(data['messages'], list) and len(data['messages']) >= 2:
                                            training_data.append(data)
                                            total_samples += 1
                                    
                                    elif 'system' in data and 'user' in data and 'assistant' in data:
                                        # Heimdall format - convert to messages
                                        messages = []
                                        if data.get('system'):
                                            messages.append({'role': 'system', 'content': data['system']})
                                        messages.append({'role': 'user', 'content': data['user']})
                                        messages.append({'role': 'assistant', 'content': data['assistant']})
                                        
                                        training_data.append({'messages': messages})
                                        total_samples += 1
                                    
                                    elif 'instruction' in data and 'output' in data:
                                        # Instruction format - convert to messages
                                        messages = [
                                            {'role': 'user', 'content': data['instruction']},
                                            {'role': 'assistant', 'content': data['output']}
                                        ]
                                        
                                        training_data.append({'messages': messages})
                                        total_samples += 1
                                        
                                except json.JSONDecodeError as e:
                                    print(f"    ⚠️  JSON error on line {line_num}: {e}")
                                    continue
                                    
                    except Exception as e:
                        print(f"    ❌ Error reading {file_path}: {e}")
                        continue
        else:
            print(f"⚠️  Directory not found: {source_dir}")
    
    # Write combined training data
    if training_data:
        print(f"\\n💾 Writing {total_samples} samples to {output_file}")
        
        with open(output_file, 'w') as f:
            for item in training_data:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\\n')
        
        print(f"✅ Successfully created training file with {total_samples} samples")
        
        # Also create a smaller validation set (10% of data)
        val_size = max(1, total_samples // 10)
        val_file = output_file.replace('training.jsonl', 'validation.jsonl')
        
        with open(val_file, 'w') as f:
            for item in training_data[-val_size:]:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\\n')
        
        print(f"✅ Created validation file with {val_size} samples: {val_file}")
        
        return output_file, total_samples
    else:
        print("❌ No training data found!")
        return None, 0

if __name__ == "__main__":
    output_file, sample_count = prepare_cybersec_training_data()
    if output_file:
        print(f"\\n🎯 Ready for training!")
        print(f"📁 Training file: {output_file}")
        print(f"📊 Sample count: {sample_count}")
    else:
        print("\\n❌ Failed to prepare training data")
