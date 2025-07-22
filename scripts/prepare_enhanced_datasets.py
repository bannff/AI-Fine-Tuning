#!/usr/bin/env python3
# Prepare enhanced cybersecurity datasets for training.
# Converts all downloaded datasets to unified JSONL format with \n fixes.
# ...

def main():
    """Main function to prepare datasets"""
    # ...
    # (rest of the existing code)

if __name__ == "__main__":
    main()

    # Move this file to scripts/
#!/usr/bin/env python3
"""
Prepare enhanced cybersecurity datasets for training.
Converts all downloaded datasets to unified JSONL format with \n fixes.
"""

import json
import os
from pathlib import Path
from datasets import load_from_disk, load_dataset
import re

def fix_newlines(text):
    """Fix literal \n characters in text"""
    if not isinstance(text, str):
        return text
    
    # Replace literal \n with actual newlines
    text = text.replace('\\n', '\n')
    # Clean up any double newlines
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def convert_conversational_dataset(dataset_path, output_file):
    """Convert a conversational dataset to our format"""
    print(f"📚 Converting {dataset_path}...")
    
    try:
        # Load the dataset
        ds = load_from_disk(dataset_path)
        train_data = ds['train']
        
        converted_count = 0
        with open(output_file, 'w') as f:
            for example in train_data:
                # Convert based on available keys
                if 'instruction' in example and 'response' in example:
                    # Airoboros/Dolly format
                    instruction = fix_newlines(example['instruction'])
                    response = fix_newlines(example['response'])
                    
                    # Add context if available
                    if 'context' in example and example['context']:
                        context = fix_newlines(example['context'])
                        instruction = f"{instruction}\n\nContext: {context}"
                    
                    entry = {
                        "messages": [
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": response}
                        ]
                    }
                    
                elif 'input' in example and 'output' in example:
                    # Open-Platypus format
                    user_input = fix_newlines(example['input'])
                    output = fix_newlines(example['output'])
                    
                    # Add instruction if available
                    if 'instruction' in example and example['instruction']:
                        instruction = fix_newlines(example['instruction'])
                        user_input = f"{instruction}\n\n{user_input}"
                    
                    entry = {
                        "messages": [
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": output}
                        ]
                    }
                else:
                    continue
                
                f.write(json.dumps(entry) + '\n')
                converted_count += 1
        
        print(f"  ✅ Converted {converted_count} examples to {output_file}")
        return converted_count
        
    except Exception as e:
        print(f"  ❌ Error converting {dataset_path}: {e}")
        return 0

def convert_cybersec_datasets(cybersec_dir, output_file):
    """Convert cybersecurity-specific datasets"""
    print(f"🔒 Converting cybersecurity datasets from {cybersec_dir}...")
    
    converted_count = 0
    
    # Network security questions
    network_file = cybersec_dir / "James4Ever0_network_security_questions" / "network_security_questions.txt"
    if network_file.exists():
        print(f"  📝 Processing network security questions...")
        with open(network_file, 'r', encoding='utf-8') as f:
            questions = [line.strip().strip('"') for line in f if line.strip()]
        
        # Create Q&A pairs for cybersecurity questions
        with open(output_file, 'a') as f:
            for question in questions[:1000]:  # Limit to first 1000 to avoid overwhelming
                if question and len(question) > 10:  # Filter out very short questions
                    entry = {
                        "messages": [
                            {"role": "user", "content": f"Cybersecurity question: {question}"},
                            {"role": "assistant", "content": f"This is a cybersecurity-related question about: {question}. I would need more context to provide a detailed answer, but this relates to network security, malware protection, firewalls, or antivirus software."}
                        ]
                    }
                    f.write(json.dumps(entry) + '\n')
                    converted_count += 1
        
        print(f"  ✅ Converted {converted_count} cybersecurity questions")
    
    return converted_count

def merge_datasets():
    """Merge all converted datasets into final training file"""
    print("🔄 Merging all enhanced datasets...")
    
    # Output directory
    output_dir = Path("/Users/danielrodrigo/Workspace/PyScience/datasets")
    enhanced_file = output_dir / "enhanced_cybersec_training_data.jsonl"
    
    # Datasets to convert
    conversational_datasets = [
        output_dir / "data_collection/cybersecurity_datasets/databricks_databricks-dolly-15k",
        output_dir / "data_collection/cybersecurity_datasets/garage-bAInd_Open-Platypus", 
        output_dir / "data_collection/cybersecurity_datasets/jondurbin_airoboros-2.2.1"
    ]
    
    cybersec_dir = Path("/Users/danielrodrigo/Workspace/cybersecurity_datasets/new_datasets")
    
    total_examples = 0
    
    # Clear the output file
    with open(enhanced_file, 'w') as f:
        pass
    
    # Convert conversational datasets
    for dataset_path in conversational_datasets:
        if dataset_path.exists():
            count = convert_conversational_dataset(dataset_path, enhanced_file)
            total_examples += count
    
    # Convert cybersecurity-specific datasets
    count = convert_cybersec_datasets(cybersec_dir, enhanced_file)
    total_examples += count
    
    # Merge with existing cleaned data
    existing_cleaned = output_dir / "heimdall_cleaned_training_data.jsonl"
    if existing_cleaned.exists():
        print("📚 Adding existing cleaned Heimdall data...")
        with open(existing_cleaned, 'r') as src, open(enhanced_file, 'a') as dst:
            for line in src:
                dst.write(line)
                total_examples += 1
    
    print(f"\n🎉 Enhanced dataset created: {enhanced_file}")
    print(f"📊 Total examples: {total_examples:,}")
    
    return enhanced_file, total_examples

def main():
    print("🚀 Preparing Enhanced Cybersecurity Training Dataset")
    print("=" * 60)
    
    # Create the merged enhanced dataset
    enhanced_file, total_examples = merge_datasets()
    
    # Verify the output
    print(f"\n✅ Dataset preparation complete!")
    print(f"📁 Output file: {enhanced_file}")
    print(f"📊 Total training examples: {total_examples:,}")
    
    # Show sample
    print(f"\n📖 Sample from enhanced dataset:")
    with open(enhanced_file, 'r') as f:
        sample = json.loads(f.readline())
        print(json.dumps(sample, indent=2))

if __name__ == "__main__":
    main()
