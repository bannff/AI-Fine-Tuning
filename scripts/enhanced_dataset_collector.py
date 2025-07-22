#!/usr/bin/env python3
# ...
# Enhanced Cybersecurity Dataset Collector
# Downloads and processes high-quality cybersecurity datasets for improved training
# ...
# (rest of the existing code)

def main():
    """Download and process enhanced cybersecurity datasets"""
    # Create output directory
    output_dir = "/Users/danielrodrigo/Workspace/PyScience/datasets/enhanced_cybersec_datasets"
    os.makedirs(output_dir, exist_ok=True)
    # ...
    # (rest of the existing code)

if __name__ == "__main__":
    main()

    # Move this file to scripts/
    # (additional code if necessary)
#!/usr/bin/env python3
"""
Enhanced Cybersecurity Dataset Collector
Downloads and processes high-quality cybersecurity datasets for improved training
"""

import os
import json
import re
import sys
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Initialize tiktoken encoder for fast and accurate token counting
# Using cl100k_base which is used by GPT-3.5/GPT-4 - good general purpose tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Fast and accurate token counting using tiktoken"""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(tokenizer.encode(text))

def estimate_tokens(text):
    """Deprecated - use count_tokens() for accurate counting"""
    return count_tokens(text)

def split_long_content(content, max_tokens=400):
    """Split content that's too long into smaller chunks while preserving meaning"""
    
    current_tokens = count_tokens(content)
    if current_tokens <= max_tokens:
        return [content]
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed limit, start new chunk
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
        if current_chunk and count_tokens(test_chunk) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Single sentence is too long, split it by words
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    test_word_chunk = temp_chunk + " " + word if temp_chunk else word
                    if count_tokens(test_word_chunk) > max_tokens:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            # Single word is too long, truncate it
                            # This is rare but can happen with very long URLs or encoded strings
                            truncated_word = word
                            while count_tokens(truncated_word) > max_tokens and len(truncated_word) > 10:
                                truncated_word = truncated_word[:-10]
                            chunks.append(truncated_word)
                            temp_chunk = ""
                    else:
                        temp_chunk = test_word_chunk
                
                if temp_chunk:
                    current_chunk = temp_chunk
        else:
            current_chunk = test_chunk
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [content[:100]]  # Fallback

def split_conversation_by_tokens(messages, max_tokens=400):
    """Split a conversation into smaller conversations if it's too long"""
    conversations = []
    
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            user_msg = messages[i]
            assistant_msg = messages[i + 1]
            
            # Split user content if too long
            user_chunks = split_long_content(user_msg['content'], max_tokens)
            assistant_chunks = split_long_content(assistant_msg['content'], max_tokens)
            
            # Create conversations from chunks
            max_chunks = max(len(user_chunks), len(assistant_chunks))
            
            for j in range(max_chunks):
                user_content = user_chunks[j] if j < len(user_chunks) else user_chunks[-1]
                assistant_content = assistant_chunks[j] if j < len(assistant_chunks) else assistant_chunks[-1]
                
                # Add context if this is a continuation
                if j > 0:
                    user_content = f"[Continued] {user_content}"
                    assistant_content = f"[Continued] {assistant_content}"
                
                conversations.append({
                    'messages': [
                        {'role': 'user', 'content': user_content},
                        {'role': 'assistant', 'content': assistant_content}
                    ]
                })
    
    return conversations if conversations else [{'messages': messages}]

def clean_text(text):
    """Clean text to fix newline and formatting issues"""
    if not isinstance(text, str):
        return text
    
    # Fix literal \n characters
    text = text.replace('\\n', '\n')
    
    # Remove excessive whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Clean up formatting
    text = text.strip()
    
    return text

def process_dataset_to_mlx_format(dataset_name, output_path, max_samples=5000, max_tokens=400):
    """Process a HuggingFace dataset into MLX-LM format with token length control"""
    print(f"📥 Processing {dataset_name}...")
    
    try:
        # Load dataset
        dataset = load_dataset(dataset_name, split='train')
        
        # Limit samples to avoid huge files
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        mlx_data = []
        truncated_count = 0
        
        for item in tqdm(dataset, desc=f"Converting {dataset_name}"):
            # Handle different dataset formats
            if 'messages' in item:
                # Already in chat format
                messages = item['messages']
                if isinstance(messages, list) and len(messages) >= 2:
                    # Clean the messages
                    cleaned_messages = []
                    for msg in messages:
                        if isinstance(msg, dict) and 'content' in msg:
                            cleaned_content = clean_text(msg['content'])
                            cleaned_messages.append({
                                'role': msg.get('role', 'user'),
                                'content': cleaned_content
                            })
                    
                    if len(cleaned_messages) >= 2:
                        # Split conversation if too long
                        conversations = split_conversation_by_tokens(cleaned_messages, max_tokens)
                        mlx_data.extend(conversations)
                        if len(conversations) > 1:
                            truncated_count += 1
            
            elif 'instruction' in item and 'output' in item:
                # Instruction-output format
                instruction = clean_text(item['instruction'])
                output = clean_text(item['output'])
                
                if instruction and output:
                    # Split if too long
                    instruction_chunks = split_long_content(instruction, max_tokens)
                    output_chunks = split_long_content(output, max_tokens)
                    
                    max_chunks = max(len(instruction_chunks), len(output_chunks))
                    for j in range(max_chunks):
                        inst_content = instruction_chunks[j] if j < len(instruction_chunks) else instruction_chunks[-1]
                        out_content = output_chunks[j] if j < len(output_chunks) else output_chunks[-1]
                        
                        if j > 0:
                            inst_content = f"[Continued] {inst_content}"
                            out_content = f"[Continued] {out_content}"
                        
                        mlx_data.append({
                            'messages': [
                                {'role': 'user', 'content': inst_content},
                                {'role': 'assistant', 'content': out_content}
                            ]
                        })
                    
                    if max_chunks > 1:
                        truncated_count += 1
            
            elif 'question' in item and 'answer' in item:
                # Q&A format
                question = clean_text(item['question'])
                answer = clean_text(item['answer'])
                
                if question and answer:
                    # Split if too long
                    question_chunks = split_long_content(question, max_tokens)
                    answer_chunks = split_long_content(answer, max_tokens)
                    
                    max_chunks = max(len(question_chunks), len(answer_chunks))
                    for j in range(max_chunks):
                        q_content = question_chunks[j] if j < len(question_chunks) else question_chunks[-1]
                        a_content = answer_chunks[j] if j < len(answer_chunks) else answer_chunks[-1]
                        
                        if j > 0:
                            q_content = f"[Continued] {q_content}"
                            a_content = f"[Continued] {a_content}"
                        
                        mlx_data.append({
                            'messages': [
                                {'role': 'user', 'content': q_content},
                                {'role': 'assistant', 'content': a_content}
                            ]
                        })
                    
                    if max_chunks > 1:
                        truncated_count += 1
        
        # Save to JSONL
        if mlx_data:
            with open(output_path, 'w') as f:
                for item in mlx_data:
                    f.write(json.dumps(item) + '\n')
            
            print(f"✅ Saved {len(mlx_data)} samples to {output_path}")
            if truncated_count > 0:
                print(f"📊 Split {truncated_count} long sequences to prevent token truncation")
            return len(mlx_data)
        else:
            print(f"❌ No suitable data found in {dataset_name}")
            return 0
            
    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {e}")
        return 0

def main():
    """Download and process enhanced cybersecurity datasets"""
    
    # Create output directory
    output_dir = "/Users/danielrodrigo/Workspace/PyScience/datasets/enhanced_cybersec_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Priority datasets (Tier 1)
    tier1_datasets = [
        "Vanessasml/cybersecurity_32k_instruction_input_output",
        "Nitral-AI/Cybersecurity-ShareGPT", 
        "vinitvek/cybersecurityattacks",
        "bnsapa/cybersecurity-ner",
        "schooly/Cyber-Security-Breaches"
    ]
    
    total_samples = 0
    successful_datasets = []
    
    print("🚀 Starting Enhanced Cybersecurity Dataset Collection")
    print("=" * 60)
    
    for dataset_name in tier1_datasets:
        output_file = f"{output_dir}/{dataset_name.replace('/', '_')}.jsonl"
        
        samples = process_dataset_to_mlx_format(
            dataset_name, 
            output_file, 
            max_samples=3000  # Limit per dataset
        )
        
        if samples > 0:
            total_samples += samples
            successful_datasets.append((dataset_name, samples))
        
        print("-" * 40)
    
    print(f"\\n✅ Collection Complete!")
    print(f"📊 Total samples collected: {total_samples}")
    print(f"📁 Successful datasets: {len(successful_datasets)}")
    
    for name, count in successful_datasets:
        print(f"  - {name}: {count} samples")
    
    # Create combined dataset
    combined_file = f"{output_dir}/enhanced_cybersec_combined.jsonl"
    print(f"\\n🔄 Creating combined dataset: {combined_file}")
    
    with open(combined_file, 'w') as outfile:
        for name, _ in successful_datasets:
            input_file = f"{output_dir}/{name.replace('/', '_')}.jsonl"
            if os.path.exists(input_file):
                with open(input_file, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
    
    print(f"✅ Combined dataset created with {total_samples} total samples")
    print(f"📁 Location: {combined_file}")

def fix_existing_data():
    """Fix formatting issues in existing training data"""
    print("🔧 Fixing existing data formatting issues...")
    
    data_files = [
        "cybersecurity_datasets/processed/heimdall_merged_cleaned.jsonl",
        "cybersecurity_datasets/processed/cybersecurity_agentic_clean_train.jsonl",
        "cybersecurity_datasets/processed/cybersecurity_agentic_clean_valid.jsonl"
    ]
    
    total_fixed = 0
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"📝 Processing {file_path}...")
            fixed_file = file_path.replace('.jsonl', '_fixed.jsonl')
            fixed_count = 0
            
            with open(file_path, 'r') as infile, open(fixed_file, 'w') as outfile:
                for line in infile:
                    try:
                        data = json.loads(line.strip())
                        
                        # Fix messages content
                        if 'messages' in data:
                            for message in data['messages']:
                                if 'content' in message:
                                    original = message['content']
                                    message['content'] = clean_text(original)
                                    if original != message['content']:
                                        fixed_count += 1
                        
                        outfile.write(json.dumps(data) + '\n')
                        
                    except json.JSONDecodeError:
                        continue
            
            # Replace original with fixed version
            os.rename(fixed_file, file_path)
            print(f"✅ Fixed {fixed_count} formatting issues in {file_path}")
            total_fixed += fixed_count
    
    print(f"🎉 Total formatting issues fixed: {total_fixed}")
    return total_fixed

def download_cybersec_datasets():
    """Download specific cybersecurity datasets under 10GB"""
    print("🛡️ Downloading cybersecurity datasets...")
    
    # Priority cybersecurity datasets under 10GB
    cybersec_datasets = [
        {
            'name': 'cw1521/ember2018-malware-v2',
            'description': 'EMBER 2018 Malware Dataset v2 - 1M malware/benign samples',
            'priority': 'high'
        },
        {
            'name': 'joyce8/EMBER2024', 
            'description': 'EMBER 2024 - 3.2M files across Win32/64, .NET, APK, ELF, PDF',
            'priority': 'high'
        },
        {
            'name': 'dtrizna/quovadis-ember',
            'description': 'Contextual and behavioral malware analysis dataset',
            'priority': 'medium'
        },
        {
            'name': 'srimeenakshiks/Android-Malware-Dataset',
            'description': 'Android malware detection features',
            'priority': 'medium'
        },
        {
            'name': 'James4Ever0/network_security_questions',
            'description': 'Network security Q&A dataset',
            'priority': 'low'
        }
    ]
    
    successful_datasets = []
    total_samples = 0
    
    # Create output directory
    output_dir = "cybersecurity_datasets/new_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_info in cybersec_datasets:
        dataset_name = dataset_info['name']
        description = dataset_info['description']
        priority = dataset_info['priority']
        
        print(f"\\n📥 Downloading {dataset_name} ({priority} priority)")
        print(f"📄 {description}")
        
        try:
            # For now, create a simple download script since we don't have the datasets library
            download_script = f"""
# Download script for {dataset_name}
# Priority: {priority}
# Description: {description}
# Command: huggingface-cli download {dataset_name} --repo-type dataset --local-dir {output_dir}/{dataset_name.replace('/', '_')}
"""
            
            script_file = f"{output_dir}/download_{dataset_name.replace('/', '_')}.sh"
            with open(script_file, 'w') as f:
                f.write(download_script)
            
            print(f"✅ Download script created: {script_file}")
            successful_datasets.append((dataset_name, description))
            
        except Exception as e:
            print(f"❌ Error preparing {dataset_name}: {str(e)}")
    
    print(f"\\n🎉 Prepared {len(successful_datasets)} cybersecurity datasets for download")
    
    # Create a master download script
    master_script = f"{output_dir}/download_all_cybersec.sh"
    with open(master_script, 'w') as f:
        f.write("#!/bin/bash\\n")
        f.write("# Master script to download all cybersecurity datasets\\n\\n")
        f.write("echo 'Starting cybersecurity dataset downloads...'\\n\\n")
        
        for dataset_info in cybersec_datasets:
            dataset_name = dataset_info['name']
            f.write(f"echo 'Downloading {dataset_name}...'\\n")
            f.write(f"huggingface-cli download {dataset_name} --repo-type dataset --local-dir {output_dir}/{dataset_name.replace('/', '_')}\\n\\n")
    
    # Make script executable
    os.chmod(master_script, 0o755)
    
    print(f"📜 Master download script created: {master_script}")
    print("\\n🚀 To download all datasets, run:")
    print(f"   bash {master_script}")
    
    return successful_datasets

def create_superior_merged_dataset():
    """Create a superior merged dataset from all available sources"""
    print("🚀 Creating superior merged cybersecurity dataset...")
    
    output_dir = "/Users/danielrodrigo/Workspace/PyScience/datasets/superior_cybersec_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all data sources
    data_sources = [
        {
            'path': '/Users/danielrodrigo/Workspace/PyScience/datasets/enhanced_cybersec_datasets/enhanced_cybersec_combined.jsonl',
            'description': 'Enhanced cybersecurity datasets',
            'priority': 'high'
        },
        {
            'path': '/Users/danielrodrigo/Workspace/PyScience/datasets/data_collection/cybersecurity_datasets/databricks_databricks-dolly-15k',
            'description': 'Databricks Dolly 15k (instruction dataset)',
            'priority': 'high',
            'format': 'huggingface'
        },
        {
            'path': '/Users/danielrodrigo/Workspace/PyScience/datasets/data_collection/cybersecurity_datasets/garage-bAInd_Open-Platypus',
            'description': 'Open Platypus dataset',
            'priority': 'medium',
            'format': 'huggingface'
        },
        {
            'path': '/Users/danielrodrigo/Workspace/PyScience/datasets/data_collection/cybersecurity_datasets/jondurbin_airoboros-2.2.1',
            'description': 'Airoboros conversational dataset',
            'priority': 'medium',
            'format': 'huggingface'
        },
        {
            'path': '/Users/danielrodrigo/Workspace/cybersecurity_datasets/new_datasets/James4Ever0_network_security_questions',
            'description': 'Network security Q&A',
            'priority': 'high',
            'format': 'huggingface'
        },
        {
            'path': '/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_datasets/processed/heimdall_merged_cleaned.jsonl',
            'description': 'Original Heimdall cleaned dataset',
            'priority': 'medium'
        }
    ]
    
    combined_data = []
    total_samples = 0
    
    for source in data_sources:
        print(f"\n📥 Processing: {source['description']} ({source['priority']} priority)")
        
        try:
            if source.get('format') == 'huggingface' and os.path.isdir(source['path']):
                # Process HuggingFace dataset format
                dataset_files = [f for f in os.listdir(source['path']) if f.endswith('.json') or f.endswith('.jsonl')]
                
                for file in dataset_files:
                    file_path = os.path.join(source['path'], file)
                    
                    with open(file_path, 'r') as f:
                        if file.endswith('.json'):
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data[:1000]:  # Limit per file
                                    processed_item = convert_to_chat_format(item)
                                    if processed_item:
                                        combined_data.append(processed_item)
                                        total_samples += 1
                        else:
                            for line in f:
                                if total_samples % 1000 == 0:
                                    print(f"  Processed {total_samples} samples...")
                                
                                try:
                                    item = json.loads(line.strip())
                                    processed_item = convert_to_chat_format(item)
                                    if processed_item:
                                        combined_data.append(processed_item)
                                        total_samples += 1
                                        
                                    # Limit total samples per source
                                    if total_samples >= 50000:
                                        break
                                except json.JSONDecodeError:
                                    continue
                                    
            elif source['path'].endswith('.jsonl') and os.path.exists(source['path']):
                # Process JSONL files directly
                with open(source['path'], 'r') as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            processed_item = convert_to_chat_format(item)
                            if processed_item:
                                combined_data.append(processed_item)
                                total_samples += 1
                        except json.JSONDecodeError:
                            continue
            
            print(f"  ✅ Added {len([d for d in combined_data if total_samples > len(combined_data) - 1000])} samples")
            
        except Exception as e:
            print(f"  ❌ Error processing {source['description']}: {e}")
    
    # Create final superior dataset
    superior_file = f"{output_dir}/superior_cybersec_combined.jsonl"
    
    print(f"\n🔄 Writing superior combined dataset...")
    with open(superior_file, 'w') as f:
        for item in combined_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Superior dataset created!")
    print(f"📁 Location: {superior_file}")
    print(f"📊 Total samples: {total_samples}")
    
    return superior_file, total_samples

def convert_to_chat_format(item):
    """Convert various dataset formats to standardized chat format"""
    try:
        # Already in chat format
        if 'messages' in item:
            messages = item['messages']
            if isinstance(messages, list) and len(messages) >= 2:
                # Clean the messages
                cleaned_messages = []
                for msg in messages:
                    if isinstance(msg, dict) and 'content' in msg:
                        cleaned_content = clean_text(msg['content'])
                        if cleaned_content:
                            cleaned_messages.append({
                                'role': msg.get('role', 'user'),
                                'content': cleaned_content
                            })
                
                if len(cleaned_messages) >= 2:
                    return {'messages': cleaned_messages}
        
        # Instruction-response format
        elif 'instruction' in item:
            instruction = clean_text(item.get('instruction', ''))
            response = clean_text(item.get('response', item.get('output', '')))
            
            if instruction and response:
                return {
                    'messages': [
                        {'role': 'user', 'content': instruction},
                        {'role': 'assistant', 'content': response}
                    ]
                }
        
        # Question-answer format
        elif 'question' in item and 'answer' in item:
            question = clean_text(item['question'])
            answer = clean_text(item['answer'])
            
            if question and answer:
                return {
                    'messages': [
                        {'role': 'user', 'content': question},
                        {'role': 'assistant', 'content': answer}
                    ]
                }
        
        # Input-output format
        elif 'input' in item and 'output' in item:
            input_text = clean_text(item['input'])
            output_text = clean_text(item['output'])
            
            if input_text and output_text:
                return {
                    'messages': [
                        {'role': 'user', 'content': input_text},
                        {'role': 'assistant', 'content': output_text}
                    ]
                }
    
    except Exception:
        pass
    
    return None

def reprocess_dataset_with_token_limits():
    """Reprocess existing superior dataset to fix token length issues"""
    print("🔧 Reprocessing superior dataset to fix token length issues...")
    
    input_file = "/Users/danielrodrigo/Workspace/PyScience/datasets/superior_cybersec_dataset/superior_cybersec_combined.jsonl"
    output_file = "/Users/danielrodrigo/Workspace/PyScience/datasets/superior_cybersec_dataset/superior_cybersec_token_limited.jsonl"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return 0
    
    processed_data = []
    split_count = 0
    total_count = 0
    
    print("📖 Reading existing dataset...")
    with open(input_file, 'r') as f:
        for line in tqdm(f, desc="Processing sequences"):
            try:
                item = json.loads(line.strip())
                total_count += 1
                
                if 'messages' in item:
                    messages = item['messages']
                    
                    # Check if any message content is too long
                    needs_splitting = False
                    for msg in messages:
                        if 'content' in msg and estimate_tokens(msg['content']) > 400:
                            needs_splitting = True
                            break
                    
                    if needs_splitting:
                        # Split the conversation
                        conversations = split_conversation_by_tokens(messages, max_tokens=400)
                        processed_data.extend(conversations)
                        split_count += 1
                    else:
                        # Keep as is
                        processed_data.append(item)
                else:
                    # Non-message format, keep as is
                    processed_data.append(item)
                    
            except json.JSONDecodeError:
                continue
    
    # Write processed data
    print(f"💾 Writing token-limited dataset...")
    with open(output_file, 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Reprocessing complete!")
    print(f"📊 Original sequences: {total_count}")
    print(f"📊 Split sequences: {split_count}")
    print(f"📊 Final sequences: {len(processed_data)}")
    print(f"📁 Output file: {output_file}")
    
    return len(processed_data)

def cleanup_old_trainings():
    """Clean up old training outputs to keep workspace clean"""
    print("🧹 Cleaning up old training outputs...")
    
    cleanup_paths = [
        "/Users/danielrodrigo/Workspace/PyScience/cybersecurity_finetuned_models/mlx_adapters_enhanced_v1",
        "/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_finetuned_models/mlx_adapters_v3",
        "/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_finetuned_models/mlx_adapters_v2",
        "/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_finetuned_models/mlx_adapters_v1"
    ]
    
    removed_count = 0
    for path in cleanup_paths:
        if os.path.exists(path):
            print(f"🗑️  Removing {path}")
            import shutil
            shutil.rmtree(path)
            removed_count += 1
            print(f"  ✅ Removed")
    
    print(f"🎉 Cleaned up {removed_count} old training directories")

def download_small_cybersec_datasets():
    """Download small, publicly available cybersecurity datasets"""
    print("🛡️ Downloading small cybersecurity datasets...")
    
    # Small, direct download datasets
    small_datasets = [
        {
            'name': 'kdd_cup_1999',
            'url': 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz',
            'filename': 'kdd_cup_1999_10percent.gz',
            'size': '~2.1MB',
            'description': 'Network intrusion detection (10% sample)'
        },
        {
            'name': 'nsl_kdd',
            'url': 'https://github.com/defcom17/NSL_KDD/raw/master/KDDTrain%2B.txt',
            'filename': 'nsl_kdd_train.txt',
            'size': '~5MB',
            'description': 'Improved KDD dataset for intrusion detection'
        },
        {
            'name': 'nsl_kdd_test',
            'url': 'https://github.com/defcom17/NSL_KDD/raw/master/KDDTest%2B.txt',
            'filename': 'nsl_kdd_test.txt', 
            'size': '~2MB',
            'description': 'NSL-KDD test set'
        }
    ]
    
    # Create output directory
    output_dir = "cybersecurity_datasets/small_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    successful_downloads = []
    
    for dataset in small_datasets:
        print(f"\\n📥 Downloading {dataset['name']} ({dataset['size']})")
        print(f"📄 {dataset['description']}")
        
        try:
            filepath = f"{output_dir}/{dataset['filename']}"
            
            # Use curl to download
            cmd = f"curl -L -o {filepath} {dataset['url']}"
            print(f"🔄 Running: {cmd}")
            
            result = os.system(cmd)
            
            if result == 0 and os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"✅ Downloaded {dataset['name']} ({file_size:,} bytes)")
                successful_downloads.append(dataset['name'])
            else:
                print(f"❌ Failed to download {dataset['name']}")
                
        except Exception as e:
            print(f"❌ Error downloading {dataset['name']}: {str(e)}")
    
    print(f"\\n🎉 Successfully downloaded {len(successful_downloads)} datasets:")
    for name in successful_downloads:
        print(f"  ✅ {name}")
    
    # Create conversion script for these datasets
    conversion_script = f"{output_dir}/convert_to_training_format.py"
    with open(conversion_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Convert downloaded cybersecurity datasets to MLX training format
"""
import os
import json
import pandas as pd
import gzip

def convert_kdd_to_training():
    """Convert KDD Cup 1999 to conversational format"""
    print("Converting KDD Cup 1999 dataset...")
    
    # Define attack types for better training
    attack_types = {
        'normal': 'normal traffic',
        'back': 'back door attack',
        'buffer_overflow': 'buffer overflow attack',
        'ftp_write': 'FTP write attack',
        'guess_passwd': 'password guessing attack',
        'imap': 'IMAP attack',
        'ipsweep': 'IP sweep reconnaissance',
        'land': 'land attack (SYN flooding)',
        'loadmodule': 'load module attack',
        'multihop': 'multi-hop attack',
        'neptune': 'Neptune DoS attack',
        'nmap': 'Nmap port scanning',
        'perl': 'Perl script attack',
        'phf': 'PHF attack',
        'pod': 'ping of death attack',
        'portsweep': 'port sweep reconnaissance',
        'rootkit': 'rootkit installation',
        'satan': 'Satan reconnaissance',
        'smurf': 'Smurf DoS attack',
        'spy': 'spy attack',
        'teardrop': 'teardrop attack',
        'warezclient': 'warez client attack',
        'warezmaster': 'warez master attack'
    }
    
    # This would process the KDD data and convert to conversational format
    # For now, create a sample
    sample_data = []
    for attack, description in list(attack_types.items())[:10]:
        sample_data.append({
            "messages": [
                {"role": "user", "content": f"What is a {attack} in cybersecurity?"},
                {"role": "assistant", "content": f"A {attack} refers to {description}. This is a type of network security threat that security professionals need to detect and prevent."}
            ]
        })
    
    output_file = "kdd_cybersec_training.jsonl"
    with open(output_file, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\\n')
    
    print(f"✅ Created {output_file} with {len(sample_data)} training examples")

if __name__ == "__main__":
    convert_kdd_to_training()
''')
    
    print(f"\\n📝 Created conversion script: {conversion_script}")
    print("Run it with: python3 convert_to_training_format.py")
    
    return successful_downloads

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--fix-data":
            fix_existing_data()
        elif sys.argv[1] == "--download-cybersec":
            download_cybersec_datasets()
        elif sys.argv[1] == "--download-small":
            download_small_cybersec_datasets()
        elif sys.argv[1] == "--create-superior":
            cleanup_old_trainings()
            superior_file, total_samples = create_superior_merged_dataset()
            print(f"\n🎉 Superior dataset ready for training!")
            print(f"📁 File: {superior_file}")
            print(f"📊 Samples: {total_samples}")
        elif sys.argv[1] == "--fix-tokens":
            # Remove old training checkpoints since they're based on truncated data
            cleanup_old_trainings()
            # Reprocess dataset with token limits
            sample_count = reprocess_dataset_with_token_limits()
            print(f"\n🎉 Token-limited dataset ready!")
            print(f"📊 Total samples: {sample_count}")
            print("\n⚠️  Removed old training checkpoints - they were based on truncated data")
            print("🚀 Ready to restart training with proper token limits")
        elif sys.argv[1] == "--cleanup":
            cleanup_old_trainings()
        else:
            print("Available options:")
            print("  --fix-data          Fix formatting in existing data")
            print("  --download-cybersec Download cybersecurity datasets")  
            print("  --download-small    Download small datasets")
            print("  --create-superior   Create superior merged dataset + cleanup")
            print("  --fix-tokens        Fix token length issues in existing dataset")
            print("  --cleanup           Clean up old training outputs")
    else:
        main()
