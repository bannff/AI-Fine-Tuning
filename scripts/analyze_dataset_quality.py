#!/usr/bin/env python3
"""
Dataset Quality Analysis and Deduplication
==========================================

This script analyzes the cybersecurity dataset for:
1. Duplicate content detection and removal
2. Data quality assessment (length, content type, etc.)
3. Content diversity analysis
4. Final dataset statistics
"""

import json
import hashlib
import re
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

def hash_content(text):
    """Create a hash of the text content for duplicate detection."""
    # Normalize whitespace and convert to lowercase for better duplicate detection
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def analyze_text_quality(text):
    """Analyze the quality of a text sample."""
    if not isinstance(text, str):
        return False, "Not a string"
    
    text = text.strip()
    
    # Basic quality checks
    if len(text) < 20:
        return False, "Too short"
    
    if len(text) > 10000:
        return False, "Too long"
    
    # Check for reasonable text content
    word_count = len(text.split())
    if word_count < 5:
        return False, "Too few words"
    
    # Check for excessive repetition
    words = text.lower().split()
    if len(set(words)) / len(words) < 0.3:
        return False, "Too repetitive"
    
    # Check for proper conversation format (cybersecurity focus)
    has_user_assistant = any(pattern in text.lower() for pattern in [
        'user:', 'assistant:', 'human:', 'ai:', 'question:', 'answer:'
    ])
    
    # Check for cybersecurity content
    cybersec_keywords = [
        'security', 'vulnerability', 'attack', 'malware', 'encryption',
        'firewall', 'sql injection', 'xss', 'csrf', 'authentication',
        'authorization', 'threat', 'exploit', 'penetration', 'cyber'
    ]
    
    has_cybersec_content = any(keyword in text.lower() for keyword in cybersec_keywords)
    
    return True, {
        'length': len(text),
        'word_count': word_count,
        'has_conversation_format': has_user_assistant,
        'has_cybersec_content': has_cybersec_content,
        'uniqueness_ratio': len(set(words)) / len(words)
    }

def analyze_dataset(input_path, output_path):
    """Analyze dataset and remove duplicates while maintaining quality."""
    print(f"Analyzing {input_path}")
    
    seen_hashes = set()
    content_stats = {
        'total_samples': 0,
        'duplicates': 0,
        'low_quality': 0,
        'kept_samples': 0,
        'quality_reasons': Counter(),
        'lengths': [],
        'word_counts': [],
        'cybersec_samples': 0,
        'conversation_samples': 0
    }
    
    # Count total lines first
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    kept_samples = []
    
    print("Processing samples...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines), 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                text = data.get('text', '')
                
                content_stats['total_samples'] += 1
                
                # Check for duplicates
                text_hash = hash_content(text)
                if text_hash in seen_hashes:
                    content_stats['duplicates'] += 1
                    continue
                
                seen_hashes.add(text_hash)
                
                # Analyze quality
                is_quality, quality_info = analyze_text_quality(text)
                
                if not is_quality:
                    content_stats['low_quality'] += 1
                    content_stats['quality_reasons'][quality_info] += 1
                    continue
                
                # Collect statistics for quality samples
                content_stats['lengths'].append(quality_info['length'])
                content_stats['word_counts'].append(quality_info['word_count'])
                
                if quality_info['has_cybersec_content']:
                    content_stats['cybersec_samples'] += 1
                
                if quality_info['has_conversation_format']:
                    content_stats['conversation_samples'] += 1
                
                # Keep this sample
                kept_samples.append(data)
                content_stats['kept_samples'] += 1
                
            except json.JSONDecodeError:
                content_stats['low_quality'] += 1
                content_stats['quality_reasons']['JSON decode error'] += 1
                continue
    
    # Write deduplicated and filtered dataset
    print(f"Writing cleaned dataset to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in kept_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return content_stats

def print_analysis_report(train_stats, valid_stats):
    """Print comprehensive analysis report."""
    print("\n" + "="*80)
    print("DATASET QUALITY ANALYSIS REPORT")
    print("="*80)
    
    for dataset_name, stats in [("TRAINING", train_stats), ("VALIDATION", valid_stats)]:
        print(f"\n{dataset_name} DATASET:")
        print("-" * 40)
        print(f"Total samples processed: {stats['total_samples']:,}")
        print(f"Duplicates removed: {stats['duplicates']:,}")
        print(f"Low quality removed: {stats['low_quality']:,}")
        print(f"Final samples kept: {stats['kept_samples']:,}")
        print(f"Retention rate: {stats['kept_samples']/stats['total_samples']*100:.1f}%")
        
        if stats['lengths']:
            print(f"\nContent Statistics:")
            print(f"Average length: {sum(stats['lengths'])/len(stats['lengths']):.0f} chars")
            print(f"Average words: {sum(stats['word_counts'])/len(stats['word_counts']):.0f}")
            print(f"Cybersecurity content: {stats['cybersec_samples']:,} ({stats['cybersec_samples']/stats['kept_samples']*100:.1f}%)")
            print(f"Conversation format: {stats['conversation_samples']:,} ({stats['conversation_samples']/stats['kept_samples']*100:.1f}%)")
        
        if stats['quality_reasons']:
            print(f"\nQuality Issues Found:")
            for reason, count in stats['quality_reasons'].most_common():
                print(f"  {reason}: {count:,}")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    total_kept = train_stats['kept_samples'] + valid_stats['kept_samples']
    total_original = train_stats['total_samples'] + valid_stats['total_samples']
    print(f"Total high-quality samples: {total_kept:,}")
    print(f"Overall retention rate: {total_kept/total_original*100:.1f}%")
    print("Dataset is ready for MLX-LM training!")
    print("="*80)

def main():
    """Main function to analyze both datasets."""
    base_dir = Path("/Users/danielrodrigo/Workspace/PyScience/datasets/primus_training")
    
    # Process training set
    train_input = base_dir / "train_fixed.jsonl"
    train_output = base_dir / "train_clean.jsonl"
    train_stats = analyze_dataset(train_input, train_output)
    
    # Process validation set
    valid_input = base_dir / "valid_fixed.jsonl"
    valid_output = base_dir / "valid_clean.jsonl"
    valid_stats = analyze_dataset(valid_input, valid_output)
    
    # Print comprehensive report
    print_analysis_report(train_stats, valid_stats)
    
    # Now replace the original files with the cleaned versions
    print("\nReplacing original fixed files with cleaned versions...")
    
    # Replace train_fixed.jsonl with train_clean.jsonl
    train_clean_path = base_dir / "train_clean.jsonl"
    train_final_path = base_dir / "train_fixed.jsonl"
    if train_clean_path.exists():
        train_clean_path.rename(train_final_path)
        print(f"✅ Updated {train_final_path}")
    
    # Replace valid_fixed.jsonl with valid_clean.jsonl
    valid_clean_path = base_dir / "valid_clean.jsonl"
    valid_final_path = base_dir / "valid_fixed.jsonl"
    if valid_clean_path.exists():
        valid_clean_path.rename(valid_final_path)
        print(f"✅ Updated {valid_final_path}")
    
    print("\n🎉 Dataset analysis and cleaning complete!")
    print("The cleaned, deduplicated datasets are now ready for training.")

if __name__ == "__main__":
    main()
