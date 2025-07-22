#!/usr/bin/env python3
# Fix JSON formatting in the cybersecurity training data
# ...

def main():
    """Main function to fix JSONL formatting"""
    # ...
    # (rest of the existing code)

if __name__ == "__main__":
    main()

    # Move this file to scripts/
#!/usr/bin/env python3
"""
Fix JSON formatting in the cybersecurity training data
"""

import json
import os

def fix_jsonl_format():
    """Fix JSONL formatting issues"""
    input_file = "/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_datasets/processed/cybersec_enhanced_training.jsonl"
    output_file = "/Users/danielrodrigo/Workspace/PyScience/datasets/cybersecurity_datasets/processed/cybersec_enhanced_training_fixed.jsonl"
    
    print("🔧 Fixing JSONL formatting...")
    
    fixed_count = 0
    total_count = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            total_count += 1
            
            # Try to split multiple JSON objects on one line
            json_objects = []
            current_json = ""
            brace_count = 0
            
            for char in line:
                current_json += char
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Complete JSON object
                        try:
                            data = json.loads(current_json)
                            json_objects.append(data)
                            current_json = ""
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            current_json = ""
            
            # Write each JSON object on its own line
            for json_obj in json_objects:
                if 'messages' in json_obj and isinstance(json_obj['messages'], list):
                    outfile.write(json.dumps(json_obj) + '\n')
                    fixed_count += 1
    
    print(f"✅ Fixed {fixed_count} JSON objects from {total_count} lines")
    
    # Replace original with fixed version
    os.rename(output_file, input_file)
    print(f"📁 Replaced original file: {input_file}")
    
    return fixed_count

if __name__ == "__main__":
    fixed_count = fix_jsonl_format()
    print(f"🎯 Ready for training with {fixed_count} properly formatted samples!")
