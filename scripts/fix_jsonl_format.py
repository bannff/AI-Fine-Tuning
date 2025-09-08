#!/usr/bin/env python3

# Fix JSON formatting in the cybersecurity training data
# Usage: python3 fix_jsonl_format.py input.jsonl output.jsonl

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
    """Wrap every line as {"text": ...}, deduplicate, and output MLX-LM compatible JSONL."""
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 fix_jsonl_format.py <input_file> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print("🔧 Fixing and deduplicating JSONL for MLX-LM...")

    seen = set()
    total = 0
    written = 0

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            total += 1
            # Try to flatten any double-encoded JSON
            text_val = None
            try:
                # Try to parse as JSON
                parsed = json.loads(line)
                if isinstance(parsed, dict) and "text" in parsed:
                    # If the value is itself a JSON string, parse again
                    inner = parsed["text"]
                    if isinstance(inner, str):
                        try:
                            inner_parsed = json.loads(inner)
                            if isinstance(inner_parsed, dict) and "text" in inner_parsed:
                                text_val = inner_parsed["text"]
                            else:
                                text_val = inner
                        except Exception:
                            text_val = inner
                    else:
                        text_val = str(inner)
                else:
                    # Not a dict with 'text', treat as string
                    text_val = line
            except Exception:
                # Not JSON, treat as plain string
                text_val = line
            text_val = text_val.strip()
            if text_val in seen:
                continue
            seen.add(text_val)
            obj = {"text": text_val}
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"✅ Wrote {written} unique samples out of {total} lines.")
    print(f"Output written to: {output_file}")
    return written

if __name__ == "__main__":
    fixed_count = fix_jsonl_format()
    print(f"🎯 Ready for training with {fixed_count} properly formatted samples!")
