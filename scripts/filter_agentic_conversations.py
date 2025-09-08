#!/usr/bin/env python3
"""
Filter out shallow or generic conversations from an agentic-augmented JSONL dataset.
- Removes entries with <3 turns
- Removes entries with placeholder/generic content
- Prints a summary of filtering
"""
import json
import sys
from pathlib import Path

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "datasets/primus_training/train_agentic_augmented.jsonl"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "datasets/primus_training/train_agentic_filtered.jsonl"

GENERIC_PHRASES = [
    "[User question from LLM output]",
    "[Assistant follow-up from LLM output]",
    "[User reply from LLM output]",
    "[Assistant final answer from LLM output]",
    "LLM OUTPUT PLACEHOLDER"
]


def is_generic(msg):
    return any(phrase in msg for phrase in GENERIC_PHRASES)

def filter_conversation(conv):
    messages = conv.get("messages", [])
    if len(messages) < 3:
        return False
    for m in messages:
        if is_generic(m.get("content", "")):
            return False
    return True

def main():
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)
    kept, dropped = 0, 0
    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            conv = json.loads(line)
            if filter_conversation(conv):
                fout.write(json.dumps(conv) + "\n")
                kept += 1
            else:
                dropped += 1
    print(f"Filtered: kept {kept}, dropped {dropped}")

if __name__ == "__main__":
    main()
