#!/usr/bin/env python3
"""
Batch-augment a JSONL dataset of conversations using an LLM to make them more agentic and multi-turn.
- Reads input JSONL (each entry: {"messages": [...]})
- For each entry, prompts an LLM to expand it into a deeper, more agentic conversation (3-5 turns)
- Writes output JSONL

NOTE: This script is a template. You must fill in the LLM API call (e.g., OpenAI, Anthropic, local model, etc.)
"""
import json
import sys
from pathlib import Path
from typing import List, Dict
import time

# === CONFIG ===
INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "datasets/primus_training/train.jsonl"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "datasets/primus_training/train_agentic_augmented.jsonl"
MAX_EXAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 100  # For testing, limit to 100
SLEEP_BETWEEN = 1.0  # seconds between API calls (avoid rate limits)

# === LLM PROMPT ===
PROMPT_TEMPLATE = """
Given the following conversation between a user and an assistant, rewrite it as a deeper, more agentic multi-turn chat (at least 3-5 turns). The assistant should ask clarifying or follow-up questions, probe for more context, and demonstrate curiosity before giving a final answer. Make the conversation realistic and focused on cybersecurity.

Original conversation:
{original}

Agentic, multi-turn conversation:
"""

def call_llm(prompt: str) -> str:
    """
    Replace this stub with your LLM API call (OpenAI, Anthropic, local, etc.)
    For now, returns the original prompt for testing.
    """
    # TODO: Integrate your LLM API here
    # Example: return openai.ChatCompletion.create(...)
    return "[LLM OUTPUT PLACEHOLDER]\n" + prompt

def format_conversation(messages: List[Dict]) -> str:
    """
    Format a list of messages as a readable chat transcript for the LLM prompt.
    """
    lines = []
    for m in messages:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def parse_llm_output(llm_output: str) -> List[Dict]:
    """
    Parse the LLM's output back into a list of messages.
    This is a stub: in production, use a robust parser or instruct the LLM to output JSON.
    """
    # For now, just return a dummy 3-turn conversation
    return [
        {"role": "user", "content": "[User question from LLM output]"},
        {"role": "assistant", "content": "[Assistant follow-up from LLM output]"},
        {"role": "user", "content": "[User reply from LLM output]"},
        {"role": "assistant", "content": "[Assistant final answer from LLM output]"},
    ]

def main():
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)
    print(f"Reading: {input_path}")
    print(f"Writing: {output_path}")
    count = 0
    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            if count >= MAX_EXAMPLES:
                break
            data = json.loads(line)
            orig_msgs = data.get("messages", [])
            orig_text = format_conversation(orig_msgs)
            prompt = PROMPT_TEMPLATE.format(original=orig_text)
            llm_output = call_llm(prompt)
            new_msgs = parse_llm_output(llm_output)
            out = {"messages": new_msgs}
            fout.write(json.dumps(out) + "\n")
            count += 1
            print(f"Processed {count}", end="\r")
            time.sleep(SLEEP_BETWEEN)
    print(f"\nDone. Augmented {count} conversations.")

if __name__ == "__main__":
    main()
