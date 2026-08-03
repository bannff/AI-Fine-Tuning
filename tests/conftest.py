"""Shared fixtures.

The tiny-model fixture is the important one: it builds a real but minimal
causal LM and tokenizer on disk with no network access, so the smoke tests
prove the training paths execute without depending on the Hugging Face Hub.
That is what makes them runnable in CI.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HAS_TRL = importlib.util.find_spec("trl") is not None
HAS_MLX = importlib.util.find_spec("mlx_lm") is not None
# Either extra brings transformers and tokenizers, which is all the tiny-model
# fixture needs to build a model on disk.
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A ~50k-parameter Llama with a byte-level BPE tokenizer, built locally.

    Session-scoped because construction costs about a second and every smoke
    test can share one copy.
    """
    if not HAS_TRANSFORMERS:
        pytest.skip("needs the 'torch' or 'apple' extra to build a model")

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out = tmp_path_factory.mktemp("tiny-model")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["hello world question answer " * 20, "the quick brown fox jumps over the lazy dog " * 20],
        trainers.BpeTrainer(
            vocab_size=512,
            special_tokens=["<pad>", "<s>", "</s>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        chat_template=CHAT_TEMPLATE,
    )
    fast.save_pretrained(out)

    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=fast.vocab_size,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            pad_token_id=fast.pad_token_id,
        )
    ).save_pretrained(out)

    return out


@pytest.fixture
def chat_dataset(tmp_path: Path) -> Path:
    """A prepared split directory in ``chat`` format."""
    data_dir = tmp_path / "data"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"what is {i} plus one"},
                {"role": "assistant", "content": f"{i + 1}"},
            ]
        }
        for i in range(24)
    ]
    data_dir.mkdir()
    for name, subset in (("train.jsonl", rows[:16]), ("valid.jsonl", rows[16:])):
        (data_dir / name).write_text(
            "\n".join(json.dumps(r) for r in subset) + "\n", encoding="utf-8"
        )
    return data_dir
