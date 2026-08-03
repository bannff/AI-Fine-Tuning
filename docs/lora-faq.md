# LoRA FAQ

## Does LoRA cost accuracy relative to full fine-tuning?

It depends on configuration, and the honest answer is "not necessarily, but not
automatically either."

The original paper reported LoRA matching or slightly exceeding full fine-tuning
on several benchmarks. On GPT-2 medium for E2E NLG it reported 70.4 BLEU against
68.2 for full fine-tuning; on DeBERTa across GLUE, 91.3% average against 91.1%.

More recent work ([Schulman et al., 2025](https://thinkingmachines.ai/blog/lora/))
is more precise about the conditions: LoRA matches full fine-tuning when applied
to all weight matrices at a rank sufficient for the dataset, and falls behind when
either condition is unmet. Restricting it to attention layers underperforms even
at a rank chosen to match the parameter count.

So the claim "LoRA never loses accuracy" is too strong. "LoRA matches full
fine-tuning when configured per
[docs/lora-practice.md](lora-practice.md)" is supportable.

## How much memory does it actually save?

The paper reports roughly 3x lower GPU memory and up to 10,000x fewer trainable
parameters for GPT-3 175B. The trainable-parameter figure is the headline but the
memory figure is what matters in practice, because the saving comes mostly from
not needing optimiser state and gradients for the frozen weights.

Base model weights still have to be resident. LoRA reduces the *training
overhead*, not the cost of holding the model.

## How many adapters can I keep?

Effectively unlimited. At tens of megabytes each, thousands are practical, and
they can be swapped at inference time against one resident base model.

## Can an adapter move between models?

No. Adapters are tied to the architecture and the specific weights they were
trained against. A Llama adapter will not load onto Phi-3, and module naming
differs between families in ways that fail confusingly rather than cleanly. See
the `qkv_proj` example in [docs/lora-practice.md](lora-practice.md).

## Should I merge before deploying?

If you are serving one adapter, yes — merging removes the small inference-time
overhead of computing `BAx` alongside `W₀x`. If you are serving many adapters
against a shared base, keep them unmerged so they stay swappable.

Merging is a real operation producing new weights. A directory containing an
`adapter_config.json` has not been merged, regardless of what it is named.

## What rank should I start with?

16 for a small task-specific dataset, which is this repository's default. Raise
toward 256 for post-training-scale SFT. For RL, ranks as low as 1 work, because
policy-gradient methods extract very little information per episode.

Change rank without changing `alpha`, and consider `use_rslora: true` if you plan
to sweep it.

## Why is my validation loss NaN?

Most often the learning rate is too high for the method. Full fine-tuning wants
roughly two orders of magnitude less than LoRA; `2e-6` versus `2e-4` are this
repository's defaults. Other things to try, in order: enable gradient
checkpointing, lower the batch size, add warmup.

## Is QLoRA just LoRA with a flag?

No, and this is worth being careful about. QLoRA needs the base model quantised.
On PyTorch `bitsandbytes` does that at load time. On MLX it is a separate
`mlx_lm convert -q` step that writes a new model directory.

Passing no quantisation option to `mlx_lm lora` gives you plain LoRA on a
full-precision base, not QLoRA. `finetune train` with `method: qlora` runs the
convert stage first, and `finetune plan` shows both stages before anything runs.

## Sources

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (Schulman et al., Thinking Machines Lab, 2025)
- [PEFT documentation](https://huggingface.co/docs/peft/)

Benchmark figures are cited from the linked papers. Content from external sources
is summarised and paraphrased rather than quoted.
