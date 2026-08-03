# How LoRA works

LoRA (Low-Rank Adaptation) trains a small number of new parameters instead of
updating the model's own weights.

## Mechanism

A frozen weight matrix `W₀` gets a low-rank additive update, factored into two
much smaller matrices:

```
h = W₀x + ΔWx = W₀x + BAx = (W₀ + BA)x
```

- `W₀` — original weights, frozen.
- `A`, `B` — trainable, with inner dimension `r` (the rank).
- `r` — typically 8 to 256, against hidden dimensions in the thousands.

`B` is initialised to zero, so `BA = 0` and the adapted model starts out
numerically identical to the base. Training therefore begins from the base
model's behaviour rather than from a perturbation of it.

Because the update is additive and linear, `BA` can be folded into `W₀` after
training, which is why a merged adapter costs nothing at inference time.

## What this buys

**Memory.** Optimiser state and gradients are only needed for `A` and `B`. This
is the dominant saving, and it is what makes single-machine fine-tuning of a
multi-billion-parameter model feasible at all.

**Storage.** An adapter is tens of megabytes against tens of gigabytes for a full
model copy. Many task-specific adapters can share one base.

**Swappability.** Adapters can be loaded and exchanged at inference time against a
single resident base model.

**No inference latency**, once merged. An unmerged adapter does add a small amount
of overhead, since `BAx` is computed alongside `W₀x`.

## What it costs

Quality relative to full fine-tuning depends on configuration, and is not
automatic. Two things dominate: which layers are adapted, and whether the rank is
sufficient for the dataset. Configured well, LoRA matches full fine-tuning on
post-training-scale data at roughly two thirds of the compute. Configured badly —
attention-only, or rank too low for the data — it measurably underperforms.

[docs/lora-practice.md](lora-practice.md) covers the configuration that
determines which of those you get.

Adapters are also architecture-specific. They are tied to the module names and
shapes of the base model they were trained against, so a Llama adapter cannot be
applied to a Phi-3 model. Module naming differences between families are a common
source of silent misconfiguration.

## Variants

| Variant | Idea |
|---|---|
| **DoRA** | Decomposes the update into magnitude and direction, learning magnitude separately. Often closes part of the remaining gap to full fine-tuning; slower per step. |
| **QLoRA** | Adapters over a quantised (typically 4-bit) frozen base. Trades some base-model fidelity for a large memory reduction. |
| **rsLoRA** | Scales by `alpha / sqrt(rank)` rather than `alpha / rank`, so the update magnitude stays stable as rank changes. |

This repository implements DoRA and QLoRA as first-class methods and rsLoRA as an
adapter option. PEFT ships considerably more (LoHa, LoKr, OFT, BOFT, VeRA,
VBLoRA, FourierFT and others, plus initialisation schemes such as PiSSA, OLoRA and
EVA); see the [PEFT documentation](https://huggingface.co/docs/peft/) if you want
to go further than the three above.

## Sources

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) (Liu et al., 2024)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
- [PEFT documentation](https://huggingface.co/docs/peft/)

Content from external sources is summarised and paraphrased rather than quoted.
