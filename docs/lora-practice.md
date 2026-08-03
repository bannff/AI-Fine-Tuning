# LoRA in practice

Configuration guidance, and why this repository's defaults are what they are.

An earlier version of this document recommended targeting attention projections
only (`q_proj`, `v_proj`, `o_proj`) and setting `alpha = 2 * rank`. Both are now
understood to be wrong, or at least strictly worse than the alternative, and the
first one is the single most consequential LoRA misconfiguration. The current
guidance is below.

## Apply adapters to every linear layer

Adapt all linear projections, including the MLP (`gate_proj`, `up_proj`,
`down_proj`), not just attention.

Research from Thinking Machines Lab
([Schulman et al., 2025](https://thinkingmachines.ai/blog/lora/)), reproduced in
[TRL's own documentation](https://huggingface.co/docs/trl/main/lora_without_regret),
found that LoRA matches full fine-tuning at around two thirds of the compute when
configured correctly, and that the first requirement is applying it to all weight
matrices. Attention-only adaptation underperforms *even at a higher rank chosen to
match the parameter count*. Rank does not buy back layer coverage.
*(Summarised; see the linked sources for the full results.)*

Concretely, on a Llama-family model at rank 8:

| `target_modules` | Trainable parameters | Adapted |
|---|---|---|
| `["qkv_proj", "o_proj"]` | 2,048 | `o_proj` only |
| `["q_proj", "k_proj", "v_proj", "o_proj"]` | 7,168 | attention |
| `"all-linear"` | 16,384 | attention + MLP |

The first row is worth dwelling on, because it was this repository's previous
hardcoded default. `qkv_proj` is a Phi-3 module name and does not exist on a Llama
model; `o_proj` does. PEFT raises only when *no* target matches, so a partially
valid list does not error. It silently trains one projection at a quarter of the
intended attention capacity and none of the MLP, with no warning and no
traceback.

In this repository: `target_modules: all-linear` is the default. Narrowing it is
allowed and raises an advisory.

## Choose rank from dataset size

Rank sets adapter capacity. Beyond the capacity a dataset needs, more rank buys
little; below it, LoRA falls behind full fine-tuning.

| Task | Dataset scale | Suggested rank |
|---|---|---|
| SFT | post-training scale | 256 |
| SFT | small, task-specific | 8–32 |
| RL | any | 1–32 |

RL needs very little capacity because policy-gradient methods extract on the order
of one bit of information per episode.

The default here is 16, which suits the small task-specific case that a single
machine is realistically training. Raise it for larger datasets.

## Keep alpha independent of rank

The update is scaled by `alpha / rank` (or `alpha / sqrt(rank)` under rsLoRA).
Tying alpha to rank means every rank change silently rescales the update and
forces you to re-tune the learning rate. Hold alpha fixed instead and change rank
freely.

If you plan to sweep rank, enable `use_rslora: true`. With `1/sqrt(rank)` scaling
the update magnitude stays roughly stable, so the optimal learning rate becomes
close to rank-independent. Rank 64 with alpha 32 gives a scale of 0.5 under
standard LoRA and 4.0 under rsLoRA.

## Use a higher learning rate than full fine-tuning

Counterintuitively, low-rank training wants a *larger* rate. Roughly two orders of
magnitude, and it is close to rank-independent because of the `1/rank` scaling.

This repository defaults to `2e-4` for `lora`/`dora`/`qlora` and `2e-6` for
`full`. Running full fine-tuning at a LoRA rate diverges; NaN validation loss is
usually the first symptom.

## Keep the effective batch size modest

LoRA tolerates large batches less well than full fine-tuning, and raising the rank
does not mitigate it. Keep `batch_size * gradient_accumulation_steps` under about
32. This repository raises an advisory above that.

## Mask the prompt

Compute loss on the completion only. Otherwise the model is also optimised to
predict the user's turn, which is not the task.

Both backends support it and it is on by default: `mask_prompt: true` maps to
`--mask-prompt` on MLX and `completion_only_loss=True` on TRL.

Getting this wrong by hand is easy and expensive. The loop this repository
replaced tokenised with `padding="max_length"` and then set `labels = input_ids`
with no `-100` masking, so with a 1024-token limit and short examples, most of
the gradient signal came from predicting `<pad>`.

## Quantisation is not a trainer flag on MLX

QLoRA means adapters over a quantised frozen base. On the PyTorch side
`bitsandbytes` loads a 4-bit base in-process. On MLX quantisation is a separate
`mlx_lm convert -q` step producing a new model directory, which you then train
against.

This matters because passing no quantisation option to `mlx_lm lora` does not give
you QLoRA. It gives you plain LoRA on a full-precision base. The previous
`train_qlora_mlx.py` in this repository did exactly that, and differed from the
LoRA script by filename only. `finetune train` with `method: qlora` runs the
convert stage first; `finetune plan` shows both stages before anything executes.

## Memory, roughly

For a 7B model:

| Mode | Approximate memory |
|---|---|
| Inference, bf16 | ~14 GB |
| LoRA training | ~16 GB |
| Full fine-tuning | ~28 GB and up |
| Merged inference | ~14 GB, no adapter overhead |

Gradient checkpointing trades compute for memory and is worth enabling for full
fine-tuning specifically, where memory scales with the whole model rather than
the adapter.

## Merging

Adapters can be folded into the base weights, which removes any inference-time
overhead:

```python
model = model.merge_and_unload()
model.save_pretrained("./merged-model")
```

On MLX the equivalent is `mlx_lm fuse`. Note that merging is a real operation with
a real output — a directory containing an `adapter_config.json` has *not* been
merged, whatever it is named.

## Sources

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (Schulman et al., Thinking Machines Lab, 2025)
- [TRL: LoRA Without Regret](https://huggingface.co/docs/trl/main/lora_without_regret)
- [PEFT documentation](https://huggingface.co/docs/peft/)
- [mlx-lm LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

Content from external sources is summarised and paraphrased rather than quoted.
