# Troubleshooting

## Flash‑Attn disabled
- Symptom: log shows “Flash Attention was auto, set to disabled”
- Fix: rebuild llama.cpp with **FA all‑quants** support

## Unexpectedly slow gen
- Check `--n-cpu-moe` (too high → CPU bottleneck)
- Ensure `--cache-type-v q4_0` + FA enabled
- Avoid tiny batch/ubatch if benchmarking

## OOM on load
- Reduce `--ctx-size`, `-b/-ub`, or increase `--n-cpu-moe`
- Adjust `--tensor-split` / split mode to balance GPU usage

### Qwen3.5 + mmproj on dual 3060 (24GB)
If Qwen3.5 vision profile OOMs, use the known-good memory profile:
- `--ctx-size 98304`
- `--split-mode layer`
- `--parallel 1`
- `--no-mmproj-offload`
- `--cache-type-k q8_0 --cache-type-v q4_0`

This is the combination that kept both 12GB cards below the cliff in retest.

## GPU imbalance
- Use `--tensor-split A,B` to move more weights to GPU1
- Measure with `nvidia-smi --query-gpu=memory.used`
