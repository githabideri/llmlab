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
- Adjust `--tensor-split` to balance GPU usage

## GPU imbalance
- Use `--tensor-split A,B` to move more weights to GPU1
- Measure with `nvidia-smi --query-gpu=memory.used`
