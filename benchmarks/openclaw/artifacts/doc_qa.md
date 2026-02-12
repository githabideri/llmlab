# DOC-QA answers (with citations)

1) **How do you start llama.cpp with the current best config?**
- Use the `llama-server` command block shown under “llama.cpp — start (Qwen3 Q2 current best)”, including key flags like `--ctx-size 131072`, `--tensor-split 14,10`, `--cache-type-k q8_0`, `--cache-type-v q4_0`, and `--flash-attn 1`.  
  Source: `llmlab/docs/runbook.md#L3-L13`

2) **What should you change if you hit OOM on load?**
- Reduce `--ctx-size`, reduce `-b/-ub`, or increase `--n-cpu-moe`; also tune `--tensor-split` to balance GPU usage.  
  Source: `llmlab/docs/troubleshooting.md#L12-L14`

3) **How do you stop the server?**
- Run `pkill -x llama-server`.  
  Source: `llmlab/docs/runbook.md#L24-L27`
