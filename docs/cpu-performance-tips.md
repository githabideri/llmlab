# CPU Inference Performance Tips

**Purpose:** Optimization guide for CPU-only LLM inference

---

## Thread Tuning

### Prompt Processing vs. Generation

Different workloads benefit from different thread counts:

```bash
# For prompt processing (use more threads)
llama-server --model model.gguf --threads 5 --threads-batch 5

# For generation (fewer threads often better)
llama-server --model model.gguf --threads 3

# Auto-tune (let llama.cpp decide)
llama-server --model model.gguf --threads -1
```

### General Guidelines

| CPU Cores | Prompt Threads | Gen Threads | Notes |
|-----------|---------------|-------------|-------|
| 4-6 | 4-5 | 2-3 | Leave 1 core for system |
| 8-12 | 6-8 | 4-6 | Balance between workloads |
| 16+ | 12-16 | 8-12 | Diminishing returns after 16 |

**Rule of thumb:** Generation is more sensitive to thread count; start lower and increase.

---

## Context Size

### Tradeoffs

| Context | Memory Impact | Use Case |
|---------|---------------|----------|
| 4096 | Low | Chat, simple tasks |
| 8192 | Medium | Most use cases |
| 32768 | High | Long documents, deep reasoning |
| 131072+ | Very High | Full book/context retention |

### Commands

```bash
# Default (from model)
--ctx-size 0

# 8K context (recommended for most use cases)
--ctx-size 8192

# 32K context (requires more RAM)
--ctx-size 32768
```

**Memory note:** KV cache scales with context. At 32K context, a 7B model may need 2-3x more RAM than at 4K.

---

## Memory Mapping

### mmap (Default)

```bash
--mmap
```

**Pros:**
- Faster model load (OS handles paging)
- Lower initial memory footprint
- Can share memory between processes

**Cons:**
- Can cause OOM if system memory is tight
- May swap under memory pressure

### no-mmap

```bash
--no-mmap
```

**Pros:**
- More predictable memory usage
- Better stability under memory pressure

**Cons:**
- Slower model load
- Higher initial memory footprint

**Recommendation:** Use `--mmap` by default; switch to `--no-mmap` only if you get OOM errors.

---

## Quantization Selection

### Memory vs. Quality Tradeoff

| Quant | Size Reduction | Quality Loss | Use Case |
|-------|---------------|--------------|----------|
| Q8_0 | ~50% | Negligible | Maximum quality |
| Q6_K | ~60% | Minimal | High quality |
| Q5_K_M | ~70% | Small | Good balance |
| Q4_K_M | ~80% | Moderate | Default choice |
| IQ3_S | ~85-90% | Noticeable | Memory constrained |
| IQ2_KS | ~90%+ | Significant | Extreme constraints |

### Selection Strategy

1. **Start with Q4_K_M** - Good balance of quality and size
2. **If memory constrained** - Try IQ3_S or IQ2_KS
3. **If quality critical** - Use Q5_K_M or Q6_K
4. **Test your workload** - Some models/tasks are more sensitive than others

---

## Batch Processing

For multiple requests:

```bash
# Process prompts in batches
--batch-size 512

# Limit maximum batch
--max-context 4096
```

**Tip:** Larger batch sizes improve throughput but increase latency per request.

---

## Model Selection for CPU

### Sweet Spots

| RAM Available | Recommended Model Size | Quant |
|---------------|----------------------|-------|
| 8-12 GB | 3-7B | Q4_K_M |
| 16-24 GB | 7-14B | Q4_K_M |
| 32-48 GB | 14-27B | Q4_K_M |
| 64+ GB | 27B+ or MoE | Q4_K_M/Q3_S |

### MoE Considerations

Mixture-of-Experts models (like Qwen3.5-35B-A3B):
- **Total params:** 35B
- **Active params:** ~3B per forward pass
- **RAM needed:** ~20-24GB for Q4_K_M
- **Speed:** Similar to dense 3-4B model

**Verdict:** MoE is excellent for CPU - high quality with low active compute.

---

## Troubleshooting Performance

### Slow Generation

```bash
# 1. Reduce thread count
llama-server --model model.gguf --threads 3

# 2. Check CPU usage
htop  # Look for single-core saturation

# 3. Verify AVX2 is enabled
cat /proc/cpuinfo | grep avx2
```

### High Memory Usage

```bash
# 1. Reduce context size
llama-server --model model.gguf --ctx-size 4096

# 2. Disable mmap
llama-server --model model.gguf --no-mmap

# 3. Use smaller quantization
# Download IQ3_M or IQ4_XS instead of Q4_K_M
```

### OOM Errors

1. **Reduce context size** - KV cache is often the culprit
2. **Use `--no-mmap`** - More predictable memory
3. **Smaller quantization** - IQ3_S, IQ2_KS
4. **Smaller model** - Drop down a size class

---

## Monitoring

### During Inference

```bash
# Check CPU usage
watch -n 1 'cat /proc/*/status 2>/dev/null | grep -A1 "^Name:.*llama" | head -20'

# Check memory
watch -n 1 'free -h'

# Check per-process CPU
top -p $(pgrep llama-server)
```

### Key Metrics

| Metric | Good | Concerning |
|--------|------|------------|
| CPU utilization | 80-95% on active cores | <50% (under-threaded) or 100% all cores (over-threaded) |
| Memory pressure | <80% used | >90% used, swapping |
| Tokens/sec | Model-dependent | <5 tok/s on CPU (check threads) |

---

## Hardware Considerations

### CPU Features That Matter

| Feature | Impact | Notes |
|---------|--------|-------|
| AVX2 | High | Essential for good performance |
| AVX-512 | Medium | Some models benefit |
| FMA | Medium | Fused multiply-add |
| L3 Cache | Medium | Larger = better for big models |
| Memory Bandwidth | High | DDR4 vs DDR5 matters |

### Memory Matters

- **Bandwidth > Frequency:** DDR4-3200 often beats DDR4-2666
- **Dual Channel:** Essential; single channel halves bandwidth
- **Capacity:** Model + KV cache + OS overhead (~4-8GB)

---

*Source: Migrated from locmox-private/llama-cpp-lxc-setup.md (2026-04-11)*
