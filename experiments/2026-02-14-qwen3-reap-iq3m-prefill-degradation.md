# Qwen3-Coder-Next-REAP-40B-A3B i1-IQ3_M: Prefill Degradation Test

**Date:** 2026-02-14  
**Model:** `Qwen3-Coder-Next-REAP-40B-A3B.i1-IQ3_M.gguf` (18.2 GB)  
**Hardware:** 2× RTX 3060 12GB (tensor-split 1,1)  
**Cache:** `--cache-type-k q8_0 --cache-type-v q4_0`

## Goal

Test whether token generation speed degrades as the KV cache fills up.

**Hypothesis:** Speed should remain flat due to:
1. Hybrid architecture (only 12/48 layers use attention)
2. Flash Attention enabled
3. Efficient KV cache quantization (q8/q4)

## Test Methodology

For each context size (64k, 128k, 196k, 262k):

1. **Baseline (empty cache):** Generate 128 tokens from small prompt → measure TG speed
2. **90% filled:** Prefill to ~90% of context, generate 128 tokens → measure TG speed
3. **~98% filled:** Continue generation to near-limit → measure TG speed

**Comparison metric:** TG tok/s at different fill levels.

## Results

### 64k Context

#### Baseline (empty cache)
- **From 2026-02-14 morning bench:** 22.4 tok/s

#### 90% filled (~58k tokens prefilled)
**Status:** Running...

| Metric | Value |
|--------|-------|
| Prompt tokens | TBD |
| PP speed | TBD |
| TG speed (at 90% fill) | TBD |
| Generated tokens | 128 |

#### ~98% filled
**Status:** Pending

---

### 128k Context

**Status:** Pending

---

### 196k Context

**Status:** Pending

---

### 262k Context (native max)

**Status:** Pending

---

## Observations

(To be filled as tests complete)

## Conclusion

(TBD after all runs)

---

**Status:** Test 1 (64k @ 90%) in progress (2026-02-14 09:13 UTC)
