# L3 — llama-bench parse

**Goal:** Parse llama-bench output and compute % drop from 32k → 128k.

**Tools:** read, exec, write

**Input:** `fixtures/llama-bench.sample.md`

**Task:**
- Extract pp32k/64k/96k/128k and tg256.
- Compute drop: `(pp32k - pp128k) / pp32k * 100`.
- Write `artifacts/bench_summary.md` with a small table + drop %.

**Pass:** numbers match the fixture and drop % is correct.
