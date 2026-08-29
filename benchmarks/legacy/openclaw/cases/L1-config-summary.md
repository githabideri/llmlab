# L1 — Config summary

**Goal:** Parse a sanitized OpenClaw config and summarize providers/models.

**Tools:** read, exec, write

**Input:** `fixtures/openclaw-config.sample.json`

**Task:**
- Extract provider name(s), model id(s), and contextWindow.
- Write `artifacts/providers.md` with a small table.

**Pass:** table includes llama-cpp → qwen3-q2 → 131072.
