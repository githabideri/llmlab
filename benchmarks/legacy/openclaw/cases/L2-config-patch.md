# L2 — Offline config patch

**Goal:** Patch a *fixture* OpenClaw config and validate JSON.

**Tools:** read, edit/write, exec

**Input:** `fixtures/openclaw-config.sample.json`

**Task:**
- Change `api` to `openai-responses`.
- Increase `contextWindow` to `262144`.
- Write to `artifacts/openclaw-config.patched.json`.
- Validate with `python -m json.tool`.

**Pass:** patched file is valid JSON and contains the changes.
