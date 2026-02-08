# L0 — Read/Write sanity

**Goal:** Ensure the agent can read a file and write a small JSON artifact.

**Tools:** read, write

**Input:** `fixtures/ping.txt`

**Task:**
- Read `fixtures/ping.txt`.
- Write `artifacts/ping.json` with:
  ```json
  {"ok": true, "value": "ping"}
  ```

**Pass:** artifact exists and JSON is valid.
