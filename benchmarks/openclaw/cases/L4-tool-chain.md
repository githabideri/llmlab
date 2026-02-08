# L4 — Tool chain + dry-run message

**Goal:** Chain tools and produce a report; verify message tool works (dry-run).

**Tools:** read, exec, write, message

**Input:** `fixtures/tasks.csv`

**Task:**
- Compute sum/avg/min/max for the `value` column (use `exec`).
- Write `artifacts/report.md` with the stats.
- Call `message.send` with `dryRun: true` containing the report summary.

**Pass:** report exists with correct stats; message tool payload valid (dry-run).
