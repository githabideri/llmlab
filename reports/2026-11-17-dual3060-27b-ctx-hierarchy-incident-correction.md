# 2026-11-17 — correction: the dual-3060 27B "8-hour wedged generation" was a misread; the real eight hours had two causes

**Scope.** Dated correction to [`2026-11-17-dual3060-27b-context-hierarchy-ram-disk`](2026-11-17-dual3060-27b-context-hierarchy-ram-disk.md) (commit `b25bde1`). That report's paragraph "A wedged generation, found by accident" attributed an eight-hour stall to an MTP spec-decode deadlock on a restored 171K-token state. That attribution was wrong. It was a misread of the log's timestamp format, layered on top of two real, separately-attributable events. The RAM-tier and disk-tier *measurements* in that report stand unchanged; only the incident attribution is corrected.

## What the log format actually says

llama.cpp server logs stamp each line with process uptime as **M.S.mmm.mmm** (minutes.second.milliseconds.microseconds), not hours.minutes.seconds. The "8.16.173" lines in the report were **8 minutes 16 seconds**, written by a 27B process that had been alive for nine minutes. The full sequence, reconstructed from the verbosity-5 log, the snapshot directory, and the per-cell JSON:

| event | measured |
|---|---|
| a 35B model that had been holding both cards (8 h 42 min, see below) is killed by PID | VRAM drains 11,696 → 5 MiB in ~3 s |
| 27B scratch relaunched (same launch script as before) | ready in ~2 s (NVMe load) |
| 128K-class disk snapshot (4.19 GB) restored into the slot | 200 OK, `n_restored = 171,215`, 11.3 s |
| follow-up request (a *regenerated* 183K-unit corpus + question) | **completed in 531 s** with `prompt_tokens = 167,169`, `cached_tokens = 42`, and the planted code **retrieved exactly** |
| the client that had sent it had already been killed by its 400 s tool cap | the orphaned process finished the request unobserved and wrote its result JSON 50 s later |
| the scratch server is then SIGTERM'd | clean stop: "all tasks already finished, no need to cancel" |

A 531-second follow-up on a ~167K-token prompt is a **normal cold re-prefill** at this box's measured ~300 tok/s prefill rate — not a deadlock. Two field-semantics facts fall out of the same evidence and are worth keeping: the `/slots` `n_prompt_tokens` counter is an *in-flight* count (so a mid-request snapshot of 152,618 against a final 167,169 is expected), and a small `cached_tokens` after a disk restore means the next request's text did not match the stored text (here: the verification corpus was regenerated with a different seed than the primmed corpus, so only 42 tokens overlapped — by construction, not corruption).

## What the eight hours actually were

The command the operator watched for ~8 h 21 min (a 35B launch attempt: a `nohup` launch inside a container exec call, a readiness loop, and one 24-token probe request) **completed on the remote side in ~20 seconds**. Its output then sat undelivered inside the agent's session daemon, which had stopped processing anything at all — requests, tool results, timeouts, the operator's UI — for **~8 h 8 min** (bounded by the daemon's own request log: last entry before the gap, first entry after). During the freeze nothing inside the estate could intervene: in-process tool timeouts live in the frozen process, and the one external watchdog in the chain had died the previous evening in a `systemd-run` re-arm failure. Two machine-side facts ran in parallel during the freeze and together explain the 13-hour production outage that followed:

1. **The 35B process I had launched held both production GPUs for 8 h 42 min** (it loaded in 55 s, then idled). Every kill attempt in between missed, because the process was launched as `cd build-dir && ./llama-server`, so its command line contained no path a pattern kill could match; it died only when finally killed by PID.
2. **The test-window orchestrator had crashed two hours into the freeze** (a Python bug: `.strip()` on a `CompletedProcess`), spent its remaining 70-minute marker timeout waiting for an event that would never come, correctly refused to relaunch the 27B against the 35B-occupied cards, and gave up — leaving production stopped with no live watchdog left.

The freeze's *in-process* mechanism remains **unattributed** from available evidence: no OOM on host or container, no power event, no guest-agent involvement (the container has none; the container-exec channel returns within seconds), and the daemon process was healthy before and after. The confirmed, reproducible lesson is the structural one: **an in-process timeout is not a backstop, because it shares the failure with whatever it is supposed to bound.**

## What this changes

- **Retracted:** "MTP spec-decode generation hang on a restored state" — no such event occurred; no upstream issue should be filed on that basis. (The 11-17 report's other upstream candidates — the RAM-tier full-state load failure and #30961/#27148 — are unaffected.)
- **Confirmed by the reconstruction:** the disk tier works end to end, including after a full model switch on the same cards — the switch-survival follow-up, once the attribution is fixed, is a *pass* (exact code after a 531 s cold re-prefill), not an unexplained gap.
- **Open anomaly for the controlled matrix:** a disk-restored state gave a 42-token prefix match against a regenerated corpus, while a *live-computed* state of the same conversation on a different process boot gave a ~167K-token match against the same corpus — same file, same corpus function, opposite outcomes. This is the "restored vs live-computed prefix match" cell of the planned isolation matrix, and it is the one question about the disk tier that remains genuinely unresolved.
- **Harness rules adopted** (they cost the 13-hour outage): two independent deadlines per long command (in-command cap plus an external timer under a unique unit name, because the in-process one froze with the process); container exec calls that background long-lived processes must redirect the child's stdin and be killable by captured PID, never by cmdline pattern; readiness means HTTP 200 *plus* a bounded real inference with a thinking-model-aware token budget (a 24-token probe on a reasoning model returns an empty answer and is not a valid liveness test); one orchestrator per machine per window, no interactive commands racing it for the same GPUs.
