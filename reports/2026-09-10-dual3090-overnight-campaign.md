# Dual-3090 Overnight Campaign — 27B Profile Attribution, the 125B Wall, and a Host That Died Quietly

**Models:** Qwen3.8-27B (W4A16-AutoRound, vLLM 0.28.0 TP2) · Qwen3.8-Flash-Next 125B MoE (UD-Q2_K_XL, llama.cpp)
**Hardware:** 2× RTX 3090 24 GB, both PCIe 4.0 x8, no NVLink, 250 W/card power limit (GPU server, Ryzen 5 5600X)
**Date:** 2026-09-10 (overnight window, ~5 h)

> **Correction (2026-09-10, same day):** the soak and ladder sections conflated *end-to-end TTFT* with *engine prefill throughput* — the 883–921 s figure is a 43K-token request's admission-to-first-token latency **under a deliberately saturated 640K workload** (queueing + scheduler contention + preemption + prefill combined, ≈ 47–49 tokens/s end-to-end), not a prefill rate; the ladder rows imply 1.21K / 0.86K tokens/s from request-TTFT, which likewise is not engine prefill throughput (no engine-side pp/s was captured this run). The matrix's **8-conc column is invalid** (a concurrency-client bug made per-stream values ≈ the single-stream rate — physically implausible; the repaired harness now proves overlap and forbids reusing those numbers). The MTP-acceptance and Flash-wall wording were tightened. No benchmark data was changed; interpretation was.
**How it ran:** a *cloud* model in an isolated container (DeepSeek via OpenRouter, no local-model dependency) babysat a single `nohup` runner script that owned every phase; an independent watchdog process held a hard deadline; a JSONL manifest made the whole run resumable. The orchestration brain was deliberately decoupled from the fleet under test — the campaign stops and restarts the very endpoint a local-model brain would be served by.

## Goals

1. **Attribute the 27B production profile.** After the 09-08 cutover to vLLM TP2, decode was ~52 t/s at 16K single-stream in production but ~150 t/s in short-context tests. Was that context-dependent degradation, a config error, or both? Decompose: {CUDA-graph mode} × {MTP on/off} × {scheduler token budget 2048 vs 8192}.
2. **Re-test Flash-Next 125B on the new topology.** The 09-02 campaign measured 30.2 t/s on a *mixed* 3090+3060+3060 box with the fitter's expert spill. Could the homogeneous 2×3090 pair do better (GPU-resident weights, host-backed PLE)?
3. **Stress the restored production endpoint** — the persistence questions (640K staged-admission soak, deep-context ladders, concurrency) that only run against the real unit.

## The 27B matrix (throwaway vLLM instance, port :18020, both GPUs)

All cells canary-validated (deterministic fixed-seed outputs compared across configurations; a mismatched cell is discarded, not averaged). Single stream, 256 tokens (mc8 = 8 concurrent × 128).

| Cell | MTP | Graphs | Batched tokens | decode @2K | decode @16K | 8-conc @2K | accept |
|---|---|---|---|---|---|---|---|
| **P0 baseline** | k=3 | piecewise | 2048 | **151.4** | 135.0 | 154.3 | 1.41 |
| P1-A | k=3 | piecewise | 2048 | 138.6 | 142.5 | 139.3 | 1.41 |
| P1-B | off | piecewise | 2048 | **71.5** | 68.3 | 71.2 | — |
| P1-C | k=3 | **eager** | 2048 | **44.6** | 46.2 | 45.6 | 1.41 |
| **P1-D** | k=3 | piecewise | **8192** | 147.5 | **150.0** | 133.9† | 1.41–1.45 |

*(t/s per stream. P0 and P1-A are the same configuration; the 151 vs 139 spread at 2K is run-to-run variance, which is why the 16K column — less noisy — is the better discriminator. †The 8-conc column is **invalid** — a client bug made per-stream values track the single-stream rate, which a true 8-way batch cannot produce; the trustworthy concurrent picture is the production battery below. Do not reuse those numbers.)*

**Attribution:**

- **MTP (speculative decoding) is the lever: ≈ 2.1×** (151 vs 71.5 t/s @2K). MTP acceptance did not degrade between the tested short and 16K cells (1.41 → 1.45 accepted tokens/verify — a small change, not a demonstrated trend).
- **`--enforce-eager` costs ≈ 3.4×** (44.6 vs 138–151). Piecewise CUDA graphs are non-negotiable on this model — eager mode is a different animal entirely (no graph reuse across the SSM/attention mix).
- **Scheduler token budget 2048 vs 8192:** neutral at 2K (151 vs 147), **8192 wins at 16K** (150 vs 135–142). The 2048 budget was a latency bias left over from tuning; at agent-realistic context depths the bigger batched budget feeds the GPU better.
- **The production mystery is solved:** it was context-dependent degradation all along, not a defect — 151 t/s at 2K, ~135–150 at 16K, and the 09-08-era "52 t/s at 16K" was the *pre-cutover* config family. Like-for-like at 16K, the new TP2 box beats the old dual-3060 numbers.
- **Promotion:** the production unit was switched to the P1-D profile (`--max-num-batched-tokens 8192`; everything else unchanged) with the old script backed up. It has served the whole fleet since, verified by live completion.

## Production battery (the restored :8082 endpoint)

**Concurrency** (128 completions per request):

| Cell | Aggregate t/s | Max TTFT | Note |
|---|---|---|---|
| 4 × 2K | 63.9 | 7.0 s | per-stream 20.7–128.7 — admission stagger makes early requests pay for the prefill pile |
| 8 × 2K | 69.5 | 13.4 s | |
| 12 × 2K | 70.0 | 20.3 s | aggregate saturates ~12 |
| 16 × 2K | 69.7 | 27.4 s | flat — queueing, not throughput |
| 4 × 16K | 8.4 | 59.7 s | one stream rode a clean window (137.5 t/s), the others paid ~14 s prefills |
| 8 × 16K | 8.0 | 126.8 s | prefill serialization dominates |
| 4 × 32K | 3.5 | 144.1 s | extreme corner, expected |

**640K staged-admission soak (30 min).** Two parent streams at 43K-token prompts with child streams admitted mid-way and late, plus a single image request, then parents resumed. Result: both parents completed; **7 preemptions** total in the 640K-token pool. Under this deliberately saturated workload, a 43K-token parent request observed **883–921 s end-to-end TTFT** — ≈ 47–49 tokens/s admission-to-first-token, i.e. queueing + scheduler contention + preemption + prefill combined, **not** raw prefill throughput (engine-side pp/s was not captured this run; the three quantities are kept separate from here on). What the soak did establish: at 262K-max contexts, *time-to-first-token is the user-visible cost*, not throughput — that number, more than any decode rate, is what decides whether 262K is the right production context for agent traffic.

**Deep-context ladders** (single stream, 256 decode tokens):

| Target | Actual prompt tokens | TTFT | Decode |
|---|---|---|---|
| 64K | 87,409 | 72.5 s | 134.2 t/s |
| 128K | 174,916 | 204.5 s | 108.4 t/s |
| 256K | — | **no data** | **no data** |

From request-TTFT the implied end-to-end rates are **1.21K** (87K) and **0.86K** (175K) tokens/s — request-level figures that include admission delay, **not** engine prefill throughput (which was not captured this run). Decode degrades 134→108 t/s from 87K to 175K tokens (the KV-residency cliff starting to show). **The 256K cell is a genuine miss, documented as such:** the campaign's completion check accepted the cell on exit code alone, and the request died instantly (1 s "complete"). A false-complete — the run did not happen, and this report does not pretend it did. The hardened package now rejects any "complete" whose recorded token counts don't match the target depth.

## 125B Flash-Next on 2×3090: a wall

The 09-02 campaign's 30.2 t/s came from a *mixed* 3090+3060+3060 topology where the fitter spilled the 51B-param PLE table to the 3060s' RAM. On the homogeneous pair, **the model does not load, full stop:**

- The fitter's per-layer aggregate buffer is **25.75 GB** — larger than a 24.37 GB 3090's usable frame.
- `--split-mode row`: "CUDA0 does not support split buffers" (arch limitation in this build); `--split-mode tensor`: "not implemented for qwen4exp"; `--split-mode layer`: the 25.75 GB OOM above.
- Manual expert-spill overrides (PLE + per-block experts to CPU) fail on the same layer buffer — the blocker is upstream of the PLE.
- Verified on **two independent builds** (the 09-02 vintages and current master) and all split modes.

So the 30.2 t/s result is not reproducible on this box with the current build family — it is a genuine hardware/build constraint, not a tuning failure. The 35B interim model stays on the secondary box until a build lands that can split this architecture. (A Q2_K_XL 125B model needs ~46 GB of non-PLE weights: two 24 GB cards can hold it *if* the tensors split; this build simply doesn't.)

## What broke (the honest part)

**The script layer.** The campaign package shipped with a dozen latent one-line-class bugs, most of which the cloud orchestrator diagnosed and patched *live, in the dark*, with every change logged and backed up:

1. Wrong venv path (pointed at a scripts directory) — vLLM never booted for the first hour, six P0 attempts lost.
2. Multi-layer shell quoting ate the single-quoted JSON arguments (`--speculative-config`, mm kwargs) through `runuser → setsid → bash -c` — the server started with `{image:{count:1}}` instead of the JSON. Fixed by writing the exact command to a file, pushing it into the container, and executing that (byte-exact, inspectable — better than the base64-encoding fix the agent first applied).
3. Health probes grepped the *body* for `"status":"ok"` — vLLM's `/health` returns an **empty 200**. Every probe in three scripts was a silent false-negative.
4. A case-sensitive GPU bus-ID match (`2d:` vs `2D:``) made the maintenance-window entry die *after stopping production, before arming the watchdog* — the scarier version of the 09-08 incident, produced by our own script. Fixed with case-insensitive matching plus a restore-on-failure path (an `enter` that fails now re-starts production instead of leaving it down).
5. The llama.cpp launcher: `echo "$array"` prints element 0 only; `$p` tripped `set -u`; a host-path pidfile written inside the container.
6. The SSE benchmark client didn't request `stream_options.include_usage` (null decode rates) and crashed on the trailing usage-only chunk.
7. **The canary had a false positive:** a 2 KB swap-in threshold sat below the machine's noise floor and invalidated the *winning* cell (P1-D) on a zero-swap sample. Fixed to 32 MB/sample. Strictness in the wrong place is worse than none.
8. **A false complete:** the 256K ladder above — exit-code-only acceptance.
8. **The 8-conc matrix cell was not actually concurrent** — per-stream decode ≈ the single-stream rate, which a true 8-way batch cannot produce; the client neither proved overlap nor recorded per-request wall windows (invalid column, repaired client now requires timestamp-proven overlap).
9. No exactly-one-runner lock (a stale-lease relaunch produced three concurrent runners); non-idempotent preflight (3–10 min per relaunch × nine relaunches); a missing test-image fixture for the vision cell.

The pattern: **dry-runs test control flow, not data-plane assumptions** (body shapes, quoting depths, path ownership, threshold noise floors). Everything is fixed and re-verified in the hardened package; the next run should start clean.

**The host layer (the big one).** At ~03:17 UTC — mid-way through the maintenance window's production restore, with the new vLLM instance allocating its 23.4 GB KV pool on both GPUs — **the host hard-crashed**. The box runs a 10-second telemetry sampler (nvidia-smi + dmesg tail + meminfo) for exactly this reason, and its last heartbeat ten seconds before death shows a machine that was *healthy*: both GPUs idle (P8), 49/64 °C, 22–42 W, no memory pressure (13 GB used of 62), load 7. It died doing nothing.

The evidence after: the journal ends mid-sentence with no shutdown sequence and no kernel panic; no MCE/EDAC/GPU-Xid/hung-task/I-O errors anywhere in the 7 h 46 min boot; SMART passed on both data disks; the ISP modem and the lab's second server (same network, different circuit) both survived — no regional power event; no kdump is installed, so a panic would have left nothing anyway. The ~4-minute gap from death to reboot is longer than a bare power reset. **Prime suspect: the power path (PSU/cable) or an unlogged motherboard fault.** This was the *second* unexplained reboot of this machine in two days (the 09-08 one also had no root cause; it destroyed an 8-hour window and a 92 GB model load). Both crashes sit adjacent to the heaviest mixed VRAM + host-RAM workloads the box has seen — correlation, not proof, but it's the working hypothesis.

**And the recovery did what it was designed to do.** The lease and watchdog lived on tmpfs and died with the box; the manifest lived on the USB SSD and survived; the production systemd unit auto-started on boot; the orchestrator re-entered the window, re-armed the watchdog, and resumed the runner, which skipped all finished cells and finished the night; the watchdog fired at its deadline into an already-healthy endpoint (idempotent) and closed the book. Zero data loss, zero hands on the machine. That is the argument for the three-layer design, made real.

## Conclusions

- **Production 27B profile: piecewise CUDA graphs + MTP k=3 + 8192 batched tokens** — 151 t/s @2K, 150 @16K, 134 at 87K, 108 at 175K single-stream; ~70 aggregate t/s at 4–16 concurrent 2K streams; canary-validated. The 262K context is safe (776K-token KV pool, 7 preemptions in a 30-min 640K soak) but under *saturated* staged load a 43K-token request observed 14–15 min end-to-end TTFT (queueing + preemption + prefill — not prefill speed). The open question is what context limit / compaction policy each *agent class* should use — a client-side design question, not a server-capacity one; the server keeps 262K as its ceiling.
- **Flash-Next 125B: wall on this topology** with the current llama.cpp build family. No flag-level tuning campaign can change that within the tested build family; a materially different implementation with working tensor/row splitting could.
- **The host needs physical attention before the next unattended run** (PSU first; kdump second so the next death leaves a corpse to examine).
- **The orchestration architecture passed its first live test**, including the one scenario it was built for — the box dying under it.

## Energy

Not captured this run (no smart-plug sampling in the package yet). Upper bound by construction: 250 W × 2 cards at the power limit plus host idle, for the ~5 h window. The 09-02 report remains the reference for measured Wh/1K-tokens on this model family.

## References

- [2026-09-02 — Qwen4Exp Flash-Next on three consumer GPUs](2026-09-02-qwen4exp-flash-next-three-gpu-campaign.md) (the 30.2 t/s baseline this report's wall section supersedes for the 2×3090 topology)
- [Qwen3.8-27B model card](../models/qwen3.8-27b-rtx3090.md) (production config, updated with the 8192 promotion)
- [GPU server hardware doc](../docs/hardware/gpu-server.md) (reliability notes)
- [2026-08-30 — dual-3060 35B squeeze / 27B node](2026-08-30-dual-3060-35b-squeeze-27b-node.md) (the previous generation's ctx-degradation curve this report's like-for-like numbers beat)
