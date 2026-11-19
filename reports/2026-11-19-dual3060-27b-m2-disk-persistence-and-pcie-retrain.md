# 40K-class disk persistence is fast; the ×1 width tax is quantified; the PCIe retrain fingerprint is directional

**Machine:** dual RTX 3060 12 GB workstation (Intel i5-7400, 32 GB DDR4,
Proxmox VE 9.2 host, nvidia module 595.71.05) — the [Qwen3.8-27B
dual-3060 card](../models/qwen3.8-27b-dual3060.md).
**Engine:** llama.cpp build b12383 (commit c56be86) — the same binary in
production; the d92ea9bb1 comparison build appears in one boot.
**Date / windows (all 2026-11-19, local CET):** M2 window
01:29:31–01:36:28 (4 boots, rc=0); M2-FIX window 02:03:36–02:05:58
(2 boots, rc=0). Production stopped for each window and live-verified
after each; both 3060 links and both sources (nvidia-smi + lspci) were
sampled before and after every cell, with explicit coverage markers.
**Classification:** measured, n=1 per cell per boot.
**Provenance:** `data/provenance/dual3060-27b-2026-11-19-m2-m2fix.json`
(binary and model SHA-256 included).

## What this report adds

1. The first clean **×1-vs-×4 prefill comparison** for the chipset-port
   card (and the first measured ×4 data point for it).
2. The **40K-class disk-persistence round trip, measured end to end and
   correct** — including across a full process restart — in under a
   second per direction. This pins the fast/slow boundary of the disk
   tier to the 1.15–4.19 GB gap between the 40K and 128K classes.
3. The **PCIe retrain fingerprint**, refined from "the driver de-trains
   the links" to a deterministic directionality: **every model-load
   event de-trains both 3060 links to Gen1 (width preserved); every
   unload/stop re-trains them to Gen3; every power-on negotiates
   Gen3 ×16/×4.**

## 1. The width tax (Gen3 ×4 vs Gen3 ×1, same engine class)

Cold prefill on the 40K reproducer fixture (36,979 tokens, planted code):

| Boot (2026-11-19 M2 window) | prompt_ms | tok/s |
|---|---|---|
| b12383, P1 verbatim + `--slot-save-path` | 63 642.6 | 581.0 |
| b12383, `--spec-type none` (MTP off) | 63 600.0 | 581.4 |
| d92ea9bb1 build, otherwise identical | 63 435.2 | 582.9 |

Reference: the 2026-11-18 M3 window (this [report](2026-11-18-dual3060-27b-matched-p1-vs-unified.md))
measured a 47,241-token prefill at **414–416 tok/s** on the same engine
class, on the chipset-port card at the measured **Gen3 ×1** condition.
Both windows ran with the link at Gen3; the only material difference is
the negotiated width of that card (×4 here, ×1 there) plus the fixture
text. **The ×1 width therefore cost roughly 40% of prefill
throughput** (within-condition spread <0.5% in both windows; the
condition gap is an order of magnitude larger than that spread).

Corollaries:

- **Engine delta ≈ 0.** b12383 vs d92ea9bb1 (96 commits ahead, carrying
  the KV-delta fixes) at 40K: 581.0 vs 582.9 tok/s. Consistent with the
  2026-11-18 16K result (identical to the second decimal). d92 stays a
  comparison build.
- **MTP-off has no prefill effect** (as expected: the speculative
  mechanism is decode-stage).
- The 2026-11-15-era "×4 is not a bottleneck" wording is **refuted in
  the prefill direction** by this comparison: ×1 was a large, measured
  bottleneck. (The 0.75 GB/s DMA ceiling of that era was a Gen3 ×1
  number; the 0.3–0.9 GB/s dmon bursts remain unconfirmed as to
  generation.)

## 2. The 40K-class disk tier: fast and correct, end to end

The M2 window's original save/erase/restore cells were **voided by a
harness bug** (the script posted an empty body; this build's `save`
action requires a JSON body `{"filename": "…"}` and answers 400
otherwise; `erase` turns out to clear the slot's registered cache
entries, not its live KV state, so those followups measured warm
continuations, not restores). The M2-FIX window re-ran the central
question with the contract established first by an in-window probe
(`save`/`restore` = POST with JSON filename body; `erase` = no-arg;
`clear`/`reset` do not exist in this build).

The cycle, both paths:

| Step | Live path (one process) | Fresh-process path (virgin) |
|---|---|---|
| 40K prefill | 63.77 s (580.1 tok/s, Gen3) | — |
| save → snapshot file | **1.15 GB** (37,001 tokens; target + MTP draft checkpoints), **0.48 s** (≈2.4 GB/s to the NVMe pool) | — |
| state loss | 40 s idle (exceeds the 30 s park window) | full process restart |
| explicit restore | **0.50 s** | **0.49 s** (≈2.3 GB/s read) |
| followup (planted code) | **cache 36,990 / prompt 4; 0.78 s total; code correct** | **cache 36,990 / prompt 4; 0.78 s total; code correct** |

So a ~40K-token session can be swapped to disk and back — or into a
virgin process, which is exactly the model-switch swap-back scenario —
in **under a second per direction**, and the restored state serves
immediately and answers correctly. The engine log for the fresh path
shows `restored 2 context checkpoint(s)` and the followup's 4-token
prompt processing confirms the reuse is real (4 = the chat-template
wrap of the followup sentence).

**Size boundary for fast persistence** (all sizes on this build, from
this and the preceding windows):

| State class | Snapshot size | Disk-restore behavior |
|---|---|---|
| 16K | 1.12 GB | 0.9 s (2026-11-17) |
| **40K** | **1.15 GB** | **0.48–0.50 s, correct (this report)** |
| 128K | 4.19 GB | silent full re-prefill (531 s, 2026-11-17) |
| 167K | 4.24 GB | evicted after restore; full re-prefill 618 s (2026-11-18) |
| (RAM-tier 1.8–3.3 GB loads) | — | fail outright, `failed to restore kv cache` (2026-11-17) |

The break sits between 1.15 GB (sub-0.5 s) and 4.19 GB (re-prefill); the
exact threshold is unmapped and is the next candidate cell.

**Idle survival (RAM tier):** after 120 s idle (exceeding the 30 s
`--cache-idle-slots` park window), the same 40K state resumed in **0.39 s**
with full prefix reuse — at the 40K class the automatic RAM prompt cache
(default 8 GiB budget) holds the state and the resume is cheap.

## 3. The PCIe retrain fingerprint: load de-trains, unload re-trains

The M3 window (2026-11-18) ended with the mystery of a Gen3→Gen1
downgrade of both links that coincided with a production reload and left
no kernel trace. The 2026-11-19 reboot experiment (owner-authorized;
three cold boots, each with a pre-driver `lspci` capture at sysinit)
showed the hardware negotiates **Gen3 ×16/×4 on every power-on**, with
the links arriving at **Gen1** after the driver's init. The two campaign
windows completed the picture, because each one stops and restarts
production (a full model load/unload cycle) while the driver stays
loaded, and every cell sampled the links:

| Event | Link state after | Observed |
|---|---|---|
| Power-on (before driver) | **Gen3** (×16 / ×4) | 3/3 reboots (pre-driver captures) |
| Driver init / model load (cold boot or runtime start) | **Gen1** (width preserved) | 3/3 cold boots; the 2026-11-18 22:48:50 reload; the 01:36, 01:54 and 02:05 starts of 2026-11-19 |
| Unload / stop (driver stays loaded) | **Gen3** | 2026-11-19 00:40, 01:29:31, 02:03:37 |

The directionality is consistent across every observed event on this
stack (nvidia module 595.71.05 on the Proxmox VE 9.2 / kernel 7.0.14
host): **the driver's load path de-trains the link speed to Gen1; the
unload path and the power-on negotiation both reach Gen3.** Width: the
chipset-port card's ×1 (the original 2026-11-18 mystery) has not
reappeared since the first 2026-11-19 cold boot (×4 ever since; its
device LnkCap is actually ×16 — the ×4 limit is the port's). No pcieport
AER, retrain, or error line accompanies any of these transitions — they
are silent to the kernel.

Consequences:

- The serving condition is **Gen1 after every model load** (¼ of the
  link's bandwidth; decode is compute-bound and unaffected; prefill and
  cold-load bandwidth are reduced — and on the chipset-port card the
  width now at least holds at the port's ×4).
- The cleanest remaining discriminator is `pcie_aspm=off` (one reboot,
  fully reversible): it separates "an ASPM-L1 exit during the load
  de-trains the link" from "the load path actively programs Gen1".
  A different nvidia module version is the second test. The on-site
  physical check (reseat/inspect) stays a cheap secondary: a
  marginal seating is hard to reconcile with clean power-on negotiation
  to Gen3 ×4 on every boot.
- Practical note for anyone operating this kind of box: a **stop/start
  cycle of the inference service re-trains the links back to Gen3**,
  which is why both campaign windows ran at Gen3 despite the box's
  Gen1 serving state — a measurement confound worth knowing about when
  interpreting prefill numbers from maintenance windows on this
  hardware.

## 4. Collateral and honest footnotes

- **Boot-time Tailscale bind race:** on the first reboot the 27B unit
  died one second after start (`couldn't bind … :8123` — the Tailscale
  address not up yet, no retry in the unit). The launcher now waits up
  to 3 minutes for the address before exec (no-op at warm start);
  verified on consecutive cold boots. The companion lifecycle endpoint
  hit the same race (Errno 99, twice) and self-healed via systemd's
  restart policy; it now carries the same wait. Model load measured
  28.6 s with a warm page cache vs 3 m 16 s cold — the load is
  disk-read dominated, not PCIe-DMA dominated, so Gen1 does not dominate
  cold-load time the way the ¼-bandwidth rule of thumb suggests (the
  ×16 card carries 60% of the traffic).
- **The 16G-budget boot was skipped** by its own preflight gate (19 GiB
  free < the 24 GiB requirement with the five-guest set running) —
  recorded, not forced. Re-running it needs a lighter guest set.
- **A harness-state failure (agent-caused) left production down ~10 min**
  (01:56–02:06): a killed window left a stale armed-watchdog marker, the
  next launch's preflight failed *after* the production stop, and the
  independent supervisor correctly stayed silent (by design it acts at
  the watchdog deadline, not on an arbitrary stop; the deadline was
  02:11:44). Manual restore, no data loss. Lesson: the marker file and
  the timer unit must always be cleared together; the supervisor's
  candidate improvement is an early-recovery branch for "marker exists +
  timer dead + production down + no live experiment + no window
  process".
- **Pre-driver captures:** the sysinit capture unit's files never landed
  on disk (a quoting bug in the unit); the journal lines it emitted are
  the surviving record. The unit's "pre-driver" label is slightly loose
  (the module's first load was ~4 s earlier); the readings were taken
  during early driver probing, before the de-training, which is the
  point.
- **n=1.** Every number here is a single run per cell; the within-window
  spreads (three engines in M2; two boots in M3) are <0.5%, which is
  what makes the condition comparisons (×4 vs ×1; 1.15 GB vs 4.19 GB)
  meaningful despite n=1.
