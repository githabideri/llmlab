# 2026-11-17 — dual-3060 27B context hierarchy: the RAM tier and disk tier in llama.cpp b12383, measured

**Goal.** Make a 27B conversation model on a consumer dual-GPU box hold several
*long-lived* logical conversations the way a big production stack would: an
active conversation's KV state in GPU VRAM, an idle one parked in a bounded
RAM cache, and a durable snapshot on disk that survives restarts and model
switches. This is the follow-up to the 2026-11-15 max-context campaign on the
same box; the engine, model, and layout are frozen from that report (llama.cpp
b12383 / commit c56be86, Qwen3.8-27B GSQ-RCO IQ3_S with its built-in MTP
draft, CPU projector, the A8 GPU-order-flip layout, 2 slots). The box: a
consumer desktop, Intel i5-7400 (4C/4T — no hyperthreading), 32 GB DDR4, B250 board, two RTX 3060
12 GB cards (one at x16, one electrically x16-but-pegged-to-x4 by the slot,
no P2P, no NVLink), running the server directly on the Proxmox host.

**Method.** Three scratch windows inside watchdog-protected maintenance
periods (production stopped, a timer that force-restores it if the run dies,
GPU-release gating before every boot — a boot started while a dying
predecessor still holds VRAM aborts at the first allocation, and this bit us
three times before we instrumented it). Every cell uses the same planted-code
fixture the 2026-11-14/15 campaigns validated: a corpus of short sentences
with a one-line "the stone door was marked with the code Kxxxx and sealed in
the vault" at ~90% depth, a request to write a 350-word story ending with
that code, and a follow-up that must answer with the code alone (temperature
0, 128-token cap — the model is a thinking model, and the reasoning channel
eats the token budget: with an 800-token cap the story never even starts, so
prims run at 8192). Success means the follow-up, after the state was evicted
from the slot, still returns the exact code fast; the server's own logs and
`/slots` telemetry say whether that came from the pool, the RAM cache, or a
cold re-prefill.

## The RAM tier: what b12383 actually restores

Flags: `--kv-unified --kv-unified-per-slot 262144 --cache-prompt
--cache-reuse 4096 --cache-idle-slots --cache-ram 4096|8192`, MTP on. When a
slot goes idle, the server saves its prompt state into a bounded RAM cache
(LRU); when a new request arrives for that slot, it tries to load it back.

**Saved state sizes (measured from the server's own accounting, including the
MTP draft-state component):**

| prompt length | total state | draft part |
|---|---|---|
| 25,182 tok | 691.8 MiB | 98.9 MiB |
| 87,317 tok | 2,029.6 MiB | 343.1 MiB |
| 171,185 tok | 3,835.3 MiB | 672.6 MiB |
| 182,010 tok | 4,068.3 MiB | 715.1 MiB |

That is ~22–25 MiB per 1K tokens (q4_0 K/V plus the f16 draft KV plus
checkpoints), so the default 8192 MiB cache holds roughly two 180K-class
states, and the 4096 MiB budget used for the primary run holds about one and
a half.

**What restores, and what doesn't (the headline):**

| cell (A prim / B competitor / A follow-up) | 4096 MiB budget | 8192 MiB budget |
|---|---|---|
| 16K-class (21–25K tok) | **RESTORED**: 2.59 s TTFT, 99.8% prefix cached, exact code | (same path) |
| 64K-class (83–87K tok) | COLD: 194.35 s; load of the 1.77 GB state **failed** | — |
| 128K-class (167–171K tok) | COLD: 524.79 s, 0% cached | COLD: 529.46 s; load of the 3.10 GB state **failed** |
| 180K-class (178–182K tok) + 119K competitor | COLD: 580.51 s, 0% cached | COLD: 580.51 s (identical to the second); load of the 3.28 GB state **failed** |

The failure signature is uniform and unambiguous: the state *saves* fine
(`prompt_save: saving prompt with length N, total state size = X MiB`),
sometimes after an LRU eviction of an older entry, but the follow-up's load
dies with `state_seq_set_data: error loading state: failed to restore kv
cache` → `srv load: failed to restore state with size <bytes>` → `slot
prompt_load: failed to load prompt from cache`, and the slot then re-prefills
from scratch (the 128K/180K follow-ups are reproducible to the second across
budgets: 524.79/529.46 and 580.51/580.51 s). Only the small state restores —
and it does so through a different mechanism, a ~150 MiB *context
checkpoint* (`restored context checkpoint … size = 149.646 MiB`), not the
full KV-sequence load.

**Doubling the budget changed nothing.** Both 128K-class states (3.8–4.1 GB
saved) fit inside 8192 MiB with room to spare, and both still failed to load;
the 4096 MiB part shows the same verdict. This is a **load-path failure, not
a capacity failure**: in b12383 the RAM tier effectively works only for
small prefixes (the checkpoint class, ≲150 MiB ≲ ~30K-token states), and the
full-state restore that the long-conversation use case needs is broken for
states in the ~1.8–3.3 GB range we tested.

**Contamination check (upstream issue #27148 shape).** Two simultaneous
16K-class sessions with different planted codes, interleaved follow-ups: no
cross-session leakage (each follow-up saw only its own code; the other
session's code appeared nowhere), and both follow-ups restored in 2.3–2.4 s.
The one thing that *can* go wrong in this design — a new request picking up
a foreign conversation's restored state — did not occur in this build under
these conditions.

**Why the 2026-11-15 "182K follow-up in 1.6 s" still stands.** That fast
follow-up was a *pool-resident* state: the unified KV pool had never evicted
the idle conversation, so no RAM load was needed at all. The two mechanisms
look identical to the client (a fast cached follow-up) but have different
failure modes: pool-resident states survive while nothing bigger competes
for the pool; once a competitor evicts them, the only path back is the RAM
load that this run shows to be broken for big states. The practical rule for
this build: **one big idle conversation per pool is fast; several are not.**

## The disk tier: `--slot-save-path` and the `/slots/{id}` endpoint

With `--slot-save-path <dir>`, each slot's state can be snapshotted to a
file and restored later; the HTTP surface is `POST /slots/{id}?action=save |
restore | erase` with a `{"filename": "..."}` body (note the slot in the
*path* — the query-string form 404s, which is how we lost a window to it),
and `GET /slots` for the per-slot state. Restore reports what it did:
`{"n_restored": 25219, "n_read": 1197105060, "timings": {"restore_ms":
964}}` — a 25K-token state is a 1.12 GB file and comes back in under a
second (~1.2 GB/s off the box's root disk). `erase` clears the slot's state
(`{"n_erased": 25219}`) but leaves the file on disk — snapshot lifecycle
(deletion, checksums, budget) is the operator's job, not the server's.

**What the disk tier did (measured):**

| test | result |
|---|---|
| 16K-class round trip: save → a different prompt overwrites the slot → restore → follow-up | **PASS**: 1.12 GB snapshot, restore in 0.9 s, follow-up 4.8 s with 99.9% of the prefix reported cached and the planted code retrieved exactly |
| 128K-class save | **PASS**: 4.19 GB file, written in a few seconds off the box's root disk |
| restore of the 128K snapshot (same boot and after a full server restart) | **PASS**: `n_restored = 171,215` both times (9.9 s / 4.5 s); after the restart the follow-up came back in 8.2 s with 167,116/167,164 tokens cached and the exact code — the conversation state **survives a server restart** |
| 27B → 35B → 27B model switch on the same cards (the 35B took the GPU, served a request, was killed; the 27B was restarted and the snapshot restored) | **restore PASS** (200, 171,215 tokens, 11.3 s); the follow-up on that boot is the one case that never returned a verifiable answer — the request wedged (see below) |
| file sizes for the storage budget | 25K-token state ≈ 1.12 GB; 171K-token state ≈ 4.19 GB (KV q8_0/q4_0 plus draft state plus checkpoints). A few snapshots fit on any disk; the binding constraints are restore time (~1 s per GB) and disk I/O, not capacity |
| corrupt-file handling (truncated / missing snapshot) | not exercised in this run (the window closed before that cell); the server validates filenames on save and the restore call returns a structured error object, but we did not measure the truncated-file path |

> **SUPERSEDED (2026-11-18).** The incident reconstruction re-dated this event:
it was not an eight-hour generation hang. The window's 4-hour watchdog had
fired and killed the scratch server; a manual follow-up sent against the dead
endpoint blocked for ~8.1 hours against a 1440 s read timeout, and that
block is what the session experienced. The server was never hung. The
"MTP draft-decode hang on a restored state" hypothesis is **retracted**.

Original text preserved for the record:

**A wedged generation, found by accident.** The follow-up sent immediately
after the post-switch restore never finished: the slot reported
`is_processing: true` with 152,576 of 152,618 prompt tokens processed for
eight hours, then the task finally "finished" with no output. The server
process stayed up and responsive to `/slots` the whole time. This is a
generation-hang in the MTP spec-decode path on a restored 171K-token state
(the same state restored and answered correctly twice in other boots), i.e.
intermittent, state-dependent, and invisible to any health probe that checks
the port. For a production hierarchy this is the failure that matters: a
restored conversation can hang the slot silently. We have the server log
(verbosity 5) of the incident; it is a candidate upstream issue alongside the
RAM load-path failure.

## What this means for running the 27B fallback card

1. **Small-prefix reuse is free and works.** The defaults (cache-prompt on,
   8192 MiB cache) already give fast follow-ups for everyday conversations
   up to roughly the 30K-token state class, with LRU bounds and no measured
   contamination.
2. **Several long-lived big conversations are not yet a solved problem in
   this build.** Past ~30–64K tokens, an evicted conversation's state cannot
   be restored from RAM in b12383 (load-path failure, not a tuning knob), so
   the second-and-later big conversations pay a full cold re-prefill
   (~3–10 min for 120–180K tokens at this box's prefill speed) whenever a
   competitor evicts them. The unified-pool configuration is still the right
   shape — it just needs a working big-state restore, which is what the
   newer mainline builds and the open upstream issues are about (see below).
3. **The production default stays the non-unified P1 profile** (the
   2026-11-15 A8 shape: GPU-order flip, 60/40 split, CPU projector,
   256K/2, `-ub 128`, default MTP). The unified P2 candidate remains
   shelved: besides this RAM-restore finding, it still carries the
   concurrent-streaming regression that appeared in the canary (burst-buffered
   second request, ~225 s first byte, one early close) and has not been
   attributed to a single flag.

## Upstream status (as of this build)

- **PR 16391** (idle-slot prompt cache) and **PR 15293** (context
  checkpoints) are in b12383 and are what the numbers above measure.
- **PR 26004** (checkpoint preservation across slot save/restore, merged
  2026-10-08) is in the build; its behavioral effect is covered by the
  round-trip battery.
- **Open:** #30961 (RAM restore failing while an idle slot still holds pool
  cells — our 180K+120K cell is exactly this shape), #27148 (foreign-state
  restore; not reproduced here), #29324 (`--cache-ram -1` is not a real
  unlimited mode — bounded budgets are the only sane ones).
- **New finding worth an issue:** in b12383, RAM-cache states in the
  ~1.8–3.3 GB range save successfully but fail to load
  (`failed to restore kv cache` from `state_seq_set_data`), independent of
  the budget and of slot occupancy; only the ≤~150 MiB checkpoint path
  restores. Repro: unified 262144/slot, MTP on, any prompt ≈ 80K+ tokens,
  let a second prompt evict the slot, ask again.

## Method notes and gotchas

- **Boot race (hit three times):** starting a new server while a dying one
  still holds VRAM aborts at the first `cudaMalloc` (the new process sees
  0.5–1.5 GiB free on a 12 GB card and dies instantly). Every launcher now
  gates on "total VRAM used < 100 MiB" before exec, and the watchdog's
  restore path does the same. The same race explains a production unit
  OOM it suffered when a test process was killed 8 s before the unit started.
- **The thinking model eats token caps.** With `--jinja`, the model reasons
  first; a tight `max_tokens` budget (800, 16) came back with *empty*
  content because the reasoning channel consumed the budget. Prims need
  ≥8K; simple follow-ups fit in 128 (measured: 51 completion tokens for
  "the code", ~30 of them reasoning).
- **Hand-rolled HTTP truncates large bodies.** A 350 KB POST built by string
  concatenation arrived as ~1.2 KB to this server (the 42 KB sibling of the
  same test was intact); urllib/streaming clients are fine. If you log raw
  bytes for stream behavior, log the payloads, not just the lengths — two
  172-byte frames turned out to be error responses.
- **The `503 "Loading model"` body is not "ready."** Readiness polling must
  require HTTP 200, and the model takes ~2–3 min to load cold.
- The liveness watchdog's self re-arm silently failed once because
  `systemd-run --unit=<name>` refuses to re-create a stopped unit of the same
  name; the run continued under direct supervision.

## Data

All raw evidence (server logs with the `prompt_save` / `prompt_load` /
`restored context checkpoint` / `making room for prompt cache entry` lines,
per-cell JSON with usage counters and TTFTs, watchdog logs, the 15.8 MB
verbosity-5 server log of the RAM-tier run) is kept privately in the homelab
campaign directory; the numbers in this report are the measured values from
that run. The two budgets were run as separate boots with identical flags
except `--cache-ram`; cells were run sequentially per boot in the order
16K → 64K → 128K → 180K → contamination probe, 4096 MiB boot first, then
128K/180K repeated at 8192 MiB.

## Addendum 2026-11-18: the 42-token anomaly is resolved (supersedes the "isolated, under investigation" status above)

A 16-cell controlled matrix (frozen deterministic corpus, sha256-verified; one
variable at a time; two engine builds) resolved the restored-vs-live prefix
anomaly. All numbers below are measured.

**The anomaly was three separate mechanisms, not one cache bug:**

1. **Pool-capacity rejection.** Two ~167K-token states cannot coexist in the
   unified pool on a 2×12 GB box (334K tokens of KV on top of the 11.5 GB
   model). The restore into the second slot returns a clean 400
   ("No available space in KV cache") — reproduced with both pool shapes
   tested (262144/slot, the original incident shape, and 200000/slot). The
   original "42 cached" reading was a virgin slot receiving the followup:
   only the residual system-template prefix matched (42 then, 0 in later
   runs; the residue is incidental).
2. **Restore succeeds, but prefix reuse is size-dependent (the real engine
   behavior).** When the pool is freed first (erase the source slot), the
   restore returns 200 and the slot is correct (it answers the frozen
   question verbatim), but at 167K the followup does not get the prefix:
   it re-prefills the full 167K tokens in 618 s with 0 cached, versus 9.9 s
   / 167,116 cached for a live same-slot followup. The verbosity-5 log of
   that request shows why: the restored state **is** a registered prompt-
   cache entry (4,240.755 MiB), but the engine logs "making room for prompt
   cache entry, removing oldest entry" for the followup — the ~4 GB entry
   does not survive the new request's allocation and its chunk key never
   matches, so the prefix is rebuilt from scratch. At the 16K-class size
   the same sequence keeps the entry and reuses the full prefix (the V7
   isolation window, below). Identical in the newer build (which contains
   the "prompt running hash" change), so that change is not the mechanism.
   Correctness survives at all sizes; performance does not, for large
   states.
3. **MTP-off cache loss (large states).** With the MTP speculator
   disabled, a *167K-class live* same-slot followup gets 0 cached and
   re-prefills in ~385 s (n=2); with it enabled the same slot reuses the
   full prefix. At the 16K class, MTP off does **not** lose reuse (V7A_B
   below). The current evidence scopes this to large states; production
   (MTP on) is unaffected.

**Practical consequence for the persistence hierarchy:** the disk tier is
durable and correct (snapshots survive restarts and model switches; a 167K
state saves in ~1–4 s to a ~4 GB file, and restore returns the correct state),
but returning to a restored conversation costs a full re-prefill (~10 min at
167K) until mechanism 2 is understood upstream. On this box the practical
shape is **one large live conversation (the 167K fixture; the demonstrated
ceiling is higher — 182K + 30K under the 11-15 P2 shape) plus N small ones
(2×27K measured, both complete in ~2 min)**; a second large conversation
must wait on disk. On the RAM tier, the precise statement: b12383 *does* have
an internal, automatic prompt cache (idle-slot state is parked in a bounded
RAM store and the server tries to reload it on the next request; `--cache-ram`
is that store's budget, not a checkpoint store) — the 2026-11-16 R window
measured that path failing for large states, and upstream #30961 is about it.
What b12383 lacks is an *externally controlled* RAM snapshot: there is no
`?action=save` equivalent that parks a state in RAM by choice; the file tier
is the only externally controlled one.

**Cell completeness and the "maximum context" caveat (stated once).**
Executed and recorded: A1, A2, A4, B1, B2, B3, C1, D1, D2, E1, E2 (11-17),
V6A2B, V6A2BE (11-18), plus the 11-18 V7 isolation window (below). Executed
but the record lost to the 11-18 disk-full incident: C2 (the cell ran and
its log shows a ~1.6 s 167K-cached follow-up; the JSON record could not be
written, so no number from it is used anywhere). The 167K (171,215-token)
fixture is the campaign's frozen corpus (seed 102): a workload chosen for
reproducibility across boots, **not** a measured server ceiling — the
demonstrated per-slot capacity is the 182,010 + 30,386 prime-plus-followup
of the 11-15 P2 shape in the same 262K pool.

**The V7 isolation window (11-18 morning, 16K-class states, the 11-16
engine shape, verbosity-5 logs) tested the one remaining hypothesis — that
the followup was being routed to the wrong slot — and ruled it out:**

| test | MTP | routing (engine log) | followup after erase+restore |
|---|---|---|---|
| explicit `id_slot` to the restored slot | on | "selected slot by id (1)" | **3.74 s, 21,083/21,117 cached, exact code** |
| explicit `id_slot` to the restored slot | **off** | "selected slot by id (1)" | **12.1 s (34-token prefill), full reuse, exact code** |
| no `id_slot`; one live + one restored identical state | on | "selected slot by LCP similarity, f_sim_best = 0.999" → the **live** twin | 3.68 s, full reuse (of the live twin, not the restore) |

So: explicit slot targeting is honored exactly (the 0-cached result is not
a request landing on a different slot); automatic routing in this build is
**LCP-similarity based** and is demonstrable line-by-line in the log; and
at the 16K class the disk tier is **fast as well as durable** (save ~1 s,
restore ~1 s, return 4–12 s, MTP on or off). The 167K failure is a
size/eviction property of the prompt-cache entry, not a registration or
routing failure. (The 11-18 first boot of that window was invalid by
construction — the prime auto-routed to a different slot than the cell
pinned — and is excluded from all conclusions.)

**Upstream status (2026-11-18).** #30961 (automatic prompt-cache restore
before competing idle slots are cleared) remains the closest open issue but
is a different path (the automatic RAM store, not explicit disk restore);
#27148/#29324 remain open; PR 26004 (KV rebase on cache restore, draft) is
in the build pipeline; the "prompt running hash" change demonstrably does
not alter the behavior. A drafted issue for the size-dependent restore-
reuse symptom is held in the private campaign record pending the owner's
filing decision; it is deliberately not filed as a duplicate of #30961.

**Storage hygiene (11-18).** The 11-18 window filled the 49 GB host root to
100% (two cell records lost; 167K-class snapshots are ~4 GB each). All
campaign snapshots now write to the 173 GB model pool behind a fail-closed
free-space preflight (≥10 GB on `/`, ≥15 GB on the snapshot filesystem),
with per-boot archival; the 11-16 9.5 GB evidence moved off the root
(checksum-verified). A production host's root filesystem at 100% is treated
as a campaign-inflicted incident, not a footnote.

**P2 (unified + ub256 + nmax2) vs P1 (non-unified + ub128) concurrency:**
at 2×27K both complete every request (no starvation, no early close); P2 is
~15–20% slower per request (wall 131 s vs 114 s). The single 225 s
starvation seen in the earlier canary did not reproduce at this size and
remains unattributed. **P1 remains the recommended production default; P2 is
shelved.**
