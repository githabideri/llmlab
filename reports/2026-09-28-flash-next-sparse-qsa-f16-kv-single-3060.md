# 2026-09-28 — Flash-Next on the single 3060: f16 KV is the short-context lever; the sparse-attention path is structurally unreachable

**Goal.** Close out the QSA (sparse-attention) decode rework for Qwen3.8-Flash-Next on the single-3060 backup box (i3-9100, RTX 3060 12 GB): (1) port upstream's gather-based sparse flash-attention (#28770-class) into the local expert-pool build and A/B it against the known-good dense path, (2) measure f16 vs q8_0 KV types at depth (required by the sparse path, beneficial in general?), (3) determine where the remaining decode time at 64K+ actually goes, (4) close the radix-top-k experiment.

**Setup.** Model: Qwen3.8-Flash-Next GSQ-RCO **IQ2_XS** (65 GB GGUF pair; 54 Q2_0 down-projections, gate/up IQ4_XS). Two local llama.cpp builds: the known-good **expert-pool** fork (per-tensor pool sizing + VRAM autoscaler + pool-budget override; the 09-26 balanced-64 config, `-mec 64`) and an **experimental sparse tree** (expert-pool + upstream's current `fattn*` kernel stack + the `n_kv_max` plumbing that feeds the sparse gate). All runs: `-ngl 99 -ncmoe 99 -fit off`, 1 slot, `-sps 0`, `--no-cache-idle-slots`, `-t 4`, pool rail 1 GiB. KV type controlled via `-ctk/-ctv` (default f16; q8_0 for baseline). Prompts: generated word-salad with three canary sentences at fixed offsets; 32K/64K/90K actual depths from a 96K context window.

**Commands** (representative):

```bash
llama-server -m <IQ2_XS gguf> --port 8089 -t 4 -ngl 99 -ncmoe 99 \
  -c 65536 --lazy-mode on -fit off -mec 64 -np 1 -sps 0 --no-cache-idle-slots \
  [-ctk q8_0 -ctv q8_0]     # f16 is the default; add these for the q8 baseline
curl -X POST http://127.0.0.1:8089/completion -H Content-Type:application/json -d @gen.json.req
```

**Observations.**

1. **f16 KV helps at short depth, not at deep.** 16K: **19.2–19.3 t/s decode** (f16) vs 15.1 (q8_0) — **+28%**, reproduced identically on the known-good binary (so no experimental code is needed to get it). 64K: 10.7 t/s (f16) vs 11.0 (q8_0) — no gain. The f16-vs-q8_0 gap that earlier A100 data showed as 24–30% at depth did **not** reproduce on this 3060 at 64K.
2. **f16 KV works at 64K but OOMs at 96K on 12 GB; the exact ceiling between those points is unmapped.** With `-c 98304`, the f16 KV cache for the 12 QSA layers (the other 36 layers are Gated DeltaNet recurrent layers with no KV cache) at 24:2 GQA, 512-dim heads pushes the load past the 12 GB ceiling and the cuda-graph pass OOMs at startup. q8_0 is the only KV type that reaches 90K+ here.
3. **The sparse-attention path never activates for this model — for structural reasons, not tuning.** Upstream's sparse MMA kernel exists only for head configs (DKQ,DV) ∈ {(512,512), (576,512), (256,256)} with a `K % 256 == 0` length gate, and the dispatch only evaluates the sparse condition under `head ≤ 256`. This model's QSA layers have **Q head = 512, K/V head = 256, GQA 24:2 (ratio 12)** — the (512,256) pair is not in the kernel whitelist, and with a 512-dim Q head the dispatch skips the sparse branch entirely. One-shot instrumentation in the gate function logged **zero invocations** across 16K/64K runs. Also caught while verifying: the gate's `n_kv_max` input must be the **gather width** (`indexer_top_k + compress_ratio − 1` = 2051 for this model), not the full KV length — with the full length the `n_kv ≥ 2·n_kv_max` condition can never pass.
4. **Radix-based top-k (#29326): closed.** A/B at 16/32/64/90K: all deltas ≤ 5% in both PP and decode (16K 77.3→76.5 PP, 15.1→14.8 TG; 90K 77.2→78.7 PP, 8.4→8.2 TG). No material benefit on this hardware.
5. **Pool health at depth is not the problem.** Hit rate drops only 86.4% → 82.0% from 16K to 90K (64-slot balanced pool) — far too small to explain the 15.1 → 8.4 t/s decode drop. The primary identified candidate for the remaining per-token O(n_kv) cost is the **QSA indexer** (a lightning-indexer that scores all n_kv positions every token); dense full-KV attention and other depth-dependent work remain possible contributors (direct per-component profiling is the follow-up).
6. Quality: all canary retrievals 3/3 at every depth on every build; no NaN/corruption in any output.

**Metrics** (IQ2_XS, expert-pool config, 1 slot; n=1 per cell):

| Depth (actual) | KV | PP (t/s) | Decode (t/s) | Notes |
|---:|---|---:|---:|---|
| 16K (15,975) | q8_0 | 77.3 | 15.1 | baseline |
| 16K | **f16** | **83.4** | **19.2** | **+28%** decode; also +19.3 on the experimental binary |
| 32K (31,955) | q8_0 | 86.6 | 14.0 | |
| 64K (63,881) | q8_0 | 82.0 | 11.0 | |
| 64K | f16 | 80.1 | 10.7 | no gain (experimental binary: 12.6, n=1, unconfirmed) |
| 90K (89,833) | q8_0 | 77.2 | 8.4 | f16 cannot load at this ctx on 12 GB |
| 96K ctx load | f16 | — | — | OOM at startup (KV ~9.6 GB) |

Energy: per-phase wall power was not instrumented on today's A/B runs (the controls ran without the GPU sampler); the 09-25 campaign measured **~55 W GPU** in decode for this quant/server class, 10.2 GB VRAM at 96K ctx.

**Conclusion.**

- **Short context (≤16K): expert-pool 64 + f16 KV** is the current best on this box — 19.2 t/s, a +28% decode gain over the q8 baseline, available from the known-good build with two flags (`-ctk f16 -ctv f16`), no code changes.
- **Deep context (64K–90K): expert-pool 64 + q8_0 KV** stays — f16 brings nothing at 64K and OOMs at 96K ctx on 12 GB (exact f16 ceiling between 64K and 96K unmapped).
- **The primary identified deep-context bottleneck is the QSA indexer (O(n_kv) per token)**, with dense full-KV attention and other depth-dependent work remaining possible contributors. The attention-side sparse path is unreachable for this model's (512, 256) QSA head config in current upstream (kernel whitelist + dispatch gate); making it reachable is kernel-development work (a new sparse MMA variant for (512,256) plus dispatch changes), not a port. The indexer-side fix (top-k over compressed index keys instead of per-token full-n_kv scoring) is the open problem.
- Candidate next experiments: (1) confirm/refute the one-shot 12.6 t/s on the experimental binary's dense-f16 path at 64K (n=3), (2) the q8-compact KV branch as the only remaining route to 90K/128K on 12 GB, (3) an upstream note that the sparse-MMA whitelist doesn't cover (DKQ=512, DV=256) QSA heads.
