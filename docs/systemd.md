# LLM Serving — systemd Unit Reference

**Services (current):**
- `vllm-dual.service` — Qwen3.8-27B, **vLLM 0.28.0 tensor-parallel 2** (production since 2026-09-08, port 8080 since 2026-09-16, both RTX 3090s)
- `gpu-power-limits.service` — 220 W per 3090 (interim since 2026-09-21: GPU 1 overheating, thermal service pending; was 250 W from 2026-09-08, 220 W single-card before that / 115 W 3060s). Default via `GPU_POWER_LIMIT_W`.
- `llama-server.service` — Qwen3.6-35B-A3B MTP, llama.cpp, backup box (port 8080)
- `llama-qfn.service` — Qwen3.8-Flash-Next (Qwen4Exp 125B) MoE hot-cache, llama.cpp fork, backup box (port 8091, on-demand load) — since 2026-09-20
- `llama-dcfr.service` — Qwen3.8-27B 3-bit (ISTA GSQ-RCO) with MTP, D-CFR-patched llama.cpp, **both 3060 boxes** (port 8090, on-demand) — since 2026-09-19
- `llama-mux.service` — per-box model-mux (stdlib Python, port 8081) — exclusive-GPU switching + telemetry authority for the 27B, both 3060 boxes — since 2026-09-19
- `llama-qwen3.8-27b.service` — Qwen3.8-27B, llama.cpp (**disabled, kept on disk** as validated rollback)
- plus historic units (BeeLlama DFlash, old longctx/reference configs)

**Dismantled 2026-09-08 (3060 pair removed from the box):** `llama-server-qwen3.6-vision.service` (dual-3060 35B router, port 8081) stopped + disabled, unit kept; the old single-3090 `llama-vllm-qwen3.8-27b.service` (vLLM 0.27.1, port 8080) disabled, kept as `.bak-*` rollback.

**Logs:** `journalctl -u <service>`

> This file is the **unit reference** (what the units contain). Day-to-day operations — status, restart, rollback, debugging — live in [runbook.md](runbook.md).

---

## vLLM — Qwen3.8-27B (Port 8080, dual RTX 3090, TP2 — current)

**Service:** `vllm-dual.service` — **production since 2026-09-08** (supersedes the single-3090 0.27.1 unit, kept disabled as rollback)  
**Placement:** dedicated LXC on the GPU-server host (Ubuntu 24.04); from the host: `pct exec <lxc-id> -- systemctl status vllm-dual`  
**Runtime:** vLLM **0.28.0** (`syv-ai` patch stack, CUDA 12 venv; first-request FlashInfer JIT needs `CUDA_PATH` set) · **Model:** Qwen3.8-27B W4A16-AutoRound, served as `qwen3.8-27b-dual` + 6 ctx-budget aliases `-160k/-128k/-100k/-80k/-64k/-32k` (same single endpoint, one KV pool)  
**Effective config:** `--tensor-parallel-size 2` (`CUDA_VISIBLE_DEVICES=0,1`) · `--max-model-len 262144` · `--max-num-seqs 16` · `--max-num-batched-tokens 8192` · fp8 KV · MTP `num_speculative_tokens 3` (drafter capped 163,840) · prefix caching · mamba-cache-mode align · `--reasoning-parser qwen3` · `--enable-auto-tool-choice --tool-call-parser qwen3_xml` · `--mamba-ssm-cache-dtype float16` · `--async-scheduling` · `--enable-prompt-tokens-details` · `--limit-mm-per-prompt {"image":{"count":16}}` + `--mm-processor-cache-type shm` (vision **on** — weight-sharded tower) · `VLLM_USE_FLASHINFER_SAMPLER=0` · **keyless**, **port 8080** (2026-09-08→16: 8082; then back to 8080 — see note below)  
**Port pin (drift-proof, 2026-09-16):** `/etc/systemd/system/vllm-dual.service.d/10-port.conf` sets `Environment=PORT=8080`; the start-script default is also 8080. An earlier undocumented drop-in moved the endpoint 8082→8080 without any consumer update, breaking every 8082 reference; the pin plus updated script default and unit description make the port explicit in three places.

**Eviction telemetry (on since 2026-09-16):** vLLM 0.28.0 exposes the prefix-cache eviction histograms (`vllm:kv_block_lifetime_seconds`, `..._idle_before_evict_seconds`, `..._reuse_gap_seconds`) behind the startup flags `--kv-cache-metrics --kv-cache-metrics-sample 0.1` (sampled; default 0.01, we run 10 %). No runtime toggle, so enabling takes one cold restart (3 min 25 s in practice) and drops the in-GPU prefix cache; the idempotent launch-dir script applies it (edit + restart + verify, documented rollback) and is re-runnable. The histograms register at boot and fill only while blocks are evicted (`_count` increments per evicted block; the reuse gap stays 0 until an evicted block is re-requested) — the formerly silent LRU is now measurable. First live data minutes after the 09-16 restart: 3 evictions recorded; prefix-cache hit rate back to ~55 % of queries within minutes.

KV pool: **710,402 tokens** (897 GPU blocks, block size 832 — the 0.28.0 hybrid-SSM block arithmetic; the 776,928 logged at the 09-08 cutover used 1024-token blocks, same bytes) → ≈ 2.71× the 262,144 max as concurrency headroom. Thinking is **on by default per request** — non-reasoning consumers should pass `enable_thinking: false` (same behaviour as the old production). Power: **220 W/card, interim since 2026-09-21** (GPU 1 running hot — 81 °C idle vs 57 °C on the new card; thermal paste + pad service pending). 250 W was the 2026-09-08 setting (+6.2–6.5% prefill at equal stability) and is the documented target again after the repaste; 220 W was the pre-cutover single-card setting.

<details><summary>Historic: single-3090 vLLM 0.27.1 unit (pre-2026-09-08, kept for rollback)</summary>

```ini
[Unit]
Description=vLLM Qwen3.8-27B W4A16 MTP fp8-KV 163k prefix-cache (RTX 3090)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/qwen38-vllm
Environment=HOME=/root
Environment=PORT=8080
Environment=CTX=long
Environment=SPEC=mtp
Environment=MAX_LEN=163840
Environment=PREFIX_CACHE=1
Environment="EXTRA_ARGS=--enable-auto-tool-choice --tool-call-parser qwen3_coder"
ExecStart=/bin/bash /opt/qwen38-vllm/single-user/start_qwen.sh
Restart=on-failure
RestartSec=10
TimeoutStartSec=25min
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

</details>

> ⚠️ **Do not replace the start script wholesale** — it carries the MTP/fp8/13-patch wiring (verified by the recipe's `verify.sh`). Add flags via `EXTRA_ARGS`, then restart (~3 min warm).
>
> **LXC memory:** raised 24 → **40 GiB** on 2026-08-30 during the (failed, reverted) CPU KV-offload experiment; kept in place — harmless, and it prepositions the box for hybrid-aware offload once it lands upstream. **Do not add KV-offload flags** to this model — it's a silent no-op (0% hit rate) on hybrid SSM/GDN architectures; see [the 2026-08-30 offload report](../reports/2026-08-30-vllm-cpu-kv-offload-hybrid-mamba-fails.md). The first long prompt after a restart also pays a one-time JIT warmup storm (FlashInfer nvcc compile if the cache is cold; the 0.28.0 script sets `CUDA_PATH` for exactly this).

---

## llama.cpp units

### Qwen3.8-27B (Port 8080, RTX 3090 — dormant fallback)

**Service:** `llama-qwen3.8-27b.service`  
**Unit:** `/etc/systemd/system/llama-qwen3.8-27b.service`  
**Status:** ⏸ **Dormant fallback** — since 2026-08-21 the 27B endpoint on :8080 was served by vLLM 0.27.1 in a separate LXC (superseded by the dual-3090 TP2 unit on 2026-09-08, which re-occupied :8080 on 2026-09-16); this unit stays on disk as the validated llama.cpp rollback  
**GPU:** RTX 3090 24GB (CUDA0)  
**Context:** 160K, q8_0/q8_0 KV, MTP speculative decoding  
**Cutover:** 2026-08-15 (replaced BeeLlama Qwen3.6-27B; stock llama.cpp 5f754ea); superseded by vLLM 2026-08-21

```ini
[Unit]
Description=llama.cpp Qwen3.8-27B MTP Vision - RTX 3090 (160K ctx, q8 KV)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp-20260815-5f754ea
Environment=CUDA_VISIBLE_DEVICES=0
Environment=LD_LIBRARY_PATH=/opt/llama.cpp-20260815-5f754ea
ExecStart=/opt/llama.cpp-20260815-5f754ea/llama-server \
  --device CUDA0 \
  -m /mnt/models/gguf/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /mnt/models/gguf/qwen3.8-27b/mmproj-BF16.gguf \
  --no-mmproj-offload \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.4 \
  -ngl all \
  -np 1 \
  --ctx-size 163840 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -b 512 -ub 64 \
  --flash-attn on \
  --fit off \
  --jinja \
  --reasoning on \
  --reasoning-preserve \
  --metrics \
  --mmap --mlock \
  --host 0.0.0.0 \
  --port 8080
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

### Qwen3.6-35B-A3B Vision (Port 8081, Dual RTX 3060 — dismantled 2026-09-08)

**Status:** ⛔ **Stopped + disabled 2026-09-08** — the 3060 pair was removed from the box (second RTX 3090 instead); the unit and preset INI are kept on the LXC as the rollback path. 35B-A3B is interim on the secondary box (see [card](../../models/qwen3.6-35b-a3b.md)). Config below frozen as found.

**Service:** `llama-server-qwen3.6-vision.service`  
**Model:** Qwen3.6-35B-A3B-UD-IQ4_XS (MTP variant) + vision F16 — mainline llama.cpp, `/opt/llama.cpp-mainline` (historical config from the 2026-08-27 optimization campaign, [report](../reports/2026-08-27-qwen3.6-35b-a3b-dual-3060-optimization.md))  
**GPU:** 2× RTX 3060 12GB (tensor-split 50,50)  
**Context:** 256K, 2 parallel slots, MTP n=3 (acceptance 0.93–0.98)

Verified core of the live unit (`systemctl cat` for the full file):

```ini
[Service]
ExecStartPre=/usr/local/sbin/wait-for-8080-health   # gates on the 27B endpoint health before start
ExecStart=/opt/llama.cpp-mainline/build/bin/llama-server \
  --device CUDA0,CUDA1 \
  -m /mnt/models/gguf/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --mmproj /mnt/models/gguf/qwen3.6-35b-a3b-mtp/mmproj-F16.gguf \
  --no-mmproj-offload \
  --image-max-tokens 1024 \
  --host 0.0.0.0 \
  --port 8081 \
  -c 262144 \
  --parallel 2 \
  --split-mode tensor \
  --tensor-split 50,50 \
  --cache-type-k q8_0 \
  --cache-type-v q4_0 \
  --batch-size 2048 \
  --ubatch-size 1024 \
  --flash-attn on \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --jinja \
  --metrics \
  --cache-prompt \
  --cache-ram 2048
```

**Drop-in:** a sequence/QOS drop-in previously gated this unit on the 27B service; since the 27B moved to vLLM in a separate LXC (2026-08-21), those `Requires=`/`After=` lines are commented out — the `ExecStartPre` health poll is the live gate. Re-engage the drop-in only when rolling 27B back to llama.cpp.

### Qwen3.6-35B-A3B-MTP (Port 8080, RTX 3060, backup box)

**Host:** backup / inference box (LXC on a Proxmox host)  
**Service:** `llama-server.service`  
**Unit:** `/etc/systemd/system/llama-server.service`  
**Status:** ✅ Active, enabled  
**Deployed:** 2026-07-02  
**Model:** Qwen3.6-35B-A3B-UD-Q4_K_XL (22 GB, MoE, MTP)  
**GPU:** RTX 3060 12GB, CUDA 13.1, llama.cpp b9850  
**Context:** 128K, hybrid CPU+GPU offload

```ini
[Unit]
Description=llama.cpp Qwen3.6-35B-A3B-MTP (RTX 3060, 128K ctx)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp
Environment=LD_LIBRARY_PATH=/opt/llama.cpp/build/bin
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/opt/llama.cpp/build/bin/llama-server \
  -m /mnt/models/qwen3.6-35b-a3b-mtp-q4/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --mmproj /mnt/models/qwen3.6-35b-a3b-mtp-q4/mmproj-F16.gguf \
  --no-mmproj-offload \
  --image-min-tokens 1024 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 --n-cpu-moe 28 \
  -c 131072 \
  -ctk q8_0 -ctv q4_0 \
  -b 4096 -ub 2048 \
  --flash-attn on \
  --no-mmap \
  -np 1 \
  --jinja \
  --reasoning-preserve \
  --host 0.0.0.0 \
  --port 8080 \
  --metrics
Restart=on-failure
RestartSec=10
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

**VRAM:** ~11.7 GiB idle / ~11.8 peak at 128K (`-ub 2048` since 2026-08-28; ~430 MiB headroom) | **CPU RAM:** ~15–18 GiB of the LXC limit (host has 48 GB)

**Update (2026-09-04 / 2026-09-19 — router mode):** the unit now runs the llama.cpp **router** instead of a single model: `--models-preset <box-preset-ini> --models-max 1 --no-models-autoload` (same shape on the secondary GPU server's clone of this box). The preset's resident section is the 35B above (`load-on-startup`); on-demand sections are loaded with `POST /models/load` and evict each other (one model in VRAM at a time). Since 2026-09-19 the per-box entry point is the **model-mux on port 8081** (below), which owns the 35B ↔ 27B exclusive switch; the INI above is the pre-router shape, kept as the flag reference.

### Qwen3.8-27B 3-bit D-CFR (Port 8090, RTX 3060, both 3060 boxes, on-demand) — since 2026-09-19

**Host:** secondary GPU server + backup/inference box (LXC on Proxmox)
**Service:** `llama-dcfr.service`
**Unit:** `/etc/systemd/system/llama-dcfr.service`
**Status:** ✅ Active (enabled; starts an empty router; the model loads on demand — it must not auto-load, it would OOM against the resident 35B)
**Model:** Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp (ISTA DASLab 3-bit GGUF, native MTP head) + mmproj BF16 on CPU
**GPU:** RTX 3060 12 GB, CUDA 13.1
**Build:** llama.cpp `925e1179` + the **D-CFR** patch ([GDN transactional replay](https://github.com/kadenball/qwen38-27b-rtx3060-dcfr) — removes the MTP draft's redundant recurrent-state copies, which is what makes 64K fit at 3-bit)
**Config:** 64K ctx (the MTP ceiling — 67K first-token-OOMs in every variant), MTP n-max 2 (accept ~60%), `-b 256 -ub 128`, q4_0/q4_0 KV, ISTA instruct sampling (0.7/0.80/20/1.5), no-think, single section, `--models-max 1`
**Measured:** 25–29 t/s decode (22.5–29.8 on the i3-9100 box), prefill ~300–425, 11.7 GB VRAM (577 MiB headroom), vision ~26 t/s
**27B-only preset** — never add the 35B here: D-CFR costs the MoE ~85% prefill

```ini
[Unit]
Description=llama.cpp Qwen3.8-27B 3-bit D-CFR (RTX 3060, 64K ctx, MTP, on-demand)
After=network.target

[Service]
Type=simple
User=root
Environment=LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:<dcfr-build>/bin
Environment=LLAMA_GDN_TRANSACTIONAL_REPLAY=1
Environment=GGML_OP_OFFLOAD_MIN_BATCH=2
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=<dcfr-build>/bin/llama-server \
  --models-preset /mnt/models/llama-27b-dcfr-preset.ini \
  --models-max 1 --no-models-autoload \
  --device CUDA0 \
  --host 0.0.0.0 \
  --port 8090 \
  --metrics
Restart=on-failure
RestartSec=10
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

**Telemetry caveat:** this build's per-token metrics counters freeze while the GPU decodes (only the decode counter moves). The mux (below, v2.2+) is the **telemetry authority** for this model — it synthesizes `/metrics` from the request-level usage data it injects into proxied requests; hub dashboards show mux-derived numbers for the 27B. An idle 8090 router holds no CUDA context, so it can run alongside the 8080 router harmlessly.

### llama-mux (Port 8081, both 3060 boxes) — since 2026-09-19

**Host:** secondary GPU server + backup/inference box
**Service:** `llama-mux.service` — hand-deployed stdlib-Python script (not in git; any change must be pushed to *both* boxes and md5-compared)
**Role:** the single endpoint per card, fronting the 8080 router (35B resident + parked sections) and the 8090 D-CFR service (3-bit 64K MTP). The one 12 GB card holds one big model at a time; the mux is what makes the roster honest:

- `GET /v1/models` / `GET /models` list exactly the roster (resident 35B + 3-bit 64K-MTP 27B).
- `POST /v1/chat/completions` routes by model name and runs `ensure_loaded()` first: unload the other backend's model → poll `nvidia-smi` to <500 MiB → load → poll until genuinely loaded (no more optimistic "success" loads) → proxy (chunked, SSE-safe, streaming). `POST /models/load|unload` is orchestrated the same way. Every switch is logged with VRAM + timing. A switch runs ~30–60 s per direction (v2.4, 2026-09-20, fixed the 27B→35B direction: the v2.3 unload POST had its URL/body arguments swapped and silently assumed success).
- **v2.2+ telemetry authority** for the 27B (see the caveat above): it injects `stream_options: {include_usage: true}` into every proxied streaming request and synthesizes the D-CFR model's `/metrics` from per-request token counts + the live decode counter; the 35B path stays a pure passthrough. The hub polls `:8081/models` and issues loads/unloads through it (202 "switching" — the mux does not hold the control connection for the whole switch).

```ini
[Unit]
Description=llama-mux — per-card model multiplexer (RTX 3060 exclusive switching)
After=network.target llama-server.service llama-dcfr.service

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 <mux-script>/llama-mux.py --port 8081
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

> The 125B Flash-Next model (backup box, port 8091) joins the same exclusive regime: a request for it evicts whatever is resident, and vice versa.

### Qwen3.8-Flash-Next / Qwen4Exp MoE hot-cache (Port 8091, RTX 3060, backup box, on-demand) — since 2026-09-20

**Host:** backup / inference box (LXC on a Proxmox host)
**Service:** `llama-qfn.service`
**Unit:** `/etc/systemd/system/llama-qfn.service`
**Status:** ✅ Active (enabled; auto-starts the empty router at boot; the model loads on demand)
**Model:** Qwen3.8-Flash-Next UD-Q2_K_XL (3-shard GGUF, 78.9 GB) — 125 B MoE / ~6 B active + 51 B PLE table streamed from host memory
**GPU:** RTX 3060 12GB, CUDA 13.1, codacus llama.cpp fork `27c54b4b` (base `b10818`) — the stock build predates `qwen4exp` support
**Config:** CPU experts (`--n-cpu-moe 99`) + **64-slot MoE hot cache** (profile-driven; ~5.5 GB VRAM) + q8_0 KV + 64K ctx + `-t 4` (one thread per physical core), no MTP (measured neutral on this CPU-bound card). 80 slots decode faster but OOM on long-prompt prefill; 128K ctx won't load at 64 slots.
**Measured:** ~14.3–14.9 t/s decode warm, 48–53 t/s at 9.6K-token prefill (40 GB-RAM tier), GPU 48 W in decode. First load ~1–2 min; first minutes after a 35B↔125B mux switch run 5–9 t/s until the PLE/expert page cache re-fills.
**Switching:** the box's model-mux (port 8081) owns exclusive-GPU access: loading QFN evicts the resident 35B and vice versa (~60–120 s round trip). The router uses `--moe-cache-profile /mnt/models/qfn-profile/q2-merged.csv` (307,776 expert-access rows, 12 traces) + `--moe-cache-slots 64` on the CLI; placement flags live in the preset INI (`/mnt/models/qwen38-qfn-preset.ini`).
**RAM:** the LXC soft limit is 40 GB (bumped from 32 on 2026-09-20; the video's "full-speed prompt reading" tier — 20 GB shows a prefill cliff).

```ini
[Unit]
Description=llama.cpp Qwen3.8-Flash-Next (Qwen4Exp 125B, MoE hot-cache, RTX 3060, on-demand)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp-codacus
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
ExecStart=/opt/llama.cpp-codacus/build-fable/bin/llama-server \
  --port 8091 --host 0.0.0.0 \
  --moe-cache-profile /mnt/models/qfn-profile/q2-merged.csv \
  --moe-cache-slots 64 \
  --models-max 1 --model-preset-path /mnt/models/qwen38-qfn-preset.ini
Restart=on-failure
RestartSec=10
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

**VRAM:** 11.4 GiB with model loaded (64K ctx) / ~1 MiB router-only. **CPU RAM:** the 81.7 GB working set leans on the 40 GB page cache (≈29 GB of it held at steady state).

### BeeLlama Qwen3.6-27B (Historic — replaced 2026-08-15)

**Service:** `beellama-qwen3.6-27b.service`  
**Status:** ❌ Disabled, inactive — kept for rollback (unit + model files untouched)  
**GPU:** RTX 3090 24GB (CUDA0)  
**Context:** 160K, DFlash speculative decoding (BeeLlama.cpp b10102)  
**Cutover:** 2026-06-19 → replaced by `llama-qwen3.8-27b.service` 2026-08-15

```ini
[Unit]
Description=BeeLlama Qwen3.6-27B with DFlash (3090, 160K ctx)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/beellama.cpp
Environment=LD_LIBRARY_PATH=/opt/beellama.cpp/build/bin
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/opt/beellama.cpp/build/bin/llama-server \
  --device CUDA0 \
  -m /mnt/models/gguf/qwen3.6-27b/Qwen3.6-27B-Q5_K_S.gguf \
  --mmproj /mnt/models/gguf/qwen3.6-27b/mmproj-Qwen_Qwen3.6-27B-f16.gguf \
  --no-mmproj-offload \
  --spec-draft-model /mnt/models/gguf/qwen3.6-27b-dflash/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-dflash-cross-ctx 1024 \
  -ngl all \
  --spec-draft-ngl all \
  --kv-unified \
  -np 1 \
  -b 2048 -ub 512 \
  --ctx-size 163840 \
  --cache-type-k q5_0 --cache-type-v q4_1 \
  --flash-attn on \
  --jinja \
  --mmap --mlock \
  --reasoning on \
  --chat-template-kwargs '{"preserve_thinking":true}' \
  --temp 0.6 --top-k 20 --top-p 1.0 --min-p 0.0 \
  --host 0.0.0.0 \
  --port 8080
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

### Legacy Service (Rollback)

**Service:** `llama-server-qwen3.6-27b-longctx.service`  
**Status:** Enabled, inactive — kept for boot safety / rollback  
**Model:** Qwen3.6-27B-Q4_K_M (mainline llama.cpp, 204K context)

### Legacy Reference Unit

**Service:** `llama-server.service`  
**Status:** Disabled — old reference config (GLM-4.7-Flash), no longer used

```ini
[Unit]
Description=llama.cpp server with slot persistence
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/llama.cpp
ExecStart=/opt/llama.cpp/build/bin/llama-server \
  --model /mnt/models/gguf/glm-4.7-flash/GLM-4.7-Flash-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 131072 \
  --parallel 1 \
  --slot-save-path /mnt/models/cache/llama-cpp/slots \
  --reasoning-format deepseek \
  --reasoning-budget -1 \
  --flash-attn on \
  --jinja \
  --split-mode layer \
  --gpu-layers 99 \
  --cache-type-k q8_0 \
  --cache-type-v q4_0 \
  --metrics
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/llama-server.log
StandardError=append:/var/log/llama-server.log

[Install]
WantedBy=multi-user.target
```

---

## Key Flags

### Performance
- `--split-mode layer` — Split model across GPUs by layer (critical for dual 3060)
- `--gpu-layers 99` — Offload all layers to GPU
- `--flash-attn on` — Enable flash attention for speed
- `--cache-type-k q8_0` — K cache quantization (1 byte/element)
- `--cache-type-v q4_0` — V cache quantization (0.5 byte/element)

### Context & Slots
- `--ctx-size 131072` — 128K context window
- `--parallel 1` — Single slot (reduces memory pressure)
- `--slot-save-path /mnt/models/cache/llama-cpp/slots` — Persistent slot state

### Reasoning
- `--reasoning-format deepseek` — DeepSeek-style `<think>` tags
- `--reasoning-budget -1` — Unlimited reasoning tokens
- `--jinja` — Use Jinja2 chat templates from model

### Observability
- `--metrics` — **Enable Prometheus metrics at /metrics endpoint** ✅
  - Added 2026-02-24 (was missing before)
  - Exposes: `prompt_tokens_seconds` (pp tok/s), `predicted_tokens_seconds` (tg tok/s)

---

## Management Commands

```bash
# Check status
systemctl status llama-server

# Start/stop/restart
systemctl start llama-server
systemctl stop llama-server
systemctl restart llama-server

# Reload after config changes
systemctl daemon-reload && systemctl restart llama-server

# View logs
journalctl -u llama-server -f
# Or direct file
tail -f /var/log/llama-server.log

# Check health
curl http://localhost:8080/health

# Query metrics
curl http://localhost:8080/metrics
```

---

## Changing Models

1. Edit service file:
   ```bash
   nano /etc/systemd/system/llama-server.service
   ```

2. Update `--model` path

3. Reload and restart:
   ```bash
   systemctl daemon-reload && systemctl restart llama-server
   ```

4. Wait for model load (~30 sec), then check:
   ```bash
   curl http://localhost:8080/health
   ```

---

## Metrics Endpoint

**URL:** http://localhost:8080/metrics

**Key metrics:**
- `llamacpp:prompt_tokens_total` — Total prompt tokens processed
- `llamacpp:tokens_predicted_total` — Total generation tokens
- `llamacpp:prompt_seconds_total` — Cumulative prompt processing time
- `llamacpp:tokens_predicted_seconds_total` — Cumulative generation time
- `llamacpp:prompt_tokens_seconds` — **Current prompt processing speed (tok/s)**
- `llamacpp:predicted_tokens_seconds` — **Current generation speed (tok/s)**

**Example query:**
```bash
curl -s http://localhost:8080/metrics | grep tokens_seconds
# llamacpp:prompt_tokens_seconds 670.5
# llamacpp:predicted_tokens_seconds 45.2
```

---

## Model History

| Date | Service | Model | Notes |
|------|---------|-------|-------|
| 2026-08-15 | `llama-qwen3.8-27b` | Qwen3.8-27B-Q4_K_M + MTP | Production cutover, stock llama.cpp 5f754ea |
| 2026-06-19 | `beellama-qwen3.6-27b` | Qwen3.6-27B-Q5_K_S + DFlash | Production cutover, BeeLlama.cpp b10102 |
| 2026-04-23 | `llama-server-qwen3.6-27b-longctx` | Qwen3.6-27B-Q4_K_M | Mainline llama.cpp, 204K context (now rollback only) |
| 2026-02-24 | `llama-server` | GLM-4.7-Flash Q4_K_XL | Old reference config (disabled) |
| 2026-02-23 | — | ZwZ-4B Q6_K | Vision model test |
| 2026-02-19 | — | Nemotron-30B-A3B IQ4_NL | MoE baseline |
| 2026-02-19 | — | Qwen3-30B-A3B Q4_K_M | Dense 30B test |

---

## Troubleshooting

### Server won't start
```bash
# Check logs
tail -50 /var/log/llama-server.log

# Check GPU memory
nvidia-smi

# Verify model file exists
ls -lh /path/to/model/
```

### OOM / Slow performance
- Reduce `--ctx-size` (e.g., 65536)
- Check `--parallel` (lower = less memory)
- Verify `--split-mode layer` is set
- Monitor with `nvidia-smi` during inference

### Metrics not showing up
- Ensure `--metrics` flag is present in ExecStart
- Reload: `systemctl daemon-reload && systemctl restart llama-server`
- Check endpoint: `curl http://localhost:8080/metrics`

---

## Related Docs
- Model specs: `llmlab/models/*.md`
- Benchmarking: `llmlab/benchmarks/`
- SSH config: `notes/llama-cpp/ssh-and-cuda.md`
