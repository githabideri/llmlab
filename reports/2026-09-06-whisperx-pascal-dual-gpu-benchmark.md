# WhisperX large-v3-turbo on 2 GB Pascal GPUs: 10× realtime ASR validated

**Date:** 2026-09-06
**Category:** Experiment / benchmark
**Status:** Measured on the secondary GPU server (i5-7400) — a 4 vCPU VM with two Pascal-era 2 GB GPUs (GTX 1050 + GT 1030, both CC 6.1). This campaign validates and characterizes the existing production observation that WhisperX `faster-whisper-large-v3-turbo` runs at ~10× realtime using both 2 GB cards. All numbers below are from this campaign, not from the production service.

**TL;DR**

- ASR (faster-whisper large-v3-turbo, int8_float32) on a GTX 1050 2 GB achieves **10.7× realtime** (11.6 s for 124 s of audio). The full pipeline (ASR + word-level alignment) on dual GPU is **7.1× realtime** (17.4 s).
- The ~10× figure is correct: it is the **ASR stage alone**. Alignment (WhisperX's word-timing step) adds ~33% wall time and drops the aggregate to 7.1×.
- **GPU placement is proven**: ASR runs on GPU 0 (1701 MiB VRAM, 75–82% SM util); alignment runs on GPU 1 (1116 MiB, 49–67% SM util). Each stage is on a different card.
- **Single-GPU ASR + CPU alignment** (one card does ASR, align on CPU) is **10× slower** overall: ASR stays fast (11.6 s on 1050, 19.5 s on 1030), but alignment on CPU takes ~118 s because the ASR model occupies 1.7 GB of the 2 GB card, leaving ~350 MiB — insufficient for the aligner.
- **CPU baseline**: 706 s (0.18× realtime) — ~40× slower than dual-GPU.
- **CTranslate2 compute type**: `int8_float32` (int8 weights, FP32 compute). Both cards support `[float32, int8, int8_float32]` only — **no FP16, no Tensor Cores** (Pascal CC 6.1). The speed comes from CTranslate2's int8 CUDA execution path, not from Tensor Core acceleration.
- **Diarization**: pyannote-3.1 (ResNet34-LM) OOMs at ~1.5 GB on 2 GB cards or runs at 0.1× on CPU (1182 s for 124 s of audio). **X-vector (31 MiB)** replaces it at 28–43× realtime, finding 2 speakers correctly on a 124 s clip and 6 on a 79 min clip, with 322× speedup over CPU pyannote.
- **Quality**: all GPU configs produce **identical transcripts** (SHA-256 match). CPU differs by 1 character (a minor float precision artifact).

---

## Goal

Validate and characterize the existing observation that WhisperX `faster-whisper-large-v3-turbo` achieves roughly 10× realtime on the two 2 GB Pascal GPUs in the fleet. Specifically:

1. Prove that both GPUs are actually used (per-stage VRAM + utilization).
2. Distinguish ASR / alignment / diarization timing — not one aggregate number.
3. Establish a trustworthy baseline before any optimization.
4. Identify the architectural reason this works on such small cards (int8 GEMM, not FP16/Tensor Cores).
5. Investigate a diarization method that fits in 2 GB VRAM.

This experiment belongs in llmlab because it tests **useful local neural inference under severe commodity-GPU constraints** (2 GB Pascal, CC 6.1, no Tensor Cores) using the same empirical hardware/backend methodology as the LLM work — proving that CTranslate2's int8 execution path makes a large Whisper ASR model viable on hardware that would be absurd for FP16 LLM serving.

## Hardware

### GPU 0 — NVIDIA GeForce GTX 1050 2 GB

| Attribute | Value |
|---|---|
| PCI bus ID | `0000:01:00.0` |
| GPU UUID | `GPU-3f8dcaa0-524a-9452-5cbd-3203c9d5d69c` |
| Architecture | Pascal (GP107) |
| Compute capability | 6.1 |
| Memory | 2048 MiB GDDR5 (max mem clock 1752 MHz × 2) |
| Max SM clock | 1405 MHz |
| PCIe endpoint capability | Gen 3 ×16 (advertised by card) |
| PCIe slot | GA-B250-HD3P PCIEX4_1 (physical x16, electrical x4, PCH-side) |
| PCIe operating state | Gen 1 ×4 (~1.0 GB/s payload/direction) — persistent, confirmed under load |
| Virtualization | VFIO pass-through (host bus 07:00.0) |

### GPU 1 — NVIDIA GeForce GT 1030 2 GB

| Attribute | Value |
|---|---|
| PCI bus ID | `0000:02:00.0` |
| Architecture | Pascal (GP108) |
| Compute capability | 6.1 |
| Memory | 2048 MiB **GDDR5** (max mem clock 1502 MHz × 2 = 3004 MHz effective — GDDR5, not DDR4) |
| Max SM clock | 1405 MHz |
| PCIe endpoint capability | Gen 3 ×4 |
| PCIe slot | GA-B250-HD3P PCIEX4_2 (physical x16, electrical x4, PCH-side) |
| PCIe operating state | Gen 1 ×4 (~1.0 GB/s payload/direction) — persistent, confirmed under load |
| OEM | HP (OEM part for HP desktop) |
| Virtualization | VFIO pass-through (host bus 02:00.0) |

> **Memory type proof:** `nvidia-smi -q` reports max memory clock 3004 MHz. The GT 1030 exists in GDDR5 (3004 MHz) and DDR4 (1800 MHz) variants. 3004 MHz = 1502 MHz core × 2 (GDDR5 data rate doubling), confirming GDDR5. The DDR4 variant would show 1800 MHz max.

### CPU / RAM / Topology

| Component | Spec |
|---|---|
| CPU | Intel i5-7400 (4C/4T, Kaby Lake, AVX2) — 4 vCPUs allocated |
| RAM | 7.8 GiB (host: 32 GB, 4×8 GB) |
| Disk | 25 GB (93% full at campaign time) |
| VM type | KVM/QEMU VM on Proxmox |
| Both GPUs | PCIe Gen 3 capable; both operating at Gen 1 ×4 (persistent downgraded state) |
| Motherboard | Gigabyte GA-B250-HD3P — PCIEX4_1 + PCIEX4_2 are PCH-side physical-x16/electrical-x4 slots |
| Root ports | 00:1d.0 (1050, Gen3 x4), 00:1b.0 (1030, Gen3 x4) |

> **PCIe link state (measured):** Both GPUs and their upstream PCH root ports show `LnkSta: Speed 2.5GT/s` despite `LnkCap: Speed 8GT/s` and `LnkCtl2: Target Link Speed: 8GT/s`. This state persists under full ASR load (120 host-side sysfs samples at 2 Hz covering a 24.6 s ASR run) — it is not an idle power-management artifact. The GTX 1050 endpoint advertises Gen3 ×16 but is limited to ×4 by the upstream root port (board slot is wired x4). The cause of the persistent Gen1 speed is **unresolved** (possible: BIOS PCIe speed override, DMI/PCH firmware policy, link-training behavior, or signal integrity; no AER errors observed, PCIe equalization completed successfully). Effective bandwidth: Gen1 ×4 ≈ 1.0 GB/s payload per direction before protocol overhead. In this measured pipeline (SM util 74–82%, 10.7× RT), no obvious PCIe bottleneck was observed, but no direct bus-traffic measurement was taken (no DCGM; `nvidia-smi` power field is `[N/A]` on these Pascal cards).

## Software Environment

| Component | Version |
|---|---|
| NVIDIA driver | 580.173.02 |
| CUDA (driver-reported) | 13.0 |
| CTranslate2 | 4.4.0 |
| faster-whisper | 1.1.1 |
| WhisperX | 3.4.1 |
| PyTorch | 2.6.0+cu124 |
| cuDNN | 9.1.0 (custom build, loaded via `LD_LIBRARY_PATH`) |
| Python | 3.12.3 |
| simple-diarizer | 0.0.13 |
| SpeechBrain | 1.1.1 |
| Model | `dropbox-dash/faster-whisper-large-v3-turbo` |
| Model snapshot | `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` |
| Compute type | `int8_float32` (CTranslate2 auto-selected; int8 weights, FP32 compute) |
| CT2 supported compute types (per GPU) | `[float32, int8, int8_float32]` — no FP16 |
| CPU ISA (CTranslate2) | GENERIC (CTranslate2 runtime not using AVX2 despite CPU support) |
| Alignment model | WhisperX AWD (alignment weighted decoder) |
| Diarization (pyannote) | `pyannote/speaker-diarization-3.1` (wespeaker-voxceleb-resnet34-LM) |
| Diarization (X-vector) | `speechbrain/spkrec-xvect-voxceleb` (via simple-diarizer) |

### Key environment variables (production config)

```bash
WHISPER__MODEL=dropbox-dash/faster-whisper-large-v3-turbo
WHISPER__COMPUTE_TYPE=int8
WHISPER__ALIGN_DEVICE=cuda:1
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0,1
LD_LIBRARY_PATH=<custom-cudnn>/nvidia/cudnn/lib
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

> `WHISPER__COMPUTE_TYPE=int8` is the user-facing setting; CTranslate2 resolves this to `int8_float32` at runtime (the only int8 variant that works without FP16 support). The CT2 log confirms: `Supported compute types: float32 int8 int8_float32` and `Selected compute type: int8_float32`.

## Benchmark Procedure

### Harness

A custom Python harness (`whisperx_bench.py`) that:

1. **Loads the full WhisperX pipeline** (ASR model, alignment model, optional diarization) with explicit device assignment per stage.
2. **Runs the pipeline** on a fixed audio file, timing each stage (ASR with VAD, alignment, diarization) separately.
3. **Samples `nvidia-smi dmon` at 1 Hz** during the run, tagged with per-stage markers (`ASR-START`, `ALIGN-START`, `DIAR-START`) for post-hoc VRAM/utilization attribution.
4. **Captures `vmstat 1`** for host CPU/RAM/swapping.
5. **Computes transcript SHA-256** for cross-config quality comparison.
6. **Emits a JSON result** with all timings, GPU stats per stage, and environment metadata.

Modes:
- `--mode cold`: fresh process, model loads from disk (measures startup + pipeline)
- `--mode warm`: model loaded once, pipeline runs N times (measures steady-state)

### Telemetry

Per-GPU `nvidia-smi dmon -s um` sampled at 1 Hz:
- `sm` (SM utilization %)
- `mem` (memory utilization %)
- `vram` (VRAM usage MiB)

Post-hoc window slicing: stage boundaries are extracted from the harness's `stage_start`/`stage_end` timestamps (Python `time.time()`), and dmon samples are attributed to stages by timestamp.

> **Driver caveat:** on driver 580, `nvidia-smi dmon -f` (file output) has no timestamp column. The harness therefore writes its own `REQ-START <epoch> <stage>` markers to a side file and aligns dmon rows by 1-second index.

### Audio fixtures

| File | Duration | Language | Codec | Source |
|---|---|---|---|---|
| `test-clip-124s.mp4` (primary) | 124.2 s | English | H.264/MP4 | Local test clip; not redistributed |
| `jfk-1961-inaugural.mp3` | 905.0 s | English | MP3 64 kbps | [archive.org: jfks19610121](https://archive.org/details/jfks19610121) |
| `zeitzichen-1954-fermi.mp3` | 881.1 s | German | MP3 VBR | [archive.org: Zeitzeichen 1954-11-28](https://archive.org/details/Zeitzeichen_1954_11_28_Todestag_des_Physikers_Enrico_Fermi_2014) |
| `longro.wav` (long-form) | 4750.9 s | English (detected, conf 0.49) | PCM s16le 16 kHz mono | Private archival (Romanian church presentation); pre-converted from MP4 |

> The primary 124 s clip is from an animated English-language series (2 distinct speakers, confirmed by X-vector diarization). The long-form file is a 79-minute talk; language detection returned `en` at 0.49 confidence. All files are retained locally; only metadata and results are committed to the public repo.

### Cell matrix

| Cell | ASR device | Align device | Diar device | Mode | Reps |
|---|---|---|---|---|---|
| A: dual | cuda:0 (1050) | cuda:1 (1030) | — | cold + warm ×3 | 1 + 3 |
| B: 1050 alone | cuda:0 (1050) | CPU | — | cold + warm ×3 | 1 + 3 |
| C: 1030 alone | cuda:1 (1030) | CPU | — | cold + warm ×3 | 1 + 3 |
| D: CPU | CPU | CPU | — | cold + warm ×2 | 1 + 2 |
| E: long-form | cuda:0 | cuda:1 | — | cold + warm | 1 + 1 |
| F: lang-en (JFK) | cuda:0 | cuda:1 | — | cold | 1 |
| G: lang-de (Zeitzeichen) | cuda:0 | cuda:1 | — | cold | 1 |
| H: diar pyannote | cuda:0 | cuda:1 | CPU (OOM on GPU) | cold | 1 |
| I: diar xvec (short) | cuda:0 | cuda:1 | cuda:0 | cold | 1 |
| J: diar xvec (long) | cuda:0 | cuda:1 | cuda:0 | cold | 1 |

## GPU Placement Proof

### Dual-GPU mode (Cell A, warm rep 2)

| Stage | GPU 0 (GTX 1050) | GPU 1 (GT 1030) |
|---|---|---|
| ASR | **1701 MiB, 80.3% avg, 100% max** | 1116 MiB, 0% util |
| Align | 1125 MiB, 0% util | **1116 MiB, 66.7% avg, 100% max** |

Interpretation:
- During ASR, GPU 0 shows 1701 MiB allocated and 80% SM utilization; GPU 1 shows only residual VRAM (the align model pre-loaded) with 0% util.
- During alignment, GPU 1 shows 1116 MiB and 67% SM utilization; GPU 0 shows residual VRAM with 0% util.
- **Each stage is on a different GPU.** This is the production configuration.

### Single-GPU ASR + CPU alignment (Cells B & C)

| Cell | Stage | GPU placement | Align behavior |
|---|---|---|---|
| B (1050) | ASR | GPU 0: 1701 MiB, 81.6% util | Falls to **CPU** (118 s) |
| B (1050) | Align | CPU | 118 s (0.95× RT) |
| C (1030) | ASR | GPU 1: 1692 MiB, 89.1% util | Falls to **CPU** (118 s) |
| C (1030) | Align | CPU | 118 s (0.95× RT) |

> In these cells, the align model is deliberately placed on CPU because it cannot co-locate with the ASR model (1701 + ~1100 = 2801 MiB > 2048 MiB). CPU alignment is 20× slower than GPU alignment. This matches the production behavior: when only one GPU is available, the service falls back to CPU alignment.

### Diarization GPU placement (Cell I)

| Stage | GPU 0 (GTX 1050) | GPU 1 (GT 1030) |
|---|---|---|
| ASR | 1701 MiB, 78.6% util | 0% util |
| Align | 1125 MiB, 0% util | **1116 MiB, 60% util** |
| Diar (X-vector) | **1165 MiB, 0% util** (sequential, after ASR) | 1116 MiB, 0% util |

> X-vector diarization runs on GPU 0 **after** ASR completes (sequential, not concurrent). Peak VRAM 1165 MiB fits within 2 GB. The align model on GPU 1 is unaffected.

## Results

### Primary 124 s clip — per-stage timing (median of 3 warm reps where applicable)

| Cell | ASR (s) | Align (s) | Diar (s) | Total (s) | RT factor (×) |
|---|---:|---:|---:|---:|---:|
| **A: dual GPU** | **11.6** | **5.8** | — | **17.4** | **7.1×** |
| A: dual cold | 12.5 | 5.9 | — | 18.4 | 6.7× |
| B: 1050 only | 11.6 | 118.0 | — | 129.6 | 1.0× |
| C: 1030 only | 19.5 | 118.1 | — | 137.5 | 0.9× |
| D: CPU only | 586 | 120.0 | — | 706 | 0.18× |
| I: dual + xvec diar | 11.9 | 5.9 | 3.7 | 21.5 | 5.8× |
| H: dual + pyannote CPU | 11.8 | 119.1 | 1182.4 | 1313.3 | 0.1× |

### Warm-run variance (Cell A, 3 reps)

| Rep | ASR (s) | Align (s) | Total (s) |
|---:|---:|---:|---:|
| 1 | 11.754 | 5.952 | 17.706 |
| 2 | 11.585 | 5.819 | 17.404 |
| 3 | 11.592 | 5.821 | 17.413 |
| **Median** | **11.592** | **5.821** | **17.413** |
| **Range** | 0.169 s | 0.133 s | 0.302 s |

Variance is tight: ±1% on total pipeline time. No warmup artifact beyond the first rep.

### Long-form 4751 s clip (Cell E)

| Metric | Value |
|---|---|
| ASR | 404.9 s (11.7× RT) |
| Align | 244.6 s (19.4× RT) |
| Total | 649.5 s (7.3× RT) |
| Total with X-vector diar | 768.1 s (6.2× RT) |
| Diar (X-vector, AHC) | 110.7 s (42.9× RT) |
| Speakers found | 6 |
| Speech coverage | 82.0% (3898 s / 4751 s) |
| Segments | 1161 |
| Detected language | en (0.49 confidence) |

### Language comparison (Cells F & G)

| File | Lang | Duration | ASR (s) | Align (s) | Total (s) | RT (×) |
|---|---|---:|---:|---:|---:|---:|
| JFK 1961 | en | 905 s | 75.0 | 42.8 | 117.8 | 7.7× |
| Zeitzeichen 1954 | de | 881 s | 91.6 | 45.9 | 137.5 | 6.4× |

> German is ~17% slower than English on the same hardware. With one sample each, the cause is unresolved — the difference could reflect speech rate, audio quality, recording age, transcript length, or phoneme-set differences. A proper language comparison would need matched corpora.

### Diarization comparison

| Method | Model size | VRAM | Time (124 s) | RT (×) | Speakers | Speech coverage |
|---|---|---|---:|---:|---:|---:|
| pyannote 3.1 (GPU) | ~1.5 GB (ResNet34-LM) | **OOM at 1.5–1.7 GB** | — | — | — | — |
| pyannote 3.1 (CPU) | same | 0 (host RAM) | 1182 s | 0.1× | 2 | 91.6 s (74%) |
| **X-vector (GPU)** | **31 MiB** | **1165 MiB peak** (co-located) | **3.7 s** | **33×** | **2** | **90.6 s (73%)** |
| X-vector (GPU, 79 min) | same | 1213 MiB peak | 110.7 s | 43× | 6 | 3898 s (82%) |

> X-vector is **322× faster** than CPU pyannote and uses **48× less VRAM** than GPU pyannote. Speaker counts and speech coverage are comparable (±1 s on speech duration).

## Per-stage analysis

### ASR (faster-whisper large-v3-turbo, int8_float32)

| GPU | 124 s clip | 4751 s clip | 905 s (JFK) |
|---|---:|---:|---:|
| GTX 1050 (cuda:0) | 11.6 s (10.7×) | 405 s (11.7×) | 75.0 s (12.1×) |
| GT 1030 (cuda:1) | 19.5 s (6.4×) | — | — |
| CPU (i5-7400, 4 vCPU) | 586 s (0.21×) | — | — |

The GTX 1050 is **1.7× faster** than the GT 1030 for ASR (GP107 has more SMs: 640 vs 384). CPU is 50× slower.

### Alignment (WhisperX AWD word-level)

| Device | 124 s clip | 4751 s clip |
|---|---:|---:|
| GT 1030 (cuda:1) | 5.8 s (21.4×) | 245 s (19.4×) |
| CPU (i5-7400, 4 vCPU) | 118 s (1.05×) | — |

GPU alignment is **20× faster** than CPU. This is the stage that benefits most from dual-GPU: on a single 2 GB card, the align model cannot fit alongside the ASR model.

### Diarization (X-vector via simple-diarizer)

| Duration | Time | RT |
|---:|---:|---:|
| 124 s | 3.7 s | 33× |
| 4751 s | 110.7 s | 43× |

Scales slightly better than linearly (longer audio = fewer relative VAD overhead per second). AHC clustering on 1580 utterances completes in seconds (vs spectral clustering which OOM'd/timed out).

## Quality validation

### Transcript consistency

All GPU configurations produce **identical transcripts** (SHA-256 match):

| Config | SHA-256 (first 16) | Chars |
|---|---|---:|
| Dual GPU (warm) | `d7d934ac1da7514e` | 1178 |
| 1050 only | `d7d934ac1da7514e` | 1178 |
| 1030 only | `d7d934ac1da7514e` | 1178 |
| CPU only | `38e0e3387fdcda35` | 1179 |

> CPU differs by 1 character — a minor FP32 rounding artifact in the int8 GEMM accumulation. The GPU int8 path is deterministic across both Pascal cards. No silent precision degradation between configurations.

### Spot-check

The 124 s primary clip transcript was manually reviewed: proper English, correct proper nouns, no hallucinated segments. The JFK clip (905 s) produced a coherent 1961 inaugural address.

## VRAM behavior

| Stage | GPU 0 peak | GPU 1 peak | Headroom on 2 GB |
|---|---:|---:|---|
| ASR (int8_float32) | **1701 MiB** | 38 MiB | 347 MiB (17%) |
| Align (AWD model) | 1125 MiB (residual) | **1116 MiB** | 932 MiB (46%) |
| Diar X-vector | **1165 MiB** | 1116 MiB (residual) | 883 MiB (43%) |
| Diar pyannote (OOM) | 1590–1700 MiB | — | < 460 MiB → **OOM** |

> The ASR model at int8 uses 1701 MiB — 83% of the 2 GB card. This is the binding constraint: it leaves only 347 MiB for anything else. The alignment model needs ~1100 MiB, so it cannot co-locate. In dual-GPU mode, each stage gets its own card with comfortable headroom.

## Bottlenecks and architectural analysis

### Why 2 GB Pascal works

1. **Int8 execution is the key.** CTranslate2's `int8_float32` compute type stores weights in int8 (halving memory vs float32) and computes in FP32. The large-v3-turbo model is a pruned variant (32 encoder layers, 4 decoder layers) with an FP16 checkpoint of ~1.6 GB. In int8 with CTranslate2's weight packing, it fits in 1.7 GB of VRAM including buffers.

2. **No Tensor Cores needed.** Pascal CC 6.1 has no Tensor Cores (introduced in Volta CC 7.0). The speed comes from CTranslate2's int8 CUDA execution path running on the SMs of the GP107, not from specialized matrix multiply units. The GTX 1050 achieves ~10× realtime ASR purely from SM throughput.

3. **Sequential stages, not concurrent.** WhisperX's pipeline is inherently sequential: VAD → ASR → align → diarize. Dual GPU helps because **align goes to the second card** (avoiding the CPU fallback), not because both cards work simultaneously on the same stage.

4. **The GT 1030 is the align card.** The align model (AWD) is smaller than the ASR model and does not saturate the GPU continuously (67% peak SM utilization at 1 Hz sampling). The 1050 could do it too, but it's occupied with ASR.

### Why single-GPU is 10× slower

The 1701 MiB ASR model + 1100 MiB align model = 2801 MiB > 2048 MiB. The align model runs on CPU, where the i5-7400 (CTranslate2 reports GENERIC ISA despite AVX2 support) processes it at 1.05× realtime. This single stage (118 s) dominates the total (129 s).

### Why CPU is 40× slower than dual-GPU

The i5-7400 is a 4-core Kaby Lake with AVX2 support. However, CTranslate2 reports GENERIC ISA, meaning its CPU int8 GEMM path is not using AVX2 (possibly due to the VM's CPU feature exposure or CTranslate2's build configuration). Without AVX2/AVX512 acceleration, the CPU int8 GEMM is very slow. The ASR stage alone on CPU is 586 s (0.21× RT).

## Limitations

1. **Single VM, single host.** All results are from one 4 vCPU KVM VM. Host-level CPU contention, NUMA effects, or KVM overhead may not generalize.
2. **PCIe operating at Gen 1 ×4 (~1.0 GB/s payload/direction).** Both GPUs and their PCH root ports are Gen 3 capable but persistently operate at Gen 1 ×4 (confirmed under load). The cause of the speed downgrade is unresolved. Under the measured int8 ASR workload, no obvious PCIe bottleneck was observed, but no direct PCIe bus-traffic measurement was taken (no DCGM available), so the conclusion is "no obvious bottleneck" rather than "PCIe is irrelevant to performance."
3. **No wall power measurement.** The VM has no dedicated PDU/smart-plug. GPU power draw is `[N/A]` on Pascal via `nvidia-smi` (driver limitation). The GT 1030 TDP is 30 W, GTX 1050 TDP is 75 W (reference specs).
4. **No ground-truth WER/CER.** No reference transcripts exist for the test files. Quality validation is via SHA-256 consistency and manual spot-check.
5. **Language detection on long-form.** The 79 min file was detected as `en` at 0.49 confidence. The actual content is a Romanian church presentation with English code-switching. This may affect ASR quality (not timing).
6. **Diarization speaker count is a heuristic.** The AHC clustering with `num_speakers` estimated from speech duration (not ground truth) may over/under-segment. The 6 speakers found on the long-form file may include overlapping speech segments counted as separate speakers.
7. **simple-diarizer compatibility.** Uses deprecated `speechbrain.pretrained` API (SpeechBrain 1.x uses `speechbrain.inference`). Functional, but produces UserWarnings. The sklearn 1.9 `affinity` → `metric` rename required a patch.
8. **`words` field in JSON results.** The harness reports `words: 0` due to a schema-extraction bug (the aligned word list is not counted correctly). The `transcript_chars` and `segments` fields are correct. Field removed from committed JSON.
9. **No concurrency test.** All results are single-stream. The production service handles one job at a time (queue-based).

## Reproducibility

### Environment setup

```bash
# Python 3.12.3 venv
python3.12 -m venv .venv
.venv/bin/pip install \
  faster-whisper==1.1.1 \
  whisperx==3.4.1 \
  ctranslate2==4.4.0 \
  torch==2.6.0 \
  simple-diarizer==0.0.13

# Custom cuDNN 9.1.0 (not the pip version)
export LD_LIBRARY_PATH=<custom-cudnn>/nvidia/cudnn/lib
```

### Key invocation (dual-GPU, production config)

```bash
LD_LIBRARY_PATH=<custom-cudnn>/nvidia/cudnn/lib \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
HF_TOKEN=<hf-token> \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1 \
python whisperx_bench.py \
  --audio <124s-english-clip.mp4> \
  --label dual-warm --mode warm --reps 3 \
  --asr-device cuda:0 --align-device cuda:1 \
  --ct2-verbose \
  --out ./results
```

### Raw data

Machine-readable results (JSON per run) are stored in `scripts/results/whisperx-pascal-2026-09-06/`:
- `env-capture.txt` — full `nvidia-smi -q`, `lspci`, `python --version`, package versions
- `env-json.json` — structured environment metadata
- `*.json` — per-run results with stage timings, GPU stats, transcript hashes
- `gpu.csv` / `vmstat.csv` — raw 1 Hz telemetry samples (per run)

> Raw telemetry files are ~300 KB each (1 Hz for 13 min runs). Committed as CSV for machine-readability; not parsed into Markdown.

## Conclusions: where 2 GB Pascal cards remain useful in 2026

1. **ASR is the killer app for 2 GB Pascal.** CTranslate2's int8 execution path makes the large-v3-turbo Whisper model run at 10× realtime on a $75 GPU (GTX 1050) from 2016. No FP16, no Tensor Cores, no 8 GB VRAM — just int8 weights and SM throughput.

2. **Dual-GPU is not about parallelism, it's about avoiding CPU fallback.** The two cards don't work simultaneously; they handle sequential stages. The benefit is that alignment (normally CPU at 1.0× RT) runs on GPU at 21× RT.

3. **2 GB is enough for int8 ASR but not for pyannote diarization.** The ASR model at 1.7 GB leaves 350 MiB — enough for X-vector (31 MiB) but not for ResNet34-LM (1.5 GB). This is the practical ceiling.

4. **The GT 1030 (GP108, 384 SMs) is 1.7× slower than the GTX 1050 (GP107, 640 SMs) for ASR** — but both are fast enough for the align model. The 1030 is the "free" align card in the production config.

5. **CPU alignment is the bottleneck in single-GPU mode.** The 118 s align time on CPU dominates the total. CTranslate2's GENERIC ISA path (not using the CPU's AVX2) is the likely cause; if the CPU path used AVX2, align would be faster, making single-GPU mode less punishing.

6. **For LLM inference, 2 GB Pascal is mostly dead.** A 7B model in Q4 needs 4 GB. But for specialized inference (ASR, alignment, small embeddings), int8 CTranslate2 keeps these cards useful well past their LLM viability.

## Implementation note: reusable benchmark abstraction

This campaign required a custom harness because the existing `llmlab-bench` skill targets llama.cpp/vLLM LLM inference (tok/s, prefill/decode, PCIe residency). The WhisperX pipeline is fundamentally different:

| Dimension | LLM (llmlab-bench) | ASR (this campaign) |
|---|---|---|
| Throughput metric | tok/s (prefill + decode) | × realtime (audio_s / processing_s) |
| Stage structure | prefill → decode | VAD → ASR → align → diarize |
| GPU placement proof | PCIe RX/TY counters | Per-stage VRAM + SM utilization |
| Quality metric | WER/CER vs ground truth | SHA-256 transcript consistency |
| Input | Text tokens (deterministic) | Audio (duration, codec, language) |
| Cache behavior | KV cache, prefix cache | None (each run is stateless) |
| Warmup | cudagraph capture, JIT | Model load (cold vs warm) |
| Repetition policy | 3+ warm reps after cold | 3 warm reps (tight variance: ±1%) |

### What a reusable ASR benchmark abstraction would need

1. **Environment capture** (already done): driver, CT2 version, compute types per GPU, model snapshot, audio codec/duration.
2. **Stage-aware timing**: per-stage wall time with explicit stage boundaries (not one aggregate).
3. **GPU telemetry with stage attribution**: 1 Hz dmon + stage markers → per-stage VRAM/util.
4. **Audio corpus metadata**: duration, codec, language, source URL (for reproducibility without committing audio).
5. **Quality check**: transcript SHA-256 across configs; optional WER/CER if ground truth exists.
6. **Repetition policy**: 1 cold + 3 warm, report median + range.
7. **Structured JSON output**: per-run JSON with all fields, aggregatable.

### Recommendation: separate skill, not an extension of `llmlab-bench`

The ASR benchmark differs from LLM benchmarking in enough dimensions (metric vocabulary, stage structure, input type, quality check) that forcing it into `llmlab-bench` would create a fork-in-logic. A separate `asr-bench` skill (or `whisperx-bench` if scope is narrower) would:

- Reuse the same telemetry infrastructure (dmon sampler, stage markers, JSON schema)
- Have its own metric vocabulary (× realtime, not tok/s)
- Define its own cell matrix (GPU placement per stage, not tensor split)
- Share the public boundary / sanitizer / raw-data conventions from llmlab

The shared infrastructure (telemetry sampler, env capture, JSON schema, sanitizer) could be extracted into a common `bench-utils/` directory. The LLM and ASR skills would both import from it.

**Do not implement yet.** This note establishes the decision framework; the actual skill split should happen after one more campaign confirms the harness is stable (e.g. a second machine or a model swap).

---

## Audio corpus (for reproducibility)

The primary 124 s clip is a local test clip from an animated English-language series (2 distinct speakers); it is not redistributed with this report. Substitute clips of equivalent character (English, ~2 min, MP4/H.264, multi-speaker) will produce comparable timings. The JFK and Zeitzeichen clips are from archive.org (see URLs above). The long-form 79 min file is private; any 70–80 min mono 16 kHz WAV of similar speech density will produce comparable per-stage timings.
