---
id: "2026-09-24-audio-assist"
date: 2026-09-24
category: "infrastructure"
host: "second-site low-power host (i3-9100T, 4 cores)"
tags: [infrastructure, diarization, asr, whisperx, nemotron, homelab]
models: ["nemotron-3-diarization", "faster-whisper-large-v3-turbo"]
---

# audio-assist — a transcription + speaker-diarization console on a 35 W CPU box

## What is this

A homelab service that takes any audio or video file and returns a **word-timed,
speaker-attributed transcript** with a visual viewer and SRT/VTT/TXT/JSON downloads.
It composes two existing inference services — it runs no heavy model of its own:

1. **WhisperX** (faster-whisper `large-v3-turbo` int8, GPU — see
   [WhisperX large-v3-turbo on 2 GB Pascal GPUs](2026-09-06-whisperx-pascal-dual-gpu-benchmark.md))
   produces the aligned word-level transcript.
2. **Nemotron-3-Diarization** (99.2M params, NVIDIA, OpenMDW 1.1 — see
   [2026-09-24 CPU feasibility report](2026-09-24-nemotron-3-diarization-cpu-feasibility.md))
   runs **on CPU on the same box as the console** and produces speaker segments.

The console itself is a ~400-line **pure-stdlib Python HTTP service** (no
dependencies beyond ffmpeg): it decodes the upload to 16 kHz WAV, dispatches the
two inference jobs, merges them, and serves the results.

## Why this placement

The diarization model is small (99.2M params, ~2 GB loaded, 35–45× realtime on a
4-core CPU) — cheap enough to run on a 35 W host with zero GPU involvement. Hosting
the console **on the same machine as the diar model** means the only network hop
in the pipeline is the ASR call (private overlay to the GPU box); the diarization
call is loopback. A small unprivileged container on the low-power second-site
host (4 vCPU / 6 GB / 8 GB disk) holds both: the model service (port 8790) and
the console (port 8792). This box has become the homelab's general **audio box**.

## Pipeline

```
upload (mp4/webm/m4a/wav/…)
  └→ ffmpeg → 16 kHz mono WAV
       ├→ WhisperX (remote, GPU box) ──→ word-timed segments + word timestamps
       └→ Nemotron-3-Diarization (loopback, CPU, 99M) ──→ speaker segments [T,8] → 80 ms frames
            └→ merge (client-side rule): a transcript word is assigned to the speaker
               active at the word's midpoint; ≥2 speakers at the midpoint ⇒ overlap
```

Output formats: plain SRT; VTT with speaker cue ids; speaker-tagged TXT
(overlap marked `+`, unassigned `–`); and a JSON containing both the merged
result and the raw model payloads.

## API

- `POST /api/run` — multipart upload (`file`, optional `language`), returns a job id
- `GET /api/jobs/<id>` — stage-by-stage state and per-stage wall time
- `GET /api/result/<id>/{json|srt|vtt|txt|transcript|diar}`
- `GET /health` — status of both upstream services
- `GET /` — web console: upload + language picker, staged progress, embedded
  speaker viewer, download links

## Measured (2026-09-24, 120 s German archival broadcast)

| Stage | Wall time | Note |
|-------|-----------|------|
| decode (ffmpeg) | 2.6 s | |
| WhisperX | 40.5 s | remote GPU box, includes its queueing overhead |
| Diarization | 8.4 s | **includes cold model load** (~5 s); warm runs 3–4 s |
| **Total** | **40.6 s** | |

Result: **5 speakers, 15 lines** — visually confirmed in the viewer (the 2014
retelling has a narrator plus four dramatized roles; no audible overlap in this
clip, which matches the model reporting none). English 120 s end-to-end measures
~25–30 s. Expect a 1-hour podcast to take on the order of 10–15 minutes.

## The speaker viewer

The console's visualization — per-speaker color lanes with a synchronized playhead,
click-to-seek, and the transcript colored per word-midpoint — is a single
self-contained HTML file (no dependencies). It is published as a standalone
tool in [`scripts/speaker-viewer/`](../scripts/speaker-viewer/README.md) and can
render any diarization payload in the same JSON shape (8-speaker ceiling).

## Design notes

- **Midpoint merge rule** — word-level speaker assignment by midpoint is a
  deliberately simple, auditable rule; it degrades gracefully (unassigned words
  render `–`, overlaps render `+`) instead of inventing speakers.
- **Lazy model load** — the diarization service loads the model on first request
  and caches it; health reports "on-demand" until then, so the service is
  *available* even when no model is in memory.
- **Single-flight diarization** — one inference at a time; concurrent console
  jobs queue. Fine for a personal service; a concurrency axis would be the
  first thing to change if this ever got real usage.
- **Batch, not streaming** — WhisperX is a batch job queue and the diarization
  call uses the model's offline configuration (30.4 s context buffer). The
  model natively supports **streaming** mode (0.32–1.04 s latency via its
  AOSC/FIFO chunking — see the model card), but no streaming client exists yet
  on this stack; that is the obvious next step, along with windowed
  (near-realtime) WhisperX on the GPU box.

## Honest limitations

- Diarization quality (DER) on this clip is **unmeasured** — no ground-truth
  labels; visual speaker consistency was the check. English accuracy is well
  supported by the model's training; German is in-model but less well
  validated.
- The 8 GB container rootfs is the real capacity constraint (uploads + 16 kHz
  WAVs accumulate per job); no cleanup policy yet.
- No auth — the service is exposed on a private overlay / LAN only, same
  posture as the WhisperX endpoint.
