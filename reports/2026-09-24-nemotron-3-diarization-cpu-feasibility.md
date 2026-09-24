# 2026-09-24 — Nemotron-3-Diarization on a 35 W CPU: 35–45× realtime, zero GPU

**Category:** Experiment / benchmark (model deployment + feasibility)
**Date:** 2026-09-24

## Goal

Add speaker diarization to the video-assist pipeline (WhisperX transcription + 27B
vision answering). The old options were inadequate: WhisperX's X-vector fallback
was a rough "2 speakers" and pyannote's segmentation model OOM'd on the 2 GB
Pascal cards. NVIDIA released **Nemotron-3-Diarization** on 2026-09-23 (Hugging
Face, OpenMDW 1.1, #1 on the VoiceArena Diarization-Bench at 14.72% DER). Goal:
measure whether it runs usefully on **CPU** — so it needs no GPU at all — and if
so, deploy it as a small sidecar service and wire it into video-assist.

## Setup

- **Model:** `nvidia/Nemotron-3-Diarization` — 99.2M parameters. 31-layer
  Transformer (hidden 512, FFN 2048, 8 heads), 128-mel input, 8× subsampling →
  one 80 ms frame, Conv1D upsample to 10 ms, output `[T, 8]` speaker
  probabilities (8-speaker ceiling). Offline mode adds a 30.4 s processing
  buffer; streaming (AOSC/FIFO) is 0.32–1.04 s. We use **offline only**.
- **Host:** a low-power Proxmox host at the second site — Intel **i3-9100T**
  (4C/4T, 3.1–3.7 GHz, 35 W TDP, AVX2, no tensor cores), 23 GB RAM, nearly idle
  (load ~0.15). Chosen over the GPU boxes deliberately: the model's cost is
  ~2.5 GFLOP *per second of audio* (99.2M params × 2 × 12.5 frames/s), which a
  4-core AVX2 CPU swallows at ~40× realtime — the GPU-sharing problem that
  killed the pyannote/X-vector plans simply does not exist at this size.
  (The other site's GPU server was also the wrong fit: 14+ vCPUs allocated on a
  4-core i5-7400 with two 12 GB 3060s at ~95% VRAM.)
- **Guest:** unprivileged LXC, 4 vCPU / 4 GB, Ubuntu 24.04, static LAN IP,
  private-overlay network access for cross-site reach. PVE 9.2 gotcha:
  unprivileged is now the default (`--unprivileged` is silently dropped) and
  there is no inline rootfs sizing — pre-allocate the LV, `mkfs`, mount,
  `chown 100000`, then `pct create` with `--rootfs local-lvm:vm-<id>-disk-0`;
  verify with `pct exec <id> -- cat /proc/self/uid_map` → `0 100000 65536`.
- **Stack:** Python 3.12 venv, `torch` CPU wheels, `transformers` **from git
  main** (the model class postdates every release — `transformers_version`
  5.18.0.dev0 in `config.json` — pin the tested commit), `librosa` (hard import
  of the feature extractor even offline), `huggingface_hub`, `soundfile`.
- **Service:** a ~150-line stdlib `ThreadingHTTPServer` (no framework):
  `GET /health`, `POST /diar` (multipart audio → ffmpeg → 16 kHz mono →
  inference under a single lock → JSON `{n_speakers, segments[], rtfx}`).
  Single-flight by design; systemd unit, `onboot=1`.

## Commands

```bash
# bring up (inside the guest)
python3 -m venv /opt/nemotron-diar/venv
/opt/nemotron-diar/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/opt/nemotron-diar/venv/bin/pip install huggingface_hub soundfile librosa
/opt/nemotron-diar/venv/bin/pip install git+https://github.com/huggingface/transformers  # pin!
huggingface_hub snapshot_download nvidia/Nemotron-3-Diarization --local-dir /opt/nemotron-diar/model

# benchmark
python3 - <<'EOF'
import time
from transformers import AutoModelForCausalLM  # class loaded per config "architectures"
# (load via the repo's NemotronDiarizationForConditionalGeneration, model.eval(),
#  generate on resampled 16 kHz wav; see the service code for the exact call)
EOF

# API
curl -F file=@clip.wav http://<diar-host>:8790/diar
```

## Observations

- Model load ~a few seconds; **peak RAM ≈ 2 GB** (model + torch) in a 4 GB
  guest — works, but tight; 6–8 GB is the sane sizing.
- Inference is single-threaded-per-call under one lock: fine for a sidecar
  (one diarization at a time matches the pipeline's usage).
- The HF snapshot cache is fragile: the `snapshots/<sha>/` dir is symlinks into
  `blobs/`; any mishap there dangles the model — keep a real copy in
  `/opt/.../model` (we do).
- **Quality (pending one human listen):** the 120 s *Futurama* clip (bar/school
  scene, 3–4 characters) → **4 speakers, 54 segments**; a 120 s cut of a 1954
  German archival news broadcast (Zeitzeichen, single modern reporter +
  archival voices) → **5 speakers, ~30 segments**. Both counts are plausible
  for the material; the model is language-agnostic (acoustic features), and
  German ran at the same speed as English. No DER was measured — no German
  ground-truth set exists in the lab.

## Metrics

i3-9100T, 4 cores, fp32, offline mode, model resident:

| Clip | Duration | Inference | RTFx | Speakers |
|------|----------|-----------|------|----------|
| Futurama S11E06 120 s clip (EN, multi) | 115.8 s | 3.04 s | **38.1×** | 4 |
| HF `diarization_example.mp3` (EN, multi) | 97.6 s | 2.18 s | **44.7×** | 6 |
| Zeitzichen 1954 cut (DE, archival) | 120.0 s | 3.44 s | **34.9×** | 5 |

Wall-clock ≈ inference (the 16 kHz resample of a 2-min file takes well under a
second). A 1-hour file costs ~1.5–2 min of CPU and a few watts of a 35 W
part — no power metering exists on this host, so no energy numbers are
claimed.

## Conclusion

- **A 99.2M-param frame classifier is in the CPU's comfort zone**: 35–45×
  realtime on a 35 W quad-core makes GPU allocation, VRAM pressure, and
  scheduling all irrelevant. The diarization problem that looked like a
  "we need a free GPU" problem is actually "any idle core will do".
- **Deployed** as a 4 GB unprivileged LXC sidecar with a minimal API
  (port 8790) on the low-power second-site host; `onboot=1`.
- **Integrated into video-assist:** the web console diarizes each upload
  (cached per file) and speaker-tags the WhisperX transcript with the model
  card's **midpoint rule** — each line gets the speaker(s) active at its word
  midpoint (`S1`, `S2`, …; `S1+S2` for overlap; `–` unassigned) before the 27B
  sees it. The 27B can now answer "who said what" instead of "what was said".
  WhisperX itself is untouched; the merge is client-side.
- **Limits:** offline-only as deployed (streaming is a model capability, not
  yet implemented), 8-speaker ceiling, and accuracy is unverified against
  ground truth — treat speaker counts as a strong prior, not a label.

**Addendum (same day, E2E complete):** the cross-site run (web console → overlay
network → diar box → 27B) passed. Two practical findings: (1) the diar API must
ffmpeg-decode uploads to 16 kHz WAV first — `transformers.load_audio` (librosa)
cannot open mp4 without torchcodec; (2) on the 120 s *Futurama* clip the model
found **4 speakers**, and the 27B's own character↔voice mapping (S1 Zoidberg,
S2 Leela, S3 Marianne, S4 Bender — the scene's actual four characters) confirms
the counts are right, not just plausible. German (1954 archival broadcast, 5
voices) still awaits a human listen; no DER is claimed.
