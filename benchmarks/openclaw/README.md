# OpenClaw Parcours (v1)

A small, OpenClaw‑specific benchmark ladder focused on tool reliability, doc comprehension, and deterministic outputs.

## How to use
- Run cases in order (L0 → L4).
- Each case lists required tools and expected artifacts.
- Use **wttr.in** for weather (no API key).

## Cases
- **L0** read/write sanity
- **L1** config summary
- **L2** offline config patch + JSON validation
- **L3** llama‑bench parse + % drop
- **L4** tool chain + message dry‑run
- **DOC‑QA** answer with citations

## Output
Artifacts should be written under `benchmarks/openclaw/artifacts/`.
