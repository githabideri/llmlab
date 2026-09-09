# Model Profiles

This directory contains per-model documentation for local LLMs tested in llmlab.

## Purpose

Experiments in `reports/` are dated raw logs. This directory **distills findings** into reusable model profiles that document:

- Performance characteristics (speed, VRAM, context)
- Known issues and failure modes
- Recommended configurations
- Hardware requirements

## In production (2026-09)

Status: **primary** · **multi-slot + vision** · **on-demand**. Every other card in this directory is a frozen test record; a card without a link below means the model is not in production use.

- [Qwen3.8-27B (dual RTX 3090)](qwen3.8-27b-rtx3090.md) — **primary** — vLLM 0.28.0 TP2, MTP k=3, fp8 KV, 256K, vision (since 2026-09-08)
- [Qwen3.6-35B-A3B](qwen3.6-35b-a3b.md) — **multi-slot + vision** — interim host: single-3060 secondary box (Q4_K_XL MTP variant, 128K) since 2026-09-08, when its dual-3060 home on the primary box was dismantled; also served by the single-3060 backup box in its backup window ([2026-06-30 report](../reports/2026-06-30-qwen3.6-35b-a3b-mtp-single-3060.md))
- ~~[Qwen3.8-27B-Uncensored (Dual RTX 3060)](legacy/qwen3.8-27b-uncensored-dual3060.md)~~ — **no longer served 2026-09-08** — its dual-3060 home was dismantled for the second 3090; card moved to [legacy](legacy/)

Every other card in this directory is a **historic test record** — a frozen point-in-time snapshot that is deliberately not maintained.

## Profile Template

Each model gets a markdown file named after the model (e.g., `nemotron-3-nano-30b-a3b.md`).

See any existing profile for the format, or use this skeleton:

```markdown
# Model Name

## Quick Facts
| Param | Value |
|-------|-------|
| Parameters | ... |
| Quant tested | ... |
| Context | ... |
| VRAM requirement | ... |

## Performance
(speeds, benchmarks)

## Known Issues
(failure modes, quirks)

## Recommended Config
(llama.cpp flags, etc.)

## Changelog
(dated updates)
```

## Contributing

When you discover something about a model:
1. Check if a profile exists
2. If yes: add to the relevant section
3. If no: create a new profile from the template

Keep findings **model-specific**. General llama.cpp tips go in `docs/`.
