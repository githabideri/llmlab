# Model Profiles

This directory contains per-model documentation for local LLMs tested in llmlab.

## Purpose

Experiments in `reports/` are dated raw logs. This directory **distills findings** into reusable model profiles that document:

- Performance characteristics (speed, VRAM, context)
- Known issues and failure modes
- Recommended configurations
- Hardware requirements

## In production (2026-08)

- [Qwen3.8-27B (RTX 3090)](qwen3.8-27b-rtx3090.md) — vLLM 0.27.1, MTP k=3, 160K, text-only
- [Qwen3.6-35B-A3B](qwen3.6-35b-a3b.md) — dual 3060 (tensor 50/50, MTP n=3, 256K ×2); the same model also serves the single-3060 backup box as a llama.cpp MTP endpoint ([2026-06-30 report](../reports/2026-06-30-qwen3.6-35b-a3b-mtp-single-3060.md))

Every other card in this directory is a **historic test record** — a frozen point-in-time snapshot that is deliberately not maintained. A card without a link in the list above means the model is not in production use.

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
