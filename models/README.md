# Model Profiles

This directory contains per-model documentation for local LLMs tested in llmlab.

## Purpose

Experiments in `experiments/` are dated raw logs. This directory **distills findings** into reusable model profiles that document:

- Performance characteristics (speed, VRAM, context)
- Known issues and failure modes
- Recommended configurations
- Hardware requirements

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
