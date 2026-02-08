# LocalBot Commands

These are OpenClaw’s LocalBot helper commands (via `localbot-commands` plugin).

## Working commands
- `/lbh` — help
- `/lbs` — status (active endpoint + model)
- `/lbe` — list endpoints + probe results
- `/lbp` — quick perf benchmark
- `/lbn [room]` — reset LocalBot session (room‑scoped)

## Architecture (sanitized)
```
config/inference-endpoints.json  → endpoint registry (llama-cpp, vLLM, Ollama)
config/localbot-models.json      → model metadata (speed, context, VRAM)
plugins/localbot-commands/       → plugin implementation
```

## Notes
- `message_received` hook fires **after** command handlers, so room auto‑detection is limited.
- `publicReset` can be enabled for specific rooms.
